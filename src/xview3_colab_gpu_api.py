#!/usr/bin/env python3
"""Authenticated Colab GPU API for the xView3 First Place JIT ensemble."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import secrets
import subprocess
import threading
import time
import urllib.request
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform as transform_coords
from tqdm.auto import tqdm


MODEL_URL = (
    "https://github.com/BloodAxe/xView3-The-First-Place-Solution/"
    "releases/download/1.0/traced_ensemble.jit"
)
MODEL_BYTES = 1_304_877_218
MAX_UPLOAD_BYTES = 90 * 1024 * 1024
MAX_SCENE_PIXELS = 40_000_000
TILE = 2048
STEP = 1536
OBJECTNESS_THRESHOLD = 0.300
VESSEL_THRESHOLD = 0.338
FISHING_THRESHOLD = 0.350
MAX_OBJECTS = 2048
CSV_FIELDS = [
    "detect_scene_row", "detect_scene_column", "objectness_p", "is_vessel_p",
    "is_fishing_p", "is_vessel", "is_fishing", "vessel_length_m",
    "lon", "lat", "source_vh",
]


def download_model(model_path: Path) -> None:
    if model_path.exists():
        if model_path.stat().st_size != MODEL_BYTES or not zipfile.is_zipfile(model_path):
            raise ValueError(f"Existing model has an unexpected size or invalid TorchScript archive: {model_path}")
        return

    partial = model_path.with_suffix(".jit.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > MODEL_BYTES:
        raise ValueError(f"Partial model exceeds expected size: {partial}")
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(MODEL_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = response.status == 206 and offset > 0
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as output, tqdm(
            total=MODEL_BYTES, initial=offset, unit="B", unit_scale=True, desc="xView3 model"
        ) as progress:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                progress.update(len(chunk))
    if partial.stat().st_size != MODEL_BYTES or not zipfile.is_zipfile(partial):
        raise ValueError("Model download was incomplete; restart the Colab API to resume it.")
    partial.replace(model_path)


def _normalize(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = torch.sigmoid(torch.from_numpy((values.astype(np.float32) + 20.0) * 0.18)).numpy()
    normalized[~valid] = 0
    return normalized.astype(np.float32), valid


def _load_scene(
    path: Path,
    vh_band: int,
    vv_band: int,
) -> tuple[np.ndarray, np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError("Input GeoTIFF has no CRS.")
        if not 1 <= vh_band <= source.count or not 1 <= vv_band <= source.count:
            raise ValueError(f"Input GeoTIFF has {source.count} bands; requested VH/VV band is missing.")
        if vh_band == vv_band:
            raise ValueError("VH and VV must use different bands.")
        transform, width, height = calculate_default_transform(
            source.crs, "EPSG:32652", source.width, source.height, *source.bounds, resolution=10
        )
        if width * height > MAX_SCENE_PIXELS:
            raise ValueError(f"Scene is too large: {width * height:,} pixels.")

    target_crs = rasterio.crs.CRS.from_epsg(32652)
    image: list[np.ndarray] = []
    valid = np.ones((height, width), dtype=bool)
    with rasterio.open(path) as source:
        with WarpedVRT(
            source,
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
            nodata=-32768,
            dtype="float32",
        ) as vrt:
            for band in (vh_band, vv_band):
                masked = vrt.read(band, masked=True)
                values = masked.astype(np.float32).filled(-32768)
                band_valid = (
                    ~np.ma.getmaskarray(masked)
                    & np.isfinite(values)
                    & (values > -30000)
                )
                normalized, band_valid = _normalize(values, band_valid)
                image.append(normalized)
                valid &= band_valid
            if source.count >= 3:
                mask = vrt.read(3, masked=True)
                mask_values = mask.astype(np.float32).filled(0)
                valid &= ~np.ma.getmaskarray(mask) & (mask_values > 0)
    if not valid.any():
        raise ValueError("No valid pixels found in VH/VV input.")
    stack = np.stack(image)
    stack[:, ~valid] = 0
    return stack, valid, transform, target_crs


def _run_model(
    model: torch.jit.ScriptModule,
    image: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    output_keys = {
        "objectness": "CENTERNET_OUTPUT_OBJECTNESS_MAP",
        "vessel": "CENTERNET_OUTPUT_VESSEL_MAP",
        "fishing": "CENTERNET_OUTPUT_FISHING_MAP",
        "size": "CENTERNET_OUTPUT_SIZE",
        "offset": "CENTERNET_OUTPUT_OFFSET",
    }
    height, width = valid.shape
    rows = [index * STEP for index in range(max(1, math.ceil((height - TILE) / STEP) + 1))]
    cols = [index * STEP for index in range(max(1, math.ceil((width - TILE) / STEP) + 1))]
    map_height, map_width = (rows[-1] + TILE) // 2, (cols[-1] + TILE) // 2
    channels = {"objectness": 1, "vessel": 1, "fishing": 1, "size": 1, "offset": 2}
    maps = {
        name: np.zeros((channels[name], map_height, map_width), dtype=np.float32)
        for name in channels
    }
    weights = np.zeros((map_height, map_width), dtype=np.float32)

    with torch.inference_mode():
        for row in tqdm(rows, desc="xView3 GPU rows"):
            for col in cols:
                tile_height = min(TILE, height - row)
                tile_width = min(TILE, width - col)
                if tile_height <= 0 or tile_width <= 0 or not valid[row:row + tile_height, col:col + tile_width].any():
                    continue
                tile = np.zeros((2, TILE, TILE), dtype=np.float32)
                tile[:, :tile_height, :tile_width] = image[:, row:row + tile_height, col:col + tile_width]
                tensor = torch.from_numpy(tile[None]).to(device)
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        output = model(tensor)
                except torch.cuda.OutOfMemoryError as error:
                    raise RuntimeError(
                        "GPU out of memory in the 12-model ensemble. Restart Colab with a larger GPU."
                    ) from error
                if not isinstance(output, dict):
                    raise TypeError(f"Unexpected JIT output type: {type(output)}")
                if not set(output_keys.values()).issubset(output):
                    raise ValueError(f"Missing expected output maps: {list(output)}")
                y, x = row // 2, col // 2
                for name, key in output_keys.items():
                    values = output[key].detach().float().cpu().numpy()
                    expected = (1, channels[name], TILE // 2, TILE // 2)
                    if values.shape != expected:
                        raise ValueError(f"Unexpected shape for {key}: {values.shape}; expected {expected}")
                    maps[name][:, y:y + TILE // 2, x:x + TILE // 2] += np.nan_to_num(
                        values[0], nan=0, posinf=0, neginf=0
                    )
                weights[y:y + TILE // 2, x:x + TILE // 2] += 1
                del output, tensor

    for name, values in maps.items():
        maps[name] = np.divide(
            values,
            weights[None],
            out=np.zeros_like(values),
            where=weights[None] > 0,
        )[:, :math.ceil(height / 2), :math.ceil(width / 2)]
    return maps


def infer(
    model: torch.jit.ScriptModule,
    path: Path,
    source_name: str,
    vh_band: int,
    vv_band: int,
    device: torch.device,
) -> str:
    image, valid, scene_transform, scene_crs = _load_scene(path, vh_band, vv_band)
    maps = _run_model(model, image, valid, device)
    objectness = torch.from_numpy(maps["objectness"][None]).float()
    peaks = objectness * (F.max_pool2d(objectness, kernel_size=3, stride=1, padding=1) == objectness)
    scores, indices = torch.topk(peaks.flatten(), k=min(MAX_OBJECTS, peaks.numel()))
    map_width = objectness.shape[-1]
    ys, xs = indices // map_width, indices % map_width
    offset = torch.from_numpy(maps["offset"])
    cx = (xs.float() + offset[0, ys, xs]) * 2
    cy = (ys.float() + offset[1, ys, xs]) * 2
    vessel = torch.from_numpy(maps["vessel"][0])[ys, xs]
    fishing = torch.from_numpy(maps["fishing"][0])[ys, xs]
    log_length = torch.from_numpy(maps["size"][0])[ys, xs]
    lengths = (torch.exp(F.relu(log_length)) - 1) * 10
    keep = (scores >= OBJECTNESS_THRESHOLD) & (vessel >= VESSEL_THRESHOLD)
    keep &= (cx >= 0) & (cx < valid.shape[1]) & (cy >= 0) & (cy < valid.shape[0])
    indices_kept = torch.where(keep)[0]
    if len(indices_kept):
        ix = cx[indices_kept].floor().long().numpy()
        iy = cy[indices_kept].floor().long().numpy()
        keep[indices_kept] &= torch.from_numpy(valid[iy, ix])

    selected = torch.where(keep)[0]
    if len(selected):
        rows = cy[selected].numpy()
        cols = cx[selected].numpy()
        gx, gy = scene_transform * (cols + 0.5, rows + 0.5)
        longitudes, latitudes = transform_coords(scene_crs, "EPSG:4326", gx.tolist(), gy.tolist())
    else:
        rows = cols = np.asarray([], dtype=float)
        longitudes = latitudes = []

    results = []
    for i, selected_index in enumerate(selected.tolist()):
        results.append({
            "detect_scene_row": float(rows[i]),
            "detect_scene_column": float(cols[i]),
            "objectness_p": float(scores[selected_index]),
            "is_vessel_p": float(vessel[selected_index]),
            "is_fishing_p": float(fishing[selected_index]),
            "is_vessel": True,
            "is_fishing": bool(fishing[selected_index] >= FISHING_THRESHOLD),
            "vessel_length_m": float(lengths[selected_index]),
            "lon": float(longitudes[i]),
            "lat": float(latitudes[i]),
            "source_vh": source_name,
        })
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()


def _download_cloudflared(path: Path) -> None:
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    request = urllib.request.Request(url, headers={"User-Agent": "xview3-colab-gpu-api"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    path.chmod(0o755)


def _start_tunnel(port: int) -> tuple[subprocess.Popen, str]:
    executable = Path("/content/cloudflared")
    if not executable.is_file():
        _download_cloudflared(executable)
    logfile = Path("/content/cloudflared-api.log")
    log = logfile.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(executable), "tunnel", "--url", f"http://127.0.0.1:{port}", "--logfile", str(logfile)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.monotonic() + 90
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"cloudflared exited with status {process.returncode}")
            log.flush()
            try:
                text = logfile.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                text = ""
            match = pattern.search(text)
            if match:
                return process, match.group(0)
            time.sleep(1)
        raise TimeoutError("Timed out waiting for a Cloudflare Quick Tunnel URL.")
    except Exception:
        process.terminate()
        raise
    finally:
        log.close()


def create_handler(
    model: torch.jit.ScriptModule,
    token: str,
    device: torch.device,
    vh_band: int,
    vv_band: int,
) -> type[BaseHTTPRequestHandler]:
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1)

    def run_job(job_id: str, scene_id: str, filename: str, content: bytes) -> None:
        with jobs_lock:
            jobs[job_id]["state"] = "running"
        try:
            with __import__("tempfile").TemporaryDirectory(prefix="xview3-") as temp_dir:
                scene_path = Path(temp_dir) / filename
                scene_path.write_bytes(content)
                result_csv = infer(model, scene_path, filename, vh_band, vv_band, device)
            with jobs_lock:
                jobs[job_id].update(state="completed", detections_csv=result_csv)
        except Exception as error:
            with jobs_lock:
                jobs[job_id].update(state="failed", error=str(error))

    class Handler(BaseHTTPRequestHandler):
        server_version = "XView3ColabGPU/1.0"

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[api] {format_string % args}")

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            import hmac
            return hmac.compare_digest(supplied, expected)

        def do_GET(self) -> None:
            if not self._authorized():
                self._send(401, {"error": "Unauthorized"})
                return
            if self.path == "/health":
                self._send(200, {"status": "ok", "device": str(device)})
                return
            match = re.fullmatch(r"/jobs/([a-f0-9-]{36})", self.path)
            if not match:
                self._send(404, {"error": "Not found"})
                return
            with jobs_lock:
                job = jobs.get(match.group(1))
                if job is None:
                    self._send(404, {"error": "Unknown or expired job"})
                    return
                result = dict(job)
            self._send(200, result)

        def do_POST(self) -> None:
            if not self._authorized():
                self._send(401, {"error": "Unauthorized"})
                return
            if self.path != "/jobs":
                self._send(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send(411, {"error": "Content-Length is required"})
                return
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self._send(413, {"error": f"Upload must be between 1 byte and {MAX_UPLOAD_BYTES} bytes"})
                return
            filename = Path(self.headers.get("X-Filename", "")).name
            scene_id = self.headers.get("X-Scene-ID", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", scene_id) or scene_id not in filename:
                self._send(400, {"error": "Scene ID must match the source GeoTIFF filename"})
                return
            content = self.rfile.read(length)
            if len(content) != length:
                self._send(400, {"error": "Incomplete GeoTIFF upload"})
                return
            job_id = str(uuid.uuid4())
            with jobs_lock:
                jobs[job_id] = {
                    "state": "queued",
                    "scene_id": scene_id,
                    "created_at": time.time(),
                }
                expired = [
                    key for key, value in jobs.items()
                    if time.time() - value["created_at"] > 3600
                ]
                for key in expired:
                    del jobs[key]
            executor.submit(run_job, job_id, scene_id, filename, content)
            self._send(202, {"job_id": job_id, "state": "queued"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--vh-band", type=int, default=2)
    parser.add_argument("--vv-band", type=int, default=1)
    parser.add_argument("--model", type=Path, default=Path("/content/xview3_firstplace_traced_ensemble.jit"))
    args = parser.parse_args()

    if args.vh_band == args.vv_band:
        raise ValueError("VH and VV band numbers must differ.")
    if torch.cuda.is_available() is False:
        raise RuntimeError("Colab GPU is unavailable. Select Runtime > Change runtime type > GPU.")

    device = torch.device("cuda")
    print(f"Loading official xView3 ensemble on {torch.cuda.get_device_name(0)}...")
    download_model(args.model)
    model = torch.jit.load(str(args.model), map_location="cpu").eval().to(device)
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        create_handler(model, token, device, args.vh_band, args.vv_band),
    )
    tunnel, public_url = _start_tunnel(args.port)
    print(f"XVIEW3_API_URL={public_url}", flush=True)
    print(f"XVIEW3_API_TOKEN={token}", flush=True)
    print("Keep this Colab cell running while using the dashboard.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        tunnel.terminate()


if __name__ == "__main__":
    main()
