"""HTTP client for Colab-hosted xView3 GPU inference."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import certifi


def _request(url: str, token: str, *, data: bytes | None = None, headers: dict | None = None) -> dict:
    request_headers = {"Authorization": f"Bearer {token}", **(headers or {})}
    request = urllib.request.Request(url, data=data, headers=request_headers)
    try:
        ca_bundle = os.environ.get("SSL_CERT_FILE") or certifi.where()
        ssl_context = ssl.create_default_context(cafile=ca_bundle)
        with urllib.request.urlopen(request, timeout=300, context=ssl_context) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"Colab GPU API returned HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not reach the Colab GPU API: {error}") from error


def submit_inference(
    api_url: str,
    token: str,
    scene_id: str,
    geotiff_path: Path,
    *,
    timeout_seconds: int = 1800,
    poll_seconds: float = 3,
) -> str:
    """Upload a scene, wait for its GPU job, and return the resulting detections CSV."""
    base_url = api_url.rstrip("/")
    if not base_url.startswith("https://"):
        raise ValueError("The Colab API URL must use HTTPS.")
    if not token:
        raise ValueError("The Colab API token is empty.")
    if not geotiff_path.is_file():
        raise FileNotFoundError(f"SAR GeoTIFF not found: {geotiff_path}")

    health = _request(f"{base_url}/health", token)
    if health.get("status") != "ok" or not str(health.get("device", "")).startswith("cuda"):
        raise RuntimeError(f"Colab GPU API is not ready: {health}")

    scene_name = geotiff_path.name
    job = _request(
        f"{base_url}/jobs",
        token,
        data=geotiff_path.read_bytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(geotiff_path.stat().st_size),
            "X-Scene-ID": scene_id,
            "X-Filename": scene_name,
        },
    )
    job_id = job.get("job_id")
    if not job_id:
        raise RuntimeError(f"Colab API did not return a job ID: {job}")

    deadline = time.monotonic() + timeout_seconds
    job_url = f"{base_url}/jobs/{urllib.parse.quote(job_id)}"
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        state = _request(job_url, token)
        if state.get("state") == "completed":
            result = state.get("detections_csv")
            if not isinstance(result, str) or not result.startswith("detect_scene_row,"):
                raise RuntimeError("Colab API returned an invalid detections CSV.")
            return result
        if state.get("state") == "failed":
            raise RuntimeError(f"Colab xView3 inference failed: {state.get('error', 'unknown error')}")
        if state.get("state") not in {"queued", "running"}:
            raise RuntimeError(f"Unexpected Colab inference job state: {state}")

    raise TimeoutError(f"Colab xView3 inference exceeded {timeout_seconds} seconds.")
