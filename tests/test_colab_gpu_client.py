import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from colab_gpu_client import submit_inference


class ColabGpuClientTests(unittest.TestCase):
    def test_https_request_uses_verified_ca_bundle(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"status":"ok"}'
        with patch("colab_gpu_client.certifi.where", return_value="/test/ca.pem"), patch(
            "colab_gpu_client.ssl.create_default_context", return_value="verified-context"
        ) as make_context, patch(
            "colab_gpu_client.urllib.request.urlopen", return_value=response
        ) as urlopen:
            from colab_gpu_client import _request

            self.assertEqual(_request("https://gpu.example/health", "token"), {"status": "ok"})

        make_context.assert_called_once_with(cafile="/test/ca.pem")
        self.assertEqual(urlopen.call_args.kwargs["context"], "verified-context")

    def test_uploads_scene_and_returns_completed_detection_csv(self):
        csv_result = "detect_scene_row,detect_scene_column\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene-123__incheon_outer_port__raw_gamma0_db.tif"
            scene_path.write_bytes(b"geotiff")
            responses = [
                {"status": "ok", "device": "cuda"},
                {"job_id": "f3e0f4a9-2eae-4666-b3b4-292c692ecc49", "state": "queued"},
                {"state": "completed", "detections_csv": csv_result},
            ]

            with patch("colab_gpu_client._request", side_effect=responses) as request, patch(
                "colab_gpu_client.time.sleep"
            ):
                result = submit_inference(
                    "https://gpu.example.trycloudflare.com",
                    "test-token",
                    "scene-123",
                    scene_path,
                    poll_seconds=0,
                )

        self.assertEqual(result, csv_result)
        self.assertEqual(request.call_count, 3)
        upload = request.call_args_list[1]
        self.assertEqual(upload.args[0], "https://gpu.example.trycloudflare.com/jobs")
        self.assertEqual(upload.kwargs["data"], b"geotiff")
        self.assertEqual(upload.kwargs["headers"]["X-Scene-ID"], "scene-123")

    def test_requires_https_for_public_colab_endpoint(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            submit_inference("http://gpu.example", "token", "scene", Path("missing.tif"))

    def test_reports_failed_gpu_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene-123.tif"
            scene_path.write_bytes(b"geotiff")
            responses = [
                {"status": "ok", "device": "cuda"},
                {"job_id": "f3e0f4a9-2eae-4666-b3b4-292c692ecc49"},
                {"state": "failed", "error": "GPU out of memory"},
            ]

            with patch("colab_gpu_client._request", side_effect=responses), patch(
                "colab_gpu_client.time.sleep"
            ):
                with self.assertRaisesRegex(RuntimeError, "GPU out of memory"):
                    submit_inference(
                        "https://gpu.example.trycloudflare.com",
                        "test-token",
                        "scene-123",
                        scene_path,
                        poll_seconds=0,
                    )


if __name__ == "__main__":
    unittest.main()
