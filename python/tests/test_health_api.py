import unittest
from unittest.mock import patch

from python.src.api.health import app


class TestHealthAPI(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    @patch("python.src.api.health.psutil")
    @patch("python.src.api.health.torch")
    def test_health_endpoint(self, mock_torch, mock_psutil):
        # Mocking
        mock_torch.cuda.is_available.return_value = False
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.virtual_memory.return_value.percent = 50.0

        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["system"]["cpu_percent"], 10.0)
        self.assertFalse(data["system"]["gpu_available"])

    def test_ready_endpoint(self):
        response = self.app.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ready"])


if __name__ == "__main__":
    unittest.main()
