from unittest.mock import MagicMock, patch

from python.src.api.verify_api import verify


def test_verify_api_success():
    with (
        patch("subprocess.Popen") as mock_popen,
        patch("requests.get") as mock_get,
        patch("requests.post") as mock_post,
        patch("time.sleep"),
    ):
        # Setup mock process
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        # Setup mock responses
        mock_resp_health = MagicMock()
        mock_resp_health.status_code = 200
        mock_resp_health.json.return_value = {"status": "online"}
        mock_get.return_value = mock_resp_health

        mock_resp_predict = MagicMock()
        mock_resp_predict.status_code = 200
        mock_resp_predict.json.return_value = {"predictions": [[0.5]]}
        mock_post.return_value = mock_resp_predict

        verify()

        mock_popen.assert_called()
        mock_get.assert_called_with("http://localhost:8000/health")
        mock_post.assert_called()
        mock_proc.terminate.assert_called()


def test_verify_api_predict_fail():
    with (
        patch("subprocess.Popen") as mock_popen,
        patch("requests.get") as mock_get,
        patch("requests.post") as mock_post,
        patch("time.sleep"),
    ):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        mock_get.return_value.status_code = 200

        # Test 500 failure with specific error message
        mock_resp_fail = MagicMock()
        mock_resp_fail.status_code = 500
        mock_resp_fail.text = "Failed to load model"
        mock_post.return_value = mock_resp_fail

        verify()

        mock_proc.terminate.assert_called()
