import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest

# Import the module to test
# Since infer.py is a script, we import it as a module to access main
from python.src import infer


def test_inference_script_success(mock_model_artifact):
    """Test the inference script end-to-end with a valid model and input."""
    # Data to infer on (list of floats)
    input_data = [10.0, 20.0, 30.0, 40.0, 50.0]
    input_json = json.dumps(input_data)

    # Mock sys.argv
    test_args = [
        "infer.py",
        "--model_path",
        str(mock_model_artifact),
        "--input_json",
        input_json,
    ]

    with patch.object(sys, "argv", test_args):
        # Capture stdout
        with patch("sys.stdout", new=StringIO()) as fake_out:
            infer.main()
            output = fake_out.getvalue()

    # Normalize output (strip newlines)
    output = output.strip()

    # Parse JSON output
    try:
        res = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail(f"Script output is not valid JSON: {output}")

    assert res["status"] == "success"
    assert "prediction" in res
    assert isinstance(res["prediction"], list)
    # Check metadata presence
    assert "metadata" in res
    assert "training_config" in res["metadata"]


def test_inference_script_missing_model():
    """Test failure when model path is invalid."""
    input_json = "[1.0, 2.0]"
    test_args = [
        "infer.py",
        "--model_path",
        "/path/to/nonexistent/model.pt",
        "--input_json",
        input_json,
    ]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            # The script calls sys.exit(1) on failure, so we expect SystemExit
            with pytest.raises(SystemExit) as excinfo:
                infer.main()

            assert excinfo.value.code == 1
            output = fake_out.getvalue()

    res = json.loads(output)
    assert res["status"] == "error"
    assert "Model not found" in res["message"]


def test_inference_script_invalid_json():
    """Test failure when input JSON is invalid."""
    input_json = "{not_a_list: 1}"
    test_args = ["infer.py", "--model_path", "dummy.pt", "--input_json", input_json]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            with pytest.raises(SystemExit) as excinfo:
                infer.main()
            assert excinfo.value.code == 1
            output = fake_out.getvalue()

    res = json.loads(output)
    assert res["status"] == "error"
