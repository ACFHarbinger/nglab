"""
Functional utilities for model loading and tensor manipulation.
"""

import json
import os
from typing import Any, cast

import torch


def torch_load_cpu(load_path: str) -> Any:
    """Load torch tensors on CPU."""
    return torch.load(
        load_path, map_location=lambda storage, loc: storage
    )  # Load on CPU


def move_to(var: Any, device: torch.device | str) -> Any:
    """
    Recursively move tensors in a dictionary to a device.
    """
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, torch.Tensor):
        return var.to(device)
    return var


def load_args(filename: str) -> dict[str, Any]:
    """
    Load arguments from a JSON file with backwards compatibility.
    """
    with open(filename) as f:
        args = cast(dict[str, Any], json.load(f))

    # Backwards compatibility
    if "data_distribution" not in args:
        args["data_distribution"] = None
        probl, *dist = args["problem"].split("_")
        if probl == "op":
            args["problem"] = probl
            args["data_distribution"] = dist[0]
    return args


def _load_model_file(
    load_path: str, model: torch.nn.Module
) -> tuple[torch.nn.Module, dict[str, Any] | None]:
    """
    Loads model parameters from a file.

    Args:
        load_path (str): Path to the saved model.
        model (nn.Module): The model instance to load weights into.

    Returns:
        tuple: (model, optimizer_state_dict)
    """
    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print(f"  [*] Loading model from {load_path}")

    load_data = torch.load(
        os.path.join(os.getcwd(), load_path), map_location=lambda storage, loc: storage
    )
    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get("optimizer", None)
        load_model_state_dict = load_data.get("model", load_data)
    # If it's a model instance, get its state dict
    elif hasattr(load_data, "state_dict"):
        load_model_state_dict = load_data.state_dict()
    else:
        load_model_state_dict = load_data

    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)
    return model, load_optimizer_state_dict


def load_model(
    path: str, epoch: int | None = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Load a model and its configuration from a directory or specific file.

    Args:
        path (str): File or directory path.
        epoch (int): Specific epoch to load if path is a directory.

    Returns:
        tuple: (model, args)
    """
    from models import LSTM, NSTransformer  # type: ignore[import-untyped]

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == ".pt"
            )
        model_filename = os.path.join(path, f"epoch-{epoch}.pt")
    else:
        assert False, f"{path} is not a valid directory or file"

    args = load_args(os.path.join(path, "args.json"))
    model_class = {"lstm": LSTM, "nstransformer": NSTransformer}.get(
        args.get("model", "attention"), None
    )
    assert model_class is not None, f"Unknown model: {model_class}"

    # Cast is needed because model_class is Union[Type[LSTM], Type[NSTransformer]]
    # but they share the same constructor signature
    model = model_class(
        args["n_seq"],
        args["hidden_dim"],
        args["embedding_dim"],
        args["n_encode_layers"],
        args["pred_len"],
    )

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get("model", {})})
    model, *_ = _load_model_file(model_filename, model)
    model.eval()  # Put in eval mode
    return model, args
