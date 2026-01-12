"""
Functional utilities for model loading and tensor manipulation.
"""
import os
import json
import torch


def torch_load_cpu(load_path):
    """Load torch tensors on CPU."""
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    """
    Recursively move tensors in a dictionary to a device.
    """
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def load_args(filename):
    """
    Load arguments from a JSON file with backwards compatibility.
    """
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def _load_model_file(load_path, model):
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
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)
    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)
    return model, load_optimizer_state_dict


def load_model(path, epoch=None):
    """
    Load a model and its configuration from a directory or specific file.

    Args:
        path (str): File or directory path.
        epoch (int): Specific epoch to load if path is a directory.

    Returns:
        tuple: (model, args)
    """
    from models import LSTM, NSTransformer

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))
    model_class = {
        'lstm': LSTM,
        'nstransformer': NSTransformer
    }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(args['n_seq'], args['hidden_dim'], args['embedding_dim'], args['n_encode_layers'], args['pred_len'])

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model, *_ = _load_model_file(model_filename, model)
    model.eval()  # Put in eval mode
    return model, args