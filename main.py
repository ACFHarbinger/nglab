import os
import sys
import json
import torch
import pprint as pp

from models import LSTM, NSTransformer
from utils.command_parser import process_arguments
from utils.train import get_inner_model
from utils.data_utils import load_data


def train_model(opts):
    # Pretty print the run args
    pp.pprint(opts)

    # Set the random seed
    torch.manual_seed(opts['seed'])

    try:
        os.makedirs(opts['save_dir'])
    except Exception as e:
        print(e)

    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts['save_dir'], "args.json"), 'w') as f:
        json.dump(opts, f, indent=True)

    # Set the device
    use_cuda = torch.cuda.is_available()
    opts['device'] = torch.device("cpu" if not use_cuda else "cuda:0")

    # Load data
    data_dir = os.path.join(os.getcwd(), "data", opts['data_dir'])
    data = load_data(data_dir)

    sys.exit(0)

    # Initialize the model
    model_class = {
        'lstm': LSTM,
        'nstransformer': NSTransformer
    }.get(opts['model'], None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(input_dim, opts['hidden_dim'], opts['n_encode_layers'], output_dim).to(opts['device'])

    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})


def main(args):
    comm, opts = args
    print(comm)
    if comm == 'train':
        train_model(opts)
    sys.exit(0)

if __name__ == "__main__":
    main(process_arguments())