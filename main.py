import os
import sys
import json
import torch
import pprint as pp
import torch.optim as optim

from models import (
    LSTM, NSTransformer,
    NoBaseline
)
from utils.data_utils import load_data
from utils.functions import torch_load_cpu
from utils.model_utils import get_inner_model
from utils.command_parser import process_arguments


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
    cur_dir = os.getcwd()
    data_dir = os.path.join(cur_dir, "data", opts['data_dir'])
    data, metadata = load_data(data_dir)

    # Initialize the model
    model_class = {
        'lstm': LSTM,
        'nstransformer': NSTransformer
    }.get(opts['model'], None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(24, opts['hidden_dim'], opts['n_encode_layers'], 6).to(opts['device'])

    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    load_params = {}
    if opts['load_path'] is not None:
        print('  [*] Loading model from {}'.format(opts['load_path']))
        load_params = torch_load_cpu(os.path.join(cur_dir, opts['load_path']))

    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_params.get('model', {})})
    
    # Initialize baseline
    baseline = NoBaseline()

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts['lr_model']}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts['lr_critic']}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_params:
        optimizer.load_state_dict(load_params['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts['device'])
    sys.exit(0)


def main(args):
    comm, opts = args
    print(comm)
    if comm == 'train':
        train_model(opts)
    sys.exit(0)

if __name__ == "__main__":
    main(process_arguments())