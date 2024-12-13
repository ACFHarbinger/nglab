import os
import sys
import time
import argparse


class ConfigsParser(argparse.ArgumentParser):
    def error(self, message):
        print(message, end=' ')
        self.print_help()
        sys.exit(2)


def process_arguments():
    parser = ConfigsParser(description="Nothing Gambles Like A Bot, your personal stock market assistant!")
    subparsers = parser.add_subparsers(help="command", dest="command")
    training_parser = subparsers.add_parser('train')

    # Model
    training_parser.add_argument('--model', default='lstm', help="Model, 'lstm' or 'nstransformer'")
    training_parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    training_parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    training_parser.add_argument('--n_encode_layers', type=int, default=3, help='Number of layers in the encoder network')
    #training_parser.add_argument('--n_decode_layers', type=int, default=3, help='Number of layers in the decoder network')

    # Training
    training_parser.add_argument('--n_epochs', type=int, default=20, help='The number of epochs to train')
    training_parser.add_argument('--batch_size', type=int, default=256, help='Number of instances per batch during training')
    training_parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    training_parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    training_parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    training_parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    # Other
    training_parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    training_parser.add_argument('--run_name', default='run', help='Name to identify the run')
    training_parser.add_argument('--output_dir', default='results', help='Directory to write output models to')
    training_parser.add_argument('--data_dir', help='Path to data directory')

    crawler_parser = subparsers.add_parser("webcrawler", aliases=["crawler"])
    crawler_parser.add_argument('--website', '--url', type=str, help='URL of the website to crawl for data.')

    args = vars(parser.parse_args())

    if args['command'] == 'webcrawler':
        sys.exit(0)

    if args['command'] == 'train':
        command = args.pop('command')
        args['run_name'] = "{}_{}".format(args['run_name'], time.strftime("%Y%m%dT%H%M%S"))
        args['save_dir'] = os.path.join(
            args['output_dir'],
            "{}_{}".format(args['problem'], args['graph_size']),
            args['run_name']
        )
        return command, args