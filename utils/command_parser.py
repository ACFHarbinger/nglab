import sys
import argparse

def process_arguments(m_path):
    parser = argparse.ArgumentParser(prog="nglab", description="Nothing Gambles Like A Bot, your personal stock market assistant!")
    subparsers = parser.add_subparsers(help="command", dest="command")
    
    crawler_parser = subparsers.add_parser("webcrawler", aliases=["crawler"])
    crawler_parser.add_argument('--website', '--url', type=str, help='URL of the website to crawl for data.')

    args = vars(parser.parse_args())

    if args['command'] == 'webcrawler':
        sys.exit(0)