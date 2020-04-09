import argparse
import logging
from train import *

def main():
    parser = argparse.ArgumentParser(description='train mnist: classification of handwritten digits')
    parser.add_argument('--lr', type=float, required=False, default=0.01)
    parser.add_argument('--type', choices=['CNN', 'MLP'], required=False, default='CNN')
    parser.add_argument('--loglevel', choices=['INFO', 'DEBUG', 'ERROR'], required=False, default='DEBUG')
    args = parser.parse_args()
    logging.basicConfig(filename="file.log", level=getattr(logging, args.loglevel))
    train(args)

if __name__ == '__main__':
    main()
