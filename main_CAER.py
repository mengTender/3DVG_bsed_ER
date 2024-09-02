import argparse

from caer_ext.parse_config import ConfigParser
from caers_tv import train, test


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-m', '--mode', default='train', type=str,
                      help='config mode (default: train, available: (train, test))')
    parser = args.parse_args()
    config = ConfigParser.from_args(args)
    if parser.mode == 'train':
        train(config)
    elif parser.mode == 'test':
        test(config, 64)
