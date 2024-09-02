import argparse
import os

from emotic_tv import train_emotic, test_emotic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train', 'test', 'train_test'])
    parser.add_argument('--data_path', type=str, help='Path to preprocessed data npy files/ csv files')
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path to save experiment files (results, models, logs)')
    parser.add_argument('--model_dir_name', type=str, default='models', help='Name of the directory to save models')
    parser.add_argument('--result_dir_name', type=str, default='results',
                        help='Name of the directory to save results(predictions, labels mat files)')
    parser.add_argument('--log_dir_name', type=str, default='logs',
                        help='Name of the directory to save logs (train, val)')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--discrete_loss_weight_type', type=str, default='dynamic',
                        choices=['dynamic', 'mean', 'static'], help='weight policy for discrete loss')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=52)  # use batch size = double(categorical emotion classes)
    # Generate args
    args_parser = parser.parse_args()
    return args_parser


def check_paths(args_parser):
    folders = [args_parser.result_dir_name, args_parser.model_dir_name]
    paths = list()
    for folder in folders:
        folder_path = os.path.join(args_parser.experiment_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)

    log_folders = ['train', 'val']
    for folder in log_folders:
        folder_path = os.path.join(args_parser.experiment_path, args_parser.log_dir_name, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)
    return paths


if __name__ == '__main__':
    args = parse_args()
    print('mode ', args.mode)

    result_path, model_path, train_log_path, val_log_path = check_paths(args)

    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
           'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy',
           'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    if args.mode == 'train':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for training')
        with open(os.path.join(args.experiment_path, 'config.txt'), 'w') as f:
            print(args, file=f)
        train_emotic(result_path, model_path, train_log_path, val_log_path, args)
    elif args.mode == 'test':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for testing')
        test_emotic(result_path, model_path, args, data_root='./datasets/emotic')
    elif args.mode == 'train_test':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for training and testing')
        with open(os.path.join(args.experiment_path, 'config.txt'), 'w') as f:
            print(args, file=f)
        train_emotic(result_path, model_path, train_log_path, val_log_path, args)
        test_emotic(result_path, model_path, args, data_root='./datasets/emotic')
    else:
        raise ValueError('Unknown mode')
