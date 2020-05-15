import argparse
from cdae import ICDAE

from load_data_ranking import load_data_neg, load_data_all


def parse_args():
    parser = argparse.ArgumentParser(description='Collaborative Denoising Auto-Encoder')
    parser.add_argument('--model', default='CDAE')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)  # 128 for unlimpair
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 1e-4 for unlimpair
    parser.add_argument('--reg_rate', type=float, default=0.1)  # 0.01 for unlimpair
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    batch_size = args.batch_size

    train_data, test_data, n_user, n_item = load_data_neg(test_size=0.2, sep="\t")

    model = None
    # Model selection
    if args.model == "CDAE":
        train_data, test_data, n_user, n_item = load_data_all(test_size=0.2, sep="\t")
        model = ICDAE(n_user, n_item)
    # build and execute the model
    if model is not None:
        model.build_network()
        model.execute(train_data, test_data)