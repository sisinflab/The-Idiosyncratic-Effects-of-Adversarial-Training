import argparse


def print_args(args):
    print('**********\nPARAMETERS:')
    for arg in vars(args):
        print("\t --{0}\t{1}".format(arg, getattr(args, arg)))
    print('**********\n')


def train_parse_args():
    parser = argparse.ArgumentParser(description="Run train of the Recommender Model.")
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--dataset', nargs='?', default='movielens', help='dataset name: movielens')
    parser.add_argument('--rec', nargs='?', default="bprmf", help="bprmf, amf, random")
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--k', type=int, default=100, help='top-k of recommendation.')
    parser.add_argument('--best_metric', type=str, default='hr')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--verbose', type=int, default=2, help='Number of epochs to store model parameters.')
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--reg', type=float, default=0, help='Regularization for user and item embeddings.')
    parser.add_argument('--bias_reg', type=float, default=0, help='Regularization for user and item BIASES.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--restore_epochs', type=int, default=1,
                        help='Default is 1: The restore epochs (Must be lower than the epochs)')
    parser.add_argument('--best', type=int, default=0, help='Parameter useful for attack scenario. Leave at 0 here.')

    # Parameters useful during the adv. training
    parser.add_argument('--adv_type', nargs='?', default="pgd", help="fgsm, future work other techniques...")
    parser.add_argument('--adv_iteration', type=int, default=2, help='Iterations for BIM/PGD Adversarial Training.')
    parser.add_argument('--adv_step_size', type=int, default=4, help='Step Size for BIM/PGD ATTACK.')
    parser.add_argument('--adv_reg', type=float, default=1.0, help='Regularization for adversarial loss')
    parser.add_argument('--adv_eps', type=float, default=0.5, help='Epsilon for adversarial weights.')

    return parser.parse_args()
