import argparse


def print_args(args):
    print('**********\nPARAMETERS:')
    for arg in vars(args):
        print("\t --{0}\t{1}".format(arg, getattr(args, arg)))
    print('**********\n')


def ttest_eval_bias_parse_args():
    parser = argparse.ArgumentParser(description="Run the Evaluation of Bias.")

    # Datasets
    parser.add_argument("--datasets", nargs="+", default=['movielens'],
                        help="You can choose between: movielens, yelp, pinterest ")

    # Ks used for the Bias Evaluation
    parser.add_argument("--list_k", nargs="+", default=[10, 50, 100])

    # File Group A
    parser.add_argument("--file_a", nargs='?', default='bprmf_emb64_ep4_XX_XX_XXep_2_best0_top100_rec.tsv',
                        help='The Base Fine on Which We Will Measure the Differences')

    # Files Group B
    parser.add_argument("--files_b", nargs="+", default=['amf_emb64_ep4_eps0.5_alpha1.0_fgsmep_4_best0_top100_rec.tsv'],
                        help='The list of files that will be compared with A for the Statistical Test')

    return parser.parse_args()


def eval_bias_parse_args():
    parser = argparse.ArgumentParser(description="Run the Evaluation of Bias.")

    # Datasets
    parser.add_argument("--datasets", nargs="+", default=['movielens'],
                        help="You can choose between: movielens, yelp, pinterest ")

    # Ks used for the Bias Evaluation
    parser.add_argument("--list_k", nargs="+", default=[10, 50, 100])

    return parser.parse_args()


def train_parse_args():
    parser = argparse.ArgumentParser(description="Run the Train of the Recommender Model.")

    # Dataset
    parser.add_argument('--dataset', nargs='?', default='movielens', help='dataset name: movielens, pinterest, yelp')

    # Recommender Model
    parser.add_argument('--rec', nargs='?', default="amf", help="bprmf, amf, random")

    # Epochs
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs.')
    parser.add_argument('--restore_epochs', type=int, default=2,
                        help='The restore epochs (Must be lower than the epochs)')

    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--reg', type=float, default=0, help='Regularization for user and item embeddings.')

    # Adversarial Hyper-parameters
    parser.add_argument('--adv_type', nargs='?', default="fgsm", help="fgsm, future work other techniques...")
    parser.add_argument("--list_adv_reg", nargs="+", default=[0], help='List of Regularization for adversarial loss')
    parser.add_argument("--list_adv_eps", nargs="+", default=[0], help='List of Epsilons for adversarial weights.')
    parser.add_argument('--adv_reg', type=float, default=1.0, help='Regularization for adversarial loss')
    parser.add_argument('--adv_eps', type=float, default=0.5, help='Epsilon for adversarial weights.')

    # Various
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--k', type=int, default=100, help='top-k of recommendation.')
    parser.add_argument('--best_metric', type=str, default='hr')
    parser.add_argument('--verbose', type=int, default=2, help='Number of epochs to store model parameters.')
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--bias_reg', type=float, default=0, help='Regularization for user and item BIASES.')
    parser.add_argument('--best', type=int, default=0, help='Parameter useful for attack scenario. Leave at 0 here.')

    # Future Work
    parser.add_argument('--adv_iteration', type=int, default=0, help='Iterations for BIM/PGD Adversarial Training.')
    parser.add_argument('--adv_step_size', type=int, default=0, help='Step Size for BIM/PGD ATTACK.')

    return parser.parse_args()


def run_attack_parse_args():
    parser = argparse.ArgumentParser(description="Run Attack.")

    # Attack Parameters
    parser.add_argument("--list_adv_attack_eps", nargs="+", default=[0.5, 1.0])

    # Dataset
    parser.add_argument('--dataset', nargs='?', default='movielens', help='dataset name: movielens, pinterest, yelp')

    # Recommender Model
    parser.add_argument('--rec', nargs='?', default="amf", help="bprmf, amf, random")

    # Epochs
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs.')
    parser.add_argument('--restore_epochs', type=int, default=2,
                        help='The restore epochs (Must be lower than the epochs)')

    # Hyper-parameters [Used for Restoring The Model]
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--reg', type=float, default=0, help='Regularization for user and item embeddings.')

    # Adversarial Hyper-parameters  [Used for Restoring The Model]
    parser.add_argument('--adv_type', nargs='?', default="fgsm", help="fgsm, future work other techniques...")
    parser.add_argument('--adv_reg', type=float, default=1.0, help='Regularization for adversarial loss')
    parser.add_argument('--adv_eps', type=float, default=0.5, help='Epsilon for adversarial weights.')

    # Various
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--k', type=int, default=100, help='top-k of recommendation.')
    parser.add_argument('--best_metric', type=str, default='hr')
    parser.add_argument('--verbose', type=int, default=2, help='Number of epochs to store model parameters.')
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--bias_reg', type=float, default=0, help='Regularization for user and item BIASES.')
    parser.add_argument('--best', type=int, default=0, help='Parameter useful for attack scenario. Leave at 0 here.')

    # Parameters useful during the adv. training [Used for Restoring The Model]
    parser.add_argument('--adv_iteration', type=int, default=10, help='Iterations for BIM/PGD Adversarial Training.')
    parser.add_argument('--adv_step_size', type=int, default=4, help='Step Size for BIM/PGD ATTACK.')

    # Parameters useful during the adv. attack (Are not useful here)
    parser.add_argument('--attack_type', nargs='?', default="fgsm", help="fgsm")
    parser.add_argument('--attack_users', nargs='?', default="full", help="full, random (to be implemented), ...")
    parser.add_argument('--attack_eps', type=float, default=0.5, help='Epsilon for adversarial ATTACK.')

    return parser.parse_args()
