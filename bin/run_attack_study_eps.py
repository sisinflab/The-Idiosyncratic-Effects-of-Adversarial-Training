import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# print(os.getcwd())

from parser.parsers import train_parse_args, print_args
from src.dataset.dataset import DataLoader
from src.util.dir_manager import manage_directories, get_paths
from src.util.general import get_model
import src.config.configs as cfg





def attack():
    args = parse_args()

    path_train_data, path_test_data, path_output_rec_result, path_output_rec_weight = read_config(
        sections_fields=[('PATHS', 'InputTrainFile'),
                         ('PATHS', 'InputTestFile'),
                         ('PATHS', 'OutputRecResult'),
                         ('PATHS', 'OutputRecWeight')])

    path_train_data, path_test_data, = path_train_data.format(
        args.dataset), path_test_data.format(args.dataset)

    if args.rec == 'bprmf':
        path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'XX',
                                                               'XX')

        path_output_rec_weight = path_output_rec_weight.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'XX',
                                                               'XX')
    elif args.rec == 'amf':
        path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'eps' + str(args.adv_eps),
                                                               '' + args.adv_type)

        path_output_rec_weight = path_output_rec_weight.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'eps' + str(args.adv_eps),
                                                               '' + args.adv_type)

    data = DataLoader(path_train_data=path_train_data
                      , path_test_data=path_test_data)

    print("RUNNING {0} Attack on DATASET {1} and Recommender {2}".format(args.attack_type, args.dataset, args.rec))
    print("- PARAMETERS:")
    for arg in vars(args):
        print("\t- " + str(arg) + " = " + str(getattr(args, arg)))
    print("\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Initialize the model under attack
    if args.rec == 'bprmf':
        model = BPRMF(data, path_output_rec_result, path_output_rec_weight, args)
    elif args.rec == 'amf':
        model = APR(data, path_output_rec_result, path_output_rec_weight, args)
    else:
        raise NotImplementedError('Unknown Recommender Model.')

    # Restore the Model Parameters
    if not model.restore():
        raise NotImplementedError('Unknown Restore Point/Model.')

    # Initialize the Attack
    if args.attack_users == 'full':
        # Start full batch attacks
        if args.attack_type == 'fgsm':
            attack_name = '{0}_ep{1}_sz{2}_'.format(args.attack_type, args.attack_eps, args.attack_users)
            model.attack_full_fgsm(args.attack_eps, attack_name)
        elif args.attack_type in ['bim', 'pgd']:
            attack_name = '{0}{1}_ep{2}_es{3}_sz{4}_'.format(args.attack_type, args.attack_iteration, args.attack_eps,
                                                             args.attack_step_size,
                                                             args.attack_users)
            model.attack_full_iterative(args.attack_type, args.attack_iteration, args.attack_eps, args.attack_step_size,
                                        attack_name)



    else:
        raise NotImplementedError('Unknown Attack USERS STRATEGY.')


def all_attack():
    args = parse_args()
    initial = 1

    if args.rec == 'amf':
        adv_epss = [0.5]
    else:
        adv_epss = [0]

    path_train_data, path_test_data, path_output_rec_result, path_output_rec_weight = read_config(
        sections_fields=[('PATHS', 'InputTrainFile'),
                         ('PATHS', 'InputTestFile'),
                         ('PATHS', 'OutputRecResult'),
                         ('PATHS', 'OutputRecWeight')])

    path_train_data, path_test_data, = path_train_data.format(
        args.dataset), path_test_data.format(args.dataset)

    if args.rec == 'bprmf':
        path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'XX',
                                                               'XX')

        path_output_rec_weight = path_output_rec_weight.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'XX',
                                                               'XX')
    elif args.rec == 'amf':
        path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'eps' + str(args.adv_eps),
                                                               '' + args.adv_type)

        path_output_rec_weight = path_output_rec_weight.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'eps' + str(args.adv_eps),
                                                               '' + args.adv_type)

    data = DataLoader(path_train_data=path_train_data
                      , path_test_data=path_test_data)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Initialize the model under attack
    if args.rec == 'bprmf':
        model = BPRMF(data, path_output_rec_result, path_output_rec_weight, args)
    elif args.rec == 'amf':
        model = APR(data, path_output_rec_result, path_output_rec_weight, args)
    else:
        raise NotImplementedError('Unknown Recommender Model.')

    # Restore the Model Parameters
    if not model.restore():
        raise NotImplementedError('Unknown Restore Point/Model.')

    for adv_eps in adv_epss:
        args.adv_eps = adv_eps

        for attack_type in ['bim', 'pgd']:
            args.attack_type = attack_type
            if attack_type in ['bim', 'pgd']:
                #attack_iterations = [1] + np.arange(0, 300, 10).tolist()[1:] + [300]
                attack_iterations = [25]
            else:
                attack_iterations = [1]

            for attack_iteration in attack_iterations:
                args.attack_iteration = attack_iteration
                for attack_eps in [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10]:
                    args.attack_eps = attack_eps

                    print('*************')
                    print("- PARAMETERS:")
                    for arg in vars(args):
                        print("\t- " + str(arg) + " = " + str(getattr(args, arg)))
                    print("\n")

                    # Initialize the Attack
                    if args.attack_users == 'full':
                        # Start full batch attacks
                        if args.attack_type == 'fgsm':
                            attack_name = '{0}_ep{1}_sz{2}_'.format(args.attack_type, args.attack_eps,
                                                                    args.attack_users)
                            model.attack_full_fgsm(args.attack_eps, attack_name)
                        elif args.attack_type in ['bim', 'pgd']:
                            attack_name = '{0}{1}_ep{2}_es{3}_sz{4}_'.format(args.attack_type, args.attack_iteration,
                                                                             args.attack_eps, args.attack_step_size,
                                                                             args.attack_users)
                            model.attack_full_iterative(args.attack_type, args.attack_iteration, args.attack_eps,
                                                        args.attack_step_size, initial, attack_name)
                            initial = 0


if __name__ == '__main__':
    all_attack()
