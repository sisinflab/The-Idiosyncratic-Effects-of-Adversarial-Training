import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# print(os.getcwd())

from parser.parsers import run_attack_parse_args, print_args
from src.dataset.dataset import DataLoader
from src.util.dir_manager import manage_directories, get_paths
from src.util.general import get_model
import src.config.configs as cfg


def run():
    args = run_attack_parse_args()
    print_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    path_train_data, path_test_data, path_output_rec_result, path_output_rec_weight, path_output_rec_list = cfg.InputTrainFile, cfg.InputTestFile, cfg.OutputRecResult, cfg.OutputRecWeight, cfg.OutputRecList

    path_train_data, path_test_data, = path_train_data.format(args.dataset), path_test_data.format(args.dataset)

    path_output_rec_result, path_output_rec_weight, path_output_rec_list = get_paths(args, path_output_rec_result,
                                                                                     path_output_rec_weight,
                                                                                     path_output_rec_list)

    # Read Data
    data = DataLoader(path_train_data=path_train_data, path_test_data=path_test_data, args=args)

    # Get Model
    model = get_model(args, data, path_output_rec_result, path_output_rec_weight, path_output_rec_list)

    # Restore the Model Parameters
    if not model.restore():
        raise NotImplementedError('Unknown Restore Point/Model.')

    for adv_attack_eps in args.list_adv_attack_eps:
        args.attack_eps = float(adv_attack_eps)
        attack_name = '{0}_eps{1}'.format(args.attack_type, args.attack_eps)
        model.attack_full_fgsm(args.attack_eps, attack_name)


if __name__ == '__main__':
    run()
