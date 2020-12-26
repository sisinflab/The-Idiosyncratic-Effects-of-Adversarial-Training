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


def run():
    args = train_parse_args()

    print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for adv_eps in args.list_adv_eps:
        for adv_reg in args.list_adv_reg:
            args.adv_eps, args.adv_reg = float(adv_eps), float(adv_reg)
            print('\tStart Adv.Eps. {} and Adv.Reg. {}'.format(adv_eps, adv_reg))

            path_train_data, path_test_data, path_output_rec_result, path_output_rec_weight, path_output_rec_list = cfg.InputTrainFile, cfg.InputTestFile, cfg.OutputRecResult, cfg.OutputRecWeight, cfg.OutputRecList

            path_train_data, path_test_data, = path_train_data.format(args.dataset), path_test_data.format(args.dataset)

            path_output_rec_result, path_output_rec_weight, path_output_rec_list = get_paths(args,
                                                                                             path_output_rec_result,
                                                                                             path_output_rec_weight,
                                                                                             path_output_rec_list)

            # Create directories to Store Results and Rec Models
            manage_directories(path_output_rec_result, path_output_rec_weight, path_output_rec_list)

            # Read Data
            data = DataLoader(path_train_data=path_train_data
                              , path_test_data=path_test_data,
                              args=args)

            # Get Model
            model = get_model(args, data, path_output_rec_result, path_output_rec_weight, path_output_rec_list)

            # Start the Training
            model.train()

            print('\tEnd Adv.Eps. {} and Adv.Reg. {}'.format(adv_eps, adv_reg))


if __name__ == '__main__':
    run()
