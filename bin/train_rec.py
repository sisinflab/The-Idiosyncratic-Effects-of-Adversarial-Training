import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# print(os.getcwd())

from parser.parsers import train_parse_args
from src.dataset.dataset import DataLoader
from src.util.dir_manager import manage_directories, get_paths
from src.util.general import get_model
import src.config.configs as cfg


def run():
    args = train_parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    path_train_data, path_test_data, path_output_rec_result, path_output_rec_weight = cfg.InputTrainFile, cfg.InputTestFile, cfg.OutputRecResult, cfg.OutputRecWeight

    path_train_data, path_test_data, = path_train_data.format(args.dataset), path_test_data.format(args.dataset)

    path_output_rec_result, path_output_rec_weight = get_paths(args, path_output_rec_result, path_output_rec_weight)

    # Create directories to Store Results and Rec Models
    manage_directories(path_output_rec_result, path_output_rec_weight)

    # Read Data
    data = DataLoader(path_train_data=path_train_data
                      , path_test_data=path_test_data)

    # Get Model
    model = get_model(args, data, path_output_rec_result, path_output_rec_weight)

    # Start the Training
    model.train()


if __name__ == '__main__':
    run()
