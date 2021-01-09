import sys
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# print(os.getcwd())

from parser.parsers import build_gradient_magnitude_plot_parse_args, print_args
from src.dataset.dataset import DataLoader
from src.util.dir_manager import manage_directories, get_paths_build
from src.util.general import get_model
import src.config.configs as cfg
from src.util.write import save_obj
from src.util.read import load_obj

color_thresholds = {
    0.01: 'green',
    0.1: 'blue',
    0.5: 'black'
}


def run():
    args = build_gradient_magnitude_plot_parse_args()

    # Fixed Parameters
    args.batch_size = 1

    # args.dataset = 'sampled_movielens'
    print_args(args)

    print('*** Start Build of Plots ***')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    args.adv_eps, args.adv_reg = float(args.adv_eps), float(args.adv_reg)
    # args.rec == 'amf'

    path_train_data, path_test_data, path_output_rec_result, path_output_rec_weight, path_output_rec_list = cfg.InputTrainFile, cfg.InputTestFile, cfg.OutputRecResult, cfg.OutputRecWeight, cfg.OutputRecList

    path_train_data, path_test_data, = path_train_data.format(args.dataset), path_test_data.format(args.dataset)

    path_output_rec_result, path_output_rec_weight, path_output_rec_list = get_paths_build(args,
                                                                                           path_output_rec_result,
                                                                                           path_output_rec_weight,
                                                                                           path_output_rec_list)

    # Read Data
    data = DataLoader(path_train_data=path_train_data
                      , path_test_data=path_test_data,
                      args=args)

    if args.train:
        # Create directories to Store Results and Rec Models
        manage_directories(path_output_rec_result, path_output_rec_weight, path_output_rec_list)

        print('Train the Model to Monitor the Gradient Magnitudes')
        # Get Model
        model = get_model(args, data, path_output_rec_result, path_output_rec_weight, path_output_rec_list)

        # Start the Training while Storing the Gradient Updates
        positive_gradient_magnitudes, negative_gradient_magnitudes, \
        adv_positive_gradient_magnitudes, adv_negative_gradient_magnitudes = model.train_to_build_plots()

        # Save Object
        save_obj(positive_gradient_magnitudes,
                 '{0}{1}-positive_gradient_magnitudes'.format(path_output_rec_result,
                                                              path_output_rec_result.split('/')[-2]))
        save_obj(negative_gradient_magnitudes,
                 '{0}{1}-negative_gradient_magnitudes'.format(path_output_rec_result,
                                                              path_output_rec_result.split('/')[-2]))
        save_obj(adv_positive_gradient_magnitudes,
                 '{0}{1}-adv_positive_gradient_magnitudes'.format(path_output_rec_result,
                                                                  path_output_rec_result.split('/')[-2]))
        save_obj(adv_negative_gradient_magnitudes,
                 '{0}{1}-adv_negative_gradient_magnitudes'.format(path_output_rec_result,
                                                                  path_output_rec_result.split('/')[-2]))
    else:
        print('Load the Model Results to Monitor the Gradient Magnitudes')

        positive_gradient_magnitudes = load_obj(
            '{0}{1}-positive_gradient_magnitudes'.format(path_output_rec_result,
                                                         path_output_rec_result.split('/')[-2]))
        negative_gradient_magnitudes = load_obj(
            '{0}{1}-negative_gradient_magnitudes'.format(path_output_rec_result,
                                                         path_output_rec_result.split('/')[-2]))
        adv_positive_gradient_magnitudes = load_obj(
            '{0}{1}-adv_positive_gradient_magnitudes'.format(path_output_rec_result,
                                                             path_output_rec_result.split('/')[-2]))
        adv_negative_gradient_magnitudes = load_obj(
            '{0}{1}-adv_negative_gradient_magnitudes'.format(path_output_rec_result,
                                                             path_output_rec_result.split('/')[-2]))

    print('Start the Generation of the Probability by Training Epochs on BPR-MF')
    generate_plot_probability_of_grad_magn(path_output_rec_result, positive_gradient_magnitudes)

    print('Start the Generation of the Probability by Training Epochs on AMF')
    generate_plot_probability_of_advers_grad_magn(path_output_rec_result, positive_gradient_magnitudes,
                                                  adv_positive_gradient_magnitudes)

    print(
        'Start the Generation of the SUM of Positive and Negative Update by Training Epochs on AMF for Short Head and Log Tail Items')
    generate_plot_sum_of_update_of_advers_grad_magn(path_output_rec_result, positive_gradient_magnitudes,
                                                    adv_positive_gradient_magnitudes, negative_gradient_magnitudes,
                                                    adv_negative_gradient_magnitudes)


def generate_plot_probability_of_grad_magn(path_output_rec_result, positive_gradient_magnitudes):
    num_epochs = len(list(positive_gradient_magnitudes.keys()))
    x_axes = sorted(list(positive_gradient_magnitudes.keys()))[:num_epochs // 2]
    # x_axes = sorted(list(positive_gradient_magnitudes.keys()))

    thresholds = [0.01, 0.5]

    for threshold in thresholds:
        print('\tPlotting for threshold: {}'.format(threshold))
        # We have 2 y-axes. One for each threshold.
        y_axes = []
        for epoch in x_axes:
            num_update = 0
            num_updated_under_threshold = 0
            for item_id in positive_gradient_magnitudes[epoch].keys():
                for per_item_update in positive_gradient_magnitudes[epoch][item_id]:
                    if per_item_update < threshold:
                        num_updated_under_threshold += 1
                    num_update += 1
            y_axes.append(num_updated_under_threshold / num_update)

        plt.plot(x_axes, y_axes, '-', color=color_thresholds[threshold], label='T: {}'.format(threshold))

    plt.xlabel = 'Training Epochs'
    plt.ylabel = 'Probability'
    plt.legend()

    # plt.show()
    plt.savefig(
        '{0}{1}-bprmf-until{2}-prob-iter.png'.format(path_output_rec_result, path_output_rec_result.split('/')[-2],
                                                     num_epochs // 2), format='png')


def generate_plot_probability_of_advers_grad_magn(path_output_rec_result, positive_gradient_magnitudes,
                                                  adv_positive_gradient_magnitudes):
    plt.figure()
    num_epochs = len(list(positive_gradient_magnitudes.keys()))
    x_axes = sorted(list(positive_gradient_magnitudes.keys()))[num_epochs // 2:]
    # x_axes = sorted(list(positive_gradient_magnitudes.keys()))

    thresholds = [0.01, 0.5]

    for threshold in thresholds:
        print('\tPlotting for threshold: {}'.format(threshold))
        # We have 2 y-axes. One for each threshold.
        y_axes = []
        for epoch in x_axes:
            num_update = 0
            num_updated_under_threshold = 0
            for item_id in positive_gradient_magnitudes[epoch].keys():
                for per_item_update in positive_gradient_magnitudes[epoch][item_id]:
                    if per_item_update < threshold:
                        num_updated_under_threshold += 1
                    num_update += 1
            y_axes.append(num_updated_under_threshold / num_update)

        plt.plot(x_axes, y_axes, '-', color=color_thresholds[threshold], label='T: {}'.format(threshold))

    for threshold in thresholds:
        print('\tPlotting for threshold: {} (Adversarial Setting)'.format(threshold))
        # We have 2 y-axes. One for each threshold.
        y_axes = []
        for epoch in x_axes:
            num_update = 0
            num_updated_under_threshold = 0
            for item_id in adv_positive_gradient_magnitudes[epoch].keys():
                for per_item_update in adv_positive_gradient_magnitudes[epoch][item_id]:
                    if per_item_update < threshold:
                        num_updated_under_threshold += 1
                    num_update += 1
            y_axes.append(num_updated_under_threshold / num_update)

        plt.plot(x_axes, y_axes, '--', color=color_thresholds[threshold], label='T: {} (Adv.)'.format(threshold))

    plt.xlabel = 'Training Epochs'
    plt.ylabel = 'Probability'
    plt.legend()

    # plt.show()
    plt.savefig(
        '{0}{1}-bprmf-amf-until{2}-prob-iter.png'.format(path_output_rec_result, path_output_rec_result.split('/')[-2],
                                                         num_epochs), format='png')


def generate_plot_sum_of_update_of_advers_grad_magn(path_output_rec_result, positive_gradient_magnitudes,
                                                    adv_positive_gradient_magnitudes, negative_gradient_magnitudes,
                                                    adv_negative_gradient_magnitudes):
    pass


if __name__ == '__main__':
    run()
