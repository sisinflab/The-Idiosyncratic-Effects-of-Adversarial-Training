import sys
import os
import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 22})

import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
import numpy as np

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
from src.evaluation.read_results import read_data, get_data_statistics
from src.evaluation.metrics import get_head_tail_split

color_thresholds = {
    0.01: 'green',
    0.1: 'blue',
    0.5: 'black'
}

marker_thresholds = {
    0.01: 'o',
    0.1: '*',
    0.5: '^'
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

    # print('Start the Generation of the Probability by Training Epochs on BPR-MF')
    # generate_plot_probability_of_grad_magn(path_output_rec_result, positive_gradient_magnitudes)
    #
    # print('Start the Generation of the Probability by Training Epochs on AMF')
    # generate_plot_probability_of_advers_grad_magn(path_output_rec_result, positive_gradient_magnitudes,
    #                                               adv_positive_gradient_magnitudes)

    #
    print('Start the Generation of the Probability by Training Epochs on BPR-MF and AMF')
    generate_plot_probability_of_grad_magn_plus_advers_grad_magn(path_output_rec_result, positive_gradient_magnitudes,
                                                                 adv_positive_gradient_magnitudes)

    #
    # print(
    #     'Start the Generation of the SUM of Positive and Negative Update by Training Epochs on AMF for Short Head and Log Tail Items')
    # generate_plot_sum_of_update_of_advers_grad_magn(args.dataset, path_output_rec_result, positive_gradient_magnitudes,
    #                                                 adv_positive_gradient_magnitudes, negative_gradient_magnitudes,
    #                                                 adv_negative_gradient_magnitudes)

    # print(
    #     'Start the Generation of the Differences SUM of Positive and Negative Update by Training Epochs on AMF for Short Head and Log Tail Items')
    # generate_plot_differences_of_update_of_advers_grad_magn(args.dataset, path_output_rec_result,
    #                                                         positive_gradient_magnitudes,
    #                                                         adv_positive_gradient_magnitudes,
    #                                                         negative_gradient_magnitudes,
    #                                                         adv_negative_gradient_magnitudes)


def generate_plot_probability_of_grad_magn(path_output_rec_result, positive_gradient_magnitudes):
    num_epochs = len(list(positive_gradient_magnitudes.keys()))
    x_axes = sorted(list(positive_gradient_magnitudes.keys()))[:num_epochs // 2]
    # x_axes = sorted(list(positive_gradient_magnitudes.keys()))

    thresholds = [0.01, 0.1, 0.5]

    for threshold in thresholds:
        print('\tPlotting for threshold: {}'.format(threshold))
        # We have 2 y-axes. One for each threshold.
        y_axes = []
        for epoch in x_axes:
            num_update = 0
            num_updated_under_threshold = 0
            for item_id in positive_gradient_magnitudes[epoch].keys():
                for per_item_update in positive_gradient_magnitudes[epoch][item_id]:
                    # import math
                    # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001))  # *-1
                    # per_item_update = 1 - 1 / (1 + math.exp(-z))
                    if per_item_update < threshold:
                        num_updated_under_threshold += 1
                    num_update += 1
            y_axes.append(num_updated_under_threshold / num_update)

        plt.plot(x_axes, y_axes, '-', color=color_thresholds[threshold],
                 label=r'$p(\omega)$' + r'$<$ {0}'.format(threshold), linewidth=2)

    plt.xlabel(r'Training Epochs $(t)$')
    plt.ylabel(r'Probability')
    plt.legend()
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(
        '{0}{1}-bprmf-until{2}-prob-iter.png'.format(path_output_rec_result, path_output_rec_result.split('/')[-2],
                                                     num_epochs // 2), format='png')
    plt.close()


def generate_plot_probability_of_advers_grad_magn(path_output_rec_result, positive_gradient_magnitudes,
                                                  adv_positive_gradient_magnitudes):
    plt.figure()
    num_epochs = len(list(positive_gradient_magnitudes.keys()))
    x_axes = sorted(list(positive_gradient_magnitudes.keys()))[num_epochs // 2:]
    # x_axes = sorted(list(positive_gradient_magnitudes.keys()))

    thresholds = [0.01, 0.1, 0.5]

    for threshold in thresholds:
        print('\tPlotting for threshold: {}'.format(threshold))
        # We have 2 y-axes. One for each threshold.
        y_axes = []
        for epoch in x_axes:
            num_update = 0
            num_updated_under_threshold = 0
            for item_id in positive_gradient_magnitudes[epoch].keys():
                for per_item_update in positive_gradient_magnitudes[epoch][item_id]:
                    # import math
                    # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001))  # *-1
                    # per_item_update = 1 - 1 / (1 + math.exp(-z))
                    if per_item_update < threshold:
                        num_updated_under_threshold += 1
                    num_update += 1
            y_axes.append(num_updated_under_threshold / num_update)

        plt.plot(x_axes, y_axes, '-', color=color_thresholds[threshold],
                 label=r'$p(\omega)$' + r'$<$ {0}'.format(threshold), linewidth=2)

    for threshold in thresholds:
        print('\tPlotting for threshold: {} (Adversarial Setting)'.format(threshold))
        # We have 2 y-axes. One for each threshold.
        y_axes = []
        for epoch in x_axes:
            num_update = 0
            num_updated_under_threshold = 0
            for item_id in adv_positive_gradient_magnitudes[epoch].keys():
                for per_item_update in adv_positive_gradient_magnitudes[epoch][item_id]:
                    # import math
                    # z = math.log(
                    #     per_item_update / (1 - per_item_update + 0.0000000000000001) + 0.0000000000000001)  # *-1
                    # per_item_update = 1 - 1 / (1 + math.exp(-z))
                    if per_item_update < threshold:
                        num_updated_under_threshold += 1
                    num_update += 1
            y_axes.append(num_updated_under_threshold / num_update)

        plt.plot(x_axes, y_axes, '--', color=color_thresholds[threshold],
                 label=r'$p(\omega^{adv})$' + r'$<$ {0}'.format(threshold), linewidth=2)

    plt.xlabel(r'Training Epochs $(t)$')
    plt.ylabel(r'Probability')
    plt.legend()
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(
        '{0}{1}-bprmf-amf-until{2}-prob-iter.png'.format(path_output_rec_result, path_output_rec_result.split('/')[-2],
                                                         num_epochs), format='png')
    plt.close()


def generate_plot_probability_of_grad_magn_plus_advers_grad_magn(path_output_rec_result, positive_gradient_magnitudes,
                                                                 adv_positive_gradient_magnitudes):
    plt.figure()
    num_epochs = len(list(positive_gradient_magnitudes.keys()))
    x_axes = sorted(list(positive_gradient_magnitudes.keys()))
    # x_axes = sorted(list(positive_gradient_magnitudes.keys()))

    thresholds = [0.01, 0.1, 0.5]

    fig = plt.figure()
    ax = plt.subplot(111)

    for threshold in thresholds:
        print('\tPlotting for threshold: {}'.format(threshold))
        # We have 2 y-axes. One for each threshold.
        y_axes_bpr = []
        y_axes_apr = []

        for epoch in x_axes:
            bpr_num_update = 0
            bpr_num_updated_under_threshold = 0
            for item_id in positive_gradient_magnitudes[epoch].keys():
                for per_item_update in positive_gradient_magnitudes[epoch][item_id]:
                    # import math
                    # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001))  # *-1
                    # per_item_update = 1 - 1 / (1 + math.exp(-z))
                    if per_item_update < threshold:
                        bpr_num_updated_under_threshold += 1
                    bpr_num_update += 1
            y_axes_bpr.append(bpr_num_updated_under_threshold / bpr_num_update)

            apr_num_update = 0
            apr_num_updated_under_threshold = 0
            for item_id in adv_positive_gradient_magnitudes[epoch].keys():
                for per_item_update in adv_positive_gradient_magnitudes[epoch][item_id]:
                    # import math
                    # z = math.log(
                    #     per_item_update / (1 - per_item_update + 0.0000000000000001) + 0.0000000000000001)  # *-1
                    # per_item_update = 1 - 1 / (1 + math.exp(-z))
                    if per_item_update < threshold:
                        apr_num_updated_under_threshold += 1
                    apr_num_update += 1
            if apr_num_update > 0:
                y_axes_apr.append(apr_num_updated_under_threshold / apr_num_update)

        l1, = ax.plot(x_axes, y_axes_bpr, '-', color=color_thresholds[threshold],
                 label=r'$p(\omega)$' + r'$<$ {0}'.format(threshold), linewidth=2)
        l2, = ax.plot(x_axes[100:], y_axes_apr, '--', color=color_thresholds[threshold],
                 label=r'$p(\omega^{adv})$' + r'$<$ {0}'.format(threshold), linewidth=2)

    plt.xlabel(r'Training Epochs ($t$)')
    plt.ylabel(r'Probability')
    # plt.legend()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])
    #
    # # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #           fancybox=True, shadow=True, ncol=2)

    plt.ylim(0, 1)
    plt.vlines(int(num_epochs / 2), ymin=0, ymax=1, colors=['red'], linewidth=3)

    xticks = np.arange(0, num_epochs + 1, step=50)
    plt.tight_layout()
    plt.xticks(xticks)
    # xticks[10] = r'$T_{BPR}=100$'
    # xticks[-1] = r'$T_{APR}=200$'
    # plt.xticklabels(xticks)

    # plt.show()
    plt.savefig(
        '{0}{1}-bprmf-amf-together-ep{2}-prob-iter.png'.format(path_output_rec_result,
                                                               path_output_rec_result.split('/')[-2],
                                                               num_epochs), format='png', bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_plot_sum_of_update_of_advers_grad_magn(dataset, path_output_rec_result, positive_gradient_magnitudes,
                                                    adv_positive_gradient_magnitudes, negative_gradient_magnitudes,
                                                    adv_negative_gradient_magnitudes):
    train, test = read_data(dataset)

    # Split in Short Head, Long Tail, and Distant Tail
    item_pop = train.groupby([cfg.item_field]).count().sort_values(cfg.user_field, ascending=False)[cfg.user_field]
    dict_item_pop = dict(zip(item_pop.index, item_pop.to_list()))

    num_users, num_items, num_ratings = get_data_statistics(train, test)

    # Add the items not in the training
    for item_id in set(range(num_items)).difference(set(item_pop.index)):
        dict_item_pop[item_id] = 0

    short_head_split = get_head_tail_split(item_pop, num_items)

    short_head_items = np.array(item_pop[:short_head_split].index)
    long_tail_items = np.array(item_pop[short_head_split:].index)

    plt.figure()

    num_epochs = len(list(positive_gradient_magnitudes.keys()))
    x_axes = sorted(list(positive_gradient_magnitudes.keys()))[num_epochs // 2:]
    # x_axes = sorted(list(positive_gradient_magnitudes.keys()))

    print('\tPlotting Positive Grad. Magnitude for Short Head and Long Tail Items')
    # We have 2 y-axes. One for each threshold.
    y_axes_short_head = []
    y_axes_long_tail = []
    for epoch in x_axes:
        sum_update_short_head = 0
        sum_update_long_tail = 0
        num_update_short_head = 0
        num_update_long_tail = 0
        for item_id in positive_gradient_magnitudes[epoch].keys():
            for per_item_update in positive_gradient_magnitudes[epoch][item_id]:
                # import math
                # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001))  # *-1
                # per_item_update = 1 - 1 / (1 + math.exp(-z))
                if item_id in short_head_items:
                    sum_update_short_head += per_item_update
                    num_update_short_head += 1
                elif item_id in long_tail_items:
                    sum_update_long_tail += per_item_update
                    num_update_long_tail += 1
        for item_id in adv_positive_gradient_magnitudes[epoch].keys():
            for per_item_update in adv_positive_gradient_magnitudes[epoch][item_id]:
                # import math
                # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001) + 0.0000000000000001 )  # *-1
                # per_item_update = 1 - 1 / (1 + math.exp(-z))
                if item_id in short_head_items:
                    sum_update_short_head += per_item_update
                    num_update_short_head += 1
                elif item_id in long_tail_items:
                    sum_update_long_tail += per_item_update
                    num_update_long_tail += 1
        y_axes_short_head.append(sum_update_short_head)
        y_axes_long_tail.append(sum_update_long_tail)

    plt.plot(x_axes, y_axes_short_head, '-', color='black', label=r'Positive Update on $I_{SH}$')
    plt.plot(x_axes, y_axes_long_tail, '-', color='red', label=r'Positive Update on $I_{LT}$')

    print('\tPlotting Negative Grad. Magnitude for Short Head and Long Tail Items')
    # We have 2 y-axes. One for each threshold.
    y_axes_short_head = []
    y_axes_long_tail = []
    for epoch in x_axes:
        sum_update_short_head = 0
        sum_update_long_tail = 0
        num_update_short_head = 0
        num_update_long_tail = 0
        for item_id in negative_gradient_magnitudes[epoch].keys():
            for per_item_update in negative_gradient_magnitudes[epoch][item_id]:
                per_item_update *= -1  # Cast to Positive Value
                # import math
                # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001) + 0.0000000000000001)  # *-1
                # per_item_update = 1 - 1 / (1 + math.exp(-z))
                if item_id in short_head_items:
                    sum_update_short_head += per_item_update * -1
                    num_update_short_head += 1
                elif item_id in long_tail_items:
                    sum_update_long_tail += per_item_update * -1
                    num_update_long_tail += 1
                # else:
                #     raise Exception
        for item_id in adv_negative_gradient_magnitudes[epoch].keys():
            for per_item_update in adv_negative_gradient_magnitudes[epoch][item_id]:
                per_item_update *= -1  # Cast to Positive Value
                # import math
                # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001) + 0.0000000000000001)  # *-1
                # per_item_update = 1 - 1 / (1 + math.exp(-z))
                if item_id in short_head_items:
                    sum_update_short_head += per_item_update * -1
                    num_update_short_head += 1
                elif item_id in long_tail_items:
                    sum_update_long_tail += per_item_update * -1
                    num_update_long_tail += 1
                # else:
                #     raise Exception
        y_axes_short_head.append(sum_update_short_head)
        y_axes_long_tail.append(sum_update_long_tail)

    plt.plot(x_axes, y_axes_short_head, '*-', color='black', label='Negative Update SH')
    plt.plot(x_axes, y_axes_long_tail, '*-', color='red', label='Negative Update LT')

    plt.xlabel('Training Epochs')
    plt.ylabel('Sum of Updates')
    plt.legend()

    plt.savefig(
        '{0}{1}-bprmf-amf-until{2}-sum-update.png'.format(path_output_rec_result, path_output_rec_result.split('/')[-2],
                                                          num_epochs), format='png')


def generate_plot_differences_of_update_of_advers_grad_magn(dataset, path_output_rec_result,
                                                            positive_gradient_magnitudes,
                                                            adv_positive_gradient_magnitudes,
                                                            negative_gradient_magnitudes,
                                                            adv_negative_gradient_magnitudes):
    train, test = read_data(dataset)

    # Split in Short Head, Long Tail, and Distant Tail
    item_pop = train.groupby([cfg.item_field]).count().sort_values(cfg.user_field, ascending=False)[cfg.user_field]
    dict_item_pop = dict(zip(item_pop.index, item_pop.to_list()))

    num_users, num_items, num_ratings = get_data_statistics(train, test)

    # Add the items not in the training
    for item_id in set(range(num_items)).difference(set(item_pop.index)):
        dict_item_pop[item_id] = 0

    short_head_split = get_head_tail_split(item_pop, num_items)

    short_head_items = np.array(item_pop[:short_head_split].index)
    long_tail_items = np.array(item_pop[short_head_split:].index)

    plt.figure()

    num_epochs = len(list(positive_gradient_magnitudes.keys()))
    x_axes = sorted(list(positive_gradient_magnitudes.keys()))
    # [num_epochs // 2:]
    # x_axes = sorted(list(positive_gradient_magnitudes.keys()))

    print('\tPlotting Positive Grad. Magnitude for Short Head and Long Tail Items')
    # We have 2 y-axes. One for each threshold.
    y_axes_short_head = []
    y_axes_long_tail = []
    for epoch in x_axes:
        sum_update_short_head = 0
        sum_update_long_tail = 0
        num_update_short_head = 0
        num_update_long_tail = 0
        for item_id in positive_gradient_magnitudes[epoch].keys():
            for per_item_update in positive_gradient_magnitudes[epoch][item_id]:
                # import math
                # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001))  # *-1
                # per_item_update = 1 - 1 / (1 + math.exp(-z))
                if item_id in short_head_items:
                    sum_update_short_head += per_item_update
                    num_update_short_head += 1
                elif item_id in long_tail_items:
                    sum_update_long_tail += per_item_update
                    num_update_long_tail += 1
        try:
            for item_id in adv_positive_gradient_magnitudes[epoch].keys():
                for per_item_update in adv_positive_gradient_magnitudes[epoch][item_id]:
                    # import math
                    # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001) + 0.0000000000000001 )  # *-1
                    # per_item_update = 1 - 1 / (1 + math.exp(-z))
                    if item_id in short_head_items:
                        sum_update_short_head += per_item_update
                        num_update_short_head += 1
                    elif item_id in long_tail_items:
                        sum_update_long_tail += per_item_update
                        num_update_long_tail += 1
        except:
            print('Epoch {} not in adversarial.'.format(epoch))

        y_axes_short_head.append(sum_update_short_head)
        y_axes_long_tail.append(sum_update_long_tail)

    print('\tRemove Negative Grad. Magnitude for Short Head and Long Tail Items')
    for epoch in x_axes:
        sum_update_short_head = 0
        sum_update_long_tail = 0
        num_update_short_head = 0
        num_update_long_tail = 0
        for item_id in negative_gradient_magnitudes[epoch].keys():
            for per_item_update in negative_gradient_magnitudes[epoch][item_id]:
                per_item_update *= -1  # Cast to Positive Value
                # import math
                # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001) + 0.0000000000000001)  # *-1
                # per_item_update = 1 - 1 / (1 + math.exp(-z))
                if item_id in short_head_items:
                    sum_update_short_head += per_item_update * -1
                    num_update_short_head += 1
                elif item_id in long_tail_items:
                    sum_update_long_tail += per_item_update * -1
                    num_update_long_tail += 1
                # else:
                #     raise Exception
        try:
            for item_id in adv_negative_gradient_magnitudes[epoch].keys():
                for per_item_update in adv_negative_gradient_magnitudes[epoch][item_id]:
                    per_item_update *= -1  # Cast to Positive Value
                    # import math
                    # z = math.log(per_item_update / (1 - per_item_update + 0.0000000000000001) + 0.0000000000000001)  # *-1
                    # per_item_update = 1 - 1 / (1 + math.exp(-z))
                    if item_id in short_head_items:
                        sum_update_short_head += per_item_update * -1
                        num_update_short_head += 1
                    elif item_id in long_tail_items:
                        sum_update_long_tail += per_item_update * -1
                        num_update_long_tail += 1
        except:
            print('Epoch {} not in adversarial.'.format(epoch))
        # y_axes_short_head[epoch - 1] = y_axes_short_head[epoch - 1] + sum_update_short_head
        # y_axes_long_tail[epoch - 1] = y_axes_long_tail[epoch - 1] + sum_update_long_tail
        y_axes_short_head[epoch - 1] = (y_axes_short_head[epoch - 1] + sum_update_short_head) / len(short_head_items)
        y_axes_long_tail[epoch - 1] = (y_axes_long_tail[epoch - 1] + sum_update_long_tail) / len(long_tail_items)

    plt.plot(x_axes, y_axes_short_head, '--', color='orange', label=r'$I_{SH}$', linewidth=3)
    plt.plot(x_axes, y_axes_long_tail, '-', color='blue', label=r'$I_{LT}$', linewidth=3)

    y_min = min(y_axes_short_head) if min(y_axes_short_head) < min(y_axes_long_tail) else min(y_axes_long_tail)
    y_max = max(y_axes_short_head) if max(y_axes_short_head) > max(y_axes_long_tail) else max(y_axes_long_tail)
    plt.vlines(int(num_epochs // 2), ymin=y_min + y_min * 0.1, ymax=y_max + y_max * 0.1, color='red',
               linestyles='solid', linewidth=3)

    plt.xlabel('Training Epochs $(t)$')
    plt.ylabel('Avg. Updates')
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        '{0}{1}-bprmf-amf-until{2}-sum-final-update.png'.format(path_output_rec_result,
                                                                path_output_rec_result.split('/')[-2],
                                                                num_epochs), format='png')

    plt.close()

    plt.figure()
    plt.plot(x_axes[:int(num_epochs // 2)], y_axes_short_head[:int(num_epochs // 2)], '-', color='orange',
             label=r'$I_{SH}$', linewidth=2)
    plt.plot(x_axes[:int(num_epochs // 2)], y_axes_long_tail[:int(num_epochs // 2)], '-', color='blue',
             label=r'$I_{LT}$', linewidth=2)

    plt.xlabel(r'Training Epochs $(t)$')
    plt.ylabel(r'Avg. Updates')
    plt.legend()

    plt.savefig(
        '{0}{1}-bprmf-until{2}-sum-final-update.png'.format(path_output_rec_result,
                                                            path_output_rec_result.split('/')[-2],
                                                            num_epochs), format='png')

    plt.close()

    plt.figure()
    plt.plot(x_axes[int(num_epochs // 2):], y_axes_short_head[int(num_epochs // 2):], '-', color='orange',
             label=r'$I_{SH}$', linewidth=2)
    plt.plot(x_axes[int(num_epochs // 2):], y_axes_long_tail[int(num_epochs // 2):], '-', color='blue',
             label=r'$I_{LT}$', linewidth=2)

    plt.xlabel(r'Training Epochs $(t)$')
    plt.ylabel(r'Avg. Updates')
    plt.tight_layout()
    plt.legend()

    plt.savefig(
        '{0}{1}-amf-until{2}-sum-final-update.png'.format(path_output_rec_result,
                                                          path_output_rec_result.split('/')[-2],
                                                          num_epochs), format='png', bbox_inches='tight', pad_inches=0, dpi=600)


if __name__ == '__main__':
    run()
