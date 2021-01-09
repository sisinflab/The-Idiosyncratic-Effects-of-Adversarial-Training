import sys
import os
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser.parsers import eval_bias_parse_args, print_args
from src.dataset.dataset import DataLoader
from src.util.dir_manager import manage_directories_evaluate_results, get_paths
from src.util.general import get_model
import src.config.configs as cfg

from src.evaluation.read_results import read_prediction_lists, get_list_of_predictions, get_list_of_test, \
    get_list_of_training, read_data, get_data_statistics
from src.evaluation.metrics import compute_gini, get_head_tail_split, catalog_coverage, mark, novelty, \
    recommender_precision, recommender_recall, average_recommendation_popularity, average_percentage_of_long_tail_items, \
    average_coverage_of_long_tail_items, ranking_based_statistical_parity, ranking_based_equal_opportunity, ndcg_at
from src.util.plot import plot_item_popularity, plot_item_popularity_by_recommendation_frequency, \
    plot_embedding_norm_by_item_popularity, plot_item_popularity_by_recommendation_score


def run():
    args = eval_bias_parse_args()
    print_args(args)

    for dataset in args.datasets:

        df_evaluation_bias = pd.DataFrame()

        print('Start Bias evaluation on Dataset: {0}'.format(dataset))

        cfg.output_rec_plot_dir, cfg.output_rec_bias_dir = cfg.output_rec_plot_dir.format(
            dataset), cfg.output_rec_bias_dir.format(dataset)

        manage_directories_evaluate_results(cfg.output_rec_plot_dir)
        manage_directories_evaluate_results(cfg.output_rec_bias_dir)

        train, test = read_data(dataset)

        catalog = sorted(train[cfg.item_field].unique())

        # Split in Short Head, Long Tail, and Distant Tail
        item_pop = train.groupby([cfg.item_field]).count().sort_values(cfg.user_field, ascending=False)[cfg.user_field]
        dict_item_pop = dict(zip(item_pop.index, item_pop.to_list()))

        num_users, num_items, num_ratings = get_data_statistics(train, test)

        # Add the items not in the training
        for item_id in set(range(num_items)).difference(set(item_pop.index)):
            dict_item_pop[item_id] = 0

        head_tail_split = get_head_tail_split(item_pop, num_items)

        head_tail_items = np.array(item_pop[:head_tail_split].index)
        long_tail_items = np.array(item_pop[head_tail_split:].index)

        # Compute Gini Index of Rated Items
        print('Head Tail', compute_gini(item_pop[:head_tail_split]))
        print('Long Tail', compute_gini(item_pop[head_tail_split:]))

        # plot_item_popularity(item_pop, head_tail_split, os.path.join(cfg.output_rec_plot_dir, cfg.item_popularity_plot))

        path_rec_list = cfg.output_rec_list_dir.format(dataset)

        list_of_test = get_list_of_test(test)
        list_of_training = get_list_of_training(train, test[cfg.user_field].unique())

        for directory_of_trained_models in os.listdir(path_rec_list):
            for name_of_prediction_list in os.listdir(os.path.join(path_rec_list, directory_of_trained_models)):
                if 'fgsm' not in name_of_prediction_list:
                    print('\tEvaluate {} on'.format(name_of_prediction_list))
                    predictions = read_prediction_lists(
                        os.path.join(os.path.join(path_rec_list, directory_of_trained_models, name_of_prediction_list)))
                    for k in args.list_k:
                        print('\t\tK={}'.format(k))
                        list_of_predictions = get_list_of_predictions(predictions, test[cfg.user_field].unique(), k)

                        # Coverage
                        coverage, percentage_of_coverage = catalog_coverage(list_of_predictions, catalog, k)

                        # Precision and Recall
                        prec, _ = recommender_precision(list_of_predictions, list_of_test)
                        recall, _ = recommender_recall(list_of_predictions, list_of_test)

                        # Mean Average Recall
                        mar = mark(list_of_test, list_of_predictions, k=10)

                        # nDCG
                        ndcg, _ = ndcg_at(list_of_predictions, list_of_test, k=k, assume_unique=True)

                        # Novelty
                        nov, _ = novelty(list_of_predictions, dict_item_pop, num_users, k)

                        # Bias Measures

                        # # Used by Himan Abdollahpouri
                        # # # Average Recommendation Popularity(ARP):
                        arp, _ = average_recommendation_popularity(list_of_predictions, dict_item_pop)
                        # # # Average Percentage of Long Tail Items (APLT):
                        aplt, _ = average_percentage_of_long_tail_items(list_of_predictions, long_tail_items)
                        # # # Average Coverage of Long Tail items (ACLT):
                        aclt, _ = average_coverage_of_long_tail_items(list_of_predictions, long_tail_items)

                        # # Proposed by Ziwei Zhu et al. -> Lower values indicate the recommendations are less biased
                        # # # Ranking-based Statistical Parity (RSP)
                        p_pop, p_tail, rsp, _, _, _ = ranking_based_statistical_parity(list_of_predictions, list_of_training,
                                                                              head_tail_items, long_tail_items)
                        # # # Ranking-based Equal Opportunity (REO)
                        pc_pop, pc_tail, reo, _, _, _ = ranking_based_equal_opportunity(list_of_predictions, list_of_test,
                                                                               head_tail_items, long_tail_items)

                        # if 'bprmf' in name_of_prediction_list:
                        #     plot_item_popularity_by_recommendation_frequency(name_of_prediction_list, original, predictions, item_pop, dataset, num_users, k)

                        df_evaluation_bias = df_evaluation_bias.append(
                            {
                                'Dataset': dataset,
                                'FileName': name_of_prediction_list,
                                'Model': name_of_prediction_list.split('_')[0],
                                'EmbK': int(name_of_prediction_list.split('_')[1].replace('emb', '')),
                                'TotEpoch': int(name_of_prediction_list.split('_')[2].replace('ep', '')),
                                'LearnRate': float(name_of_prediction_list.split('_')[3].replace('lr', '')),
                                'Epsilon': float(
                                    name_of_prediction_list.split('_')[4].replace('XX', '0').replace('eps', '')),
                                'Alpha': float(
                                    name_of_prediction_list.split('_')[5].replace('XX', '0').replace('alpha', '')),
                                'Epoch': int(name_of_prediction_list.split('_')[7]),
                                'Top-K': k,
                                'Coverage': coverage,
                                'Coverage[%]': percentage_of_coverage,
                                'Precision': prec,
                                'Recall': recall,
                                'MAR': mar,
                                'nDCG': ndcg,
                                'Novelty': nov,
                                'ARP': arp,
                                'APLT': aplt,
                                'ACLT': aclt,
                                'P_Pop': p_pop,
                                'P_Tail': p_tail,
                                'RSP': rsp,
                                'PC_Pop': pc_pop,
                                'PC_Tail': pc_tail,
                                'REO': reo
                            }, ignore_index=True
                        )

        df_evaluation_bias[cfg.column_order].to_csv(os.path.join(cfg.output_rec_bias_dir, cfg.bias_results), index=None)

        print('Completed the Bias evaluation on Dataset: {0}'.format(dataset))


if __name__ == '__main__':
    run()
