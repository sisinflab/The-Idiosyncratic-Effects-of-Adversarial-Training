import sys
import os
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser.parsers import ttest_eval_bias_parse_args, print_args
import src.config.configs as cfg

from src.evaluation.read_results import read_prediction_lists, get_list_of_predictions, get_list_of_test, \
    get_list_of_training, read_data, get_data_statistics
from src.evaluation.metrics import compute_gini, get_head_tail_split, catalog_coverage, mark, novelty, \
    recommender_precision, recommender_recall, average_recommendation_popularity, average_percentage_of_long_tail_items, \
    average_coverage_of_long_tail_items, ranking_based_statistical_parity, ranking_based_equal_opportunity, ndcg_at, \
    get_stars


def run():
    args = ttest_eval_bias_parse_args()
    print_args(args)

    for dataset in args.datasets:

        df_evaluation_bias = pd.DataFrame()

        print('Start T-Test of Bias evaluation on Dataset: {0}'.format(dataset))

        cfg.output_rec_plot_dir, cfg.output_rec_bias_dir = cfg.output_rec_plot_dir.format(
            dataset), cfg.output_rec_bias_dir.format(dataset)

        train, test = read_data(dataset)

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

        path_rec_list = cfg.output_rec_list_dir.format(dataset)

        list_of_test = get_list_of_test(test)
        list_of_training = get_list_of_training(train, test[cfg.user_field].unique())
        for k in args.list_k:
            print('K={}'.format(k))

            # Evaluate_predictions_of_group_a
            print("\tStart Group A (Baseline)")
            name_of_prediction_list = args.file_a
            directory_of_trained_models = name_of_prediction_list.split('ep_')[0]
            print('\t\tEvaluate {}'.format(name_of_prediction_list))
            predictions = read_prediction_lists(
                os.path.join(os.path.join(path_rec_list, directory_of_trained_models, name_of_prediction_list)))

            list_of_predictions = get_list_of_predictions(predictions, test[cfg.user_field].unique(), k)

            # Precision and Recall
            prec, t_prec = recommender_precision(list_of_predictions, list_of_test)
            recall, t_recall = recommender_recall(list_of_predictions, list_of_test)

            # nDCG
            ndcg, t_ndcg = ndcg_at(list_of_predictions, list_of_test, k=k, assume_unique=True)

            # Novelty
            nov, t_nov = novelty(list_of_predictions, dict_item_pop, num_users, k)

            # Bias Measures

            # # Used by Himan Abdollahpouri
            # # # Average Recommendation Popularity(ARP):
            arp, t_arp = average_recommendation_popularity(list_of_predictions, dict_item_pop)
            # # # Average Percentage of Long Tail Items (APLT):
            aplt, t_aplt = average_percentage_of_long_tail_items(list_of_predictions, long_tail_items)
            # # # Average Coverage of Long Tail items (ACLT):
            aclt, t_aclt = average_coverage_of_long_tail_items(list_of_predictions, long_tail_items)

            # # Proposed by Ziwei Zhu et al. -> Lower values indicate the recommendations are less biased
            # # # Ranking-based Statistical Parity (RSP)
            p_pop, p_tail, rsp, t_p_pop, t_p_tail, t_rsp = ranking_based_statistical_parity(list_of_predictions,
                                                                                            list_of_training,
                                                                                            head_tail_items,
                                                                                            long_tail_items)
            # # # Ranking-based Equal Opportunity (REO)
            pc_pop, pc_tail, reo, t_pc_pop, t_pc_tail, t_reo = ranking_based_equal_opportunity(
                list_of_predictions, list_of_test,
                head_tail_items, long_tail_items)

            df_evaluation_bias = df_evaluation_bias.append(
                {
                    'Dataset': dataset,
                    'FileName': name_of_prediction_list,
                    'Model': name_of_prediction_list.split('_')[0],
                    'EmbK': int(name_of_prediction_list.split('_')[1].replace('emb', '')),
                    'TotEpoch': int(name_of_prediction_list.split('_')[2].replace('ep', '')),
                    'LearnRate': int(name_of_prediction_list.split('_')[3].replace('lr', '')),
                    'Epsilon': float(name_of_prediction_list.split('_')[4].replace('XX', '0').replace('eps', '')),
                    'Alpha': float(name_of_prediction_list.split('_')[5].replace('XX', '0').replace('alpha', '')),
                    'Epoch': int(name_of_prediction_list.split('_')[7]),
                    'Top-K': k,
                    'Precision': prec,
                    'Recall': recall,
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

            # Evaluate_predictions on groups B
            print("\tStart Groups B")
            for name_of_prediction_list in args.files_b:
                print('\t\tEvaluate {}'.format(name_of_prediction_list))
                directory_of_trained_models = name_of_prediction_list.split('ep_')[0]
                predictions = read_prediction_lists(
                    os.path.join(os.path.join(path_rec_list, directory_of_trained_models, name_of_prediction_list)))

                list_of_predictions = get_list_of_predictions(predictions, test[cfg.user_field].unique(), k)

                # Precision and Recall
                prec_b, t_prec_b = recommender_precision(list_of_predictions, list_of_test)
                recall_b, t_recall_b = recommender_recall(list_of_predictions, list_of_test)

                # nDCG
                ndcg_b, t_ndcg_b = ndcg_at(list_of_predictions, list_of_test, k=k, assume_unique=True)

                # Novelty
                nov_b, t_nov_b = novelty(list_of_predictions, dict_item_pop, num_users, k)

                # Bias Measures

                # # Used by Himan Abdollahpouri
                # # # Average Recommendation Popularity(ARP):
                arp_b, t_arp_b = average_recommendation_popularity(list_of_predictions, dict_item_pop)
                # # # Average Percentage of Long Tail Items (APLT):
                aplt_b, t_aplt_b = average_percentage_of_long_tail_items(list_of_predictions, long_tail_items)
                # # # Average Coverage of Long Tail items (ACLT):
                aclt_b, t_aclt_b = average_coverage_of_long_tail_items(list_of_predictions, long_tail_items)

                # # Proposed by Ziwei Zhu et al. -> Lower values indicate the recommendations are less biased
                # # # Ranking-based Statistical Parity (RSP)
                p_pop_b, p_tail_b, rsp_b, t_p_pop_b, t_p_tail_b, t_rsp_b = ranking_based_statistical_parity(
                    list_of_predictions,
                    list_of_training,
                    head_tail_items,
                    long_tail_items)
                # # # Ranking-based Equal Opportunity (REO)
                pc_pop_b, pc_tail_b, reo_b, t_pc_pop_b, t_pc_tail_b, t_reo_b = ranking_based_equal_opportunity(
                    list_of_predictions, list_of_test,
                    head_tail_items, long_tail_items)

                df_evaluation_bias = df_evaluation_bias.append(
                    {
                        'Dataset': dataset,
                        'FileName': name_of_prediction_list,
                        'Model': name_of_prediction_list.split('_')[0],
                        'EmbK': int(name_of_prediction_list.split('_')[1].replace('emb', '')),
                        'TotEpoch': int(name_of_prediction_list.split('_')[2].replace('ep', '')),
                        'LearnRate': int(name_of_prediction_list.split('_')[3].replace('lr', '')),
                        'Epsilon': float(name_of_prediction_list.split('_')[4].replace('XX', '0').replace('eps', '')),
                        'Alpha': float(name_of_prediction_list.split('_')[5].replace('XX', '0').replace('alpha', '')),
                        'Epoch': int(name_of_prediction_list.split('_')[7]),
                        'Top-K': k,
                        'Precision': prec_b,
                        'Recall': recall_b,
                        'nDCG': ndcg_b,
                        'Novelty': nov_b,
                        'ARP': arp_b,
                        'APLT': aplt_b,
                        'ACLT': aclt_b,
                        'P_Pop': p_pop_b,
                        'P_Tail': p_tail_b,
                        'RSP': rsp_b,
                        'PC_Pop': pc_pop_b,
                        'PC_Tail': pc_tail_b,
                        'REO': reo_b,
                        'T-Precision': get_stars(t_prec, t_prec_b),
                        'T-Recall': get_stars(t_recall, t_recall_b),
                        'T-nDCG': get_stars(t_ndcg, t_ndcg_b),
                        'T-Novelty': get_stars(t_nov, t_nov_b),
                        'T-ARP': get_stars(t_arp, t_arp_b),
                        'T-APLT': get_stars(t_aplt, t_aplt_b),
                        'T-ACLT': get_stars(t_aclt, t_aclt_b),
                        'T-P_Pop': get_stars(t_p_pop, t_p_pop_b),
                        'T-P_Tail': get_stars(t_p_tail, t_p_tail_b),
                        'T-RSP': get_stars(t_rsp, t_rsp_b),
                        'T-PC_Pop': get_stars(t_pc_pop, t_pc_pop_b),
                        'T-PC_Tail': get_stars(t_pc_tail, t_pc_tail_b),
                        'T-REO': get_stars(t_reo, t_reo_b)
                    }, ignore_index=True
                )

        df_evaluation_bias[cfg.column_order_ttest].to_csv(os.path.join(cfg.output_rec_bias_dir, cfg.ttest_bias_results),
                                                          index=None)

        print('Completed the T-Test of Bias evaluation on Dataset: {0}'.format(dataset))


if __name__ == '__main__':
    run()
