import pandas as pd
import numpy as np

from src.data.read import read_prediction_lists, get_list_of_predictions, get_list_of_test, get_list_of_training, \
    read_embeddings
from src.evaluation.metrics import compute_gini, get_head_tail_split, catalog_coverage, mark, novelty, \
    recommender_precision, recommender_recall, average_recommendation_popularity, average_percentage_of_long_tail_items, \
    average_coverage_of_long_tail_items, ranking_based_statistical_parity, ranking_based_equal_opportunity, ndcg_at
from src.utils.utils import plot_item_popularity, plot_item_popularity_by_recommendation_frequency, \
    plot_embedding_norm_by_item_popularity, plot_item_popularity_by_recommendation_score

user_field = 'user_id'
item_field = 'item_id'

results = 'Dataset\tModel\tTOP-K\tcoverage\tperc_coverage\tprec\trecall\tmar\tnDCG\tnovelty\tarp\taplt\taclt\tp_pop\tp_tail\trsp\tpc_pop\tpc_tail\treo'
print(results)

for dataset in ['movielens', 'lastfm', 'amazon_women', 'amazon_men']:

    print('Dataset: {0}'.format(dataset))

    train = pd.read_csv('../dataset/{0}/trainingset.tsv'.format(dataset), sep='\t', header=None)
    train.columns = [user_field, item_field]
    test = pd.read_csv('../dataset/{0}/testset.tsv'.format(dataset), sep='\t', header=None)
    test.columns = [user_field, item_field]

    catalog = sorted(train[item_field].unique())

    # Split in Short Head, Long Tail, and Distant Tail
    # The first vertical line separates the top 20% of items by popularity – these items cumulatively have many more ratings than the 80% tail items to the right.

    item_pop = train.groupby([item_field]).count().sort_values(user_field, ascending=False)[user_field]
    item_pop.head()
    dict_item_pop = dict(zip(item_pop.index, item_pop.to_list()))

    num_ratings = sum(np.array(item_pop))
    num_users = train[user_field].nunique()
    num_items = train[item_field].nunique()

    head_tail_split = get_head_tail_split(item_pop)

    head_tail_items = np.array(item_pop[:head_tail_split].index)
    long_tail_items = np.array(item_pop[head_tail_split:].index)

    print('Head Tail', compute_gini(item_pop[:head_tail_split]))
    print('Long Tail', compute_gini(item_pop[head_tail_split:]))

    plot_item_popularity(item_pop, head_tail_split, dataset)

    # Evaluate Metrics

    original, advers_eps05, advers_eps10, advers_eps20 = read_prediction_lists(dataset)

    list_of_test = get_list_of_test(test)
    list_of_training = get_list_of_training(train, test[user_field].unique())

    res = {'original': original, 'advers_eps05': advers_eps05, 'advers_eps10': advers_eps10}

    for key, predictions in res.items():

        for k in [5, 10, 15]:
            list_of_predictions = get_list_of_predictions(predictions, test[user_field].unique(), k)

            # COVERAGE
            coverage, percentage_of_coverage = catalog_coverage(list_of_predictions, catalog, k)

            # # Recommender Precision and Recall
            prec, recall = recommender_precision(list_of_predictions, list_of_test), recommender_recall(
                list_of_predictions, list_of_test)

            ndcg = ndcg_at(list_of_predictions, list_of_test, k=k, assume_unique=True)

            # # Mean Average Recall
            mar = mark(list_of_test, list_of_predictions, k=10)

            # # Novelty
            nov, mean_self_information = novelty(list_of_predictions, dict_item_pop, num_users, k)

            # Bias Measures

            # # Used by Himan Abdollahpouri
            # # # Average Recommendation Popularity(ARP):
            arp = average_recommendation_popularity(list_of_predictions, dict_item_pop)
            # # # Average Percentage of Long Tail Items (APLT):
            aplt = average_percentage_of_long_tail_items(list_of_predictions, long_tail_items)
            # # # Average Coverage of Long Tail items (ACLT):
            aclt = average_coverage_of_long_tail_items(list_of_predictions, long_tail_items)

            # # Proposed by Ziwei Zhu et al. -> Lower values indicate the recommendations are less biased
            # # # Ranking-based Statistical Parity (RSP)
            p_pop, p_tail, rsp = ranking_based_statistical_parity(list_of_predictions, list_of_training, head_tail_items,
                                                                  long_tail_items)
            # Ranking-based Equal Opportunity (REO)
            pc_pop, pc_tail, reo = ranking_based_equal_opportunity(list_of_predictions, list_of_test, head_tail_items,
                                                                   long_tail_items)

            # Correlation Plots of item popularity and recommendation frequency.
            if key != 'original':
                plot_item_popularity_by_recommendation_frequency(key, original, predictions, item_pop, dataset,
                                                             num_users, k)

            result = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\t{18}'.format(
                dataset,
                key,
                k,
                coverage,
                percentage_of_coverage,
                prec,
                recall,
                mar,
                ndcg,
                nov,
                arp,
                aplt,
                aclt,
                p_pop,
                p_tail,
                rsp,
                pc_pop,
                pc_tail,
                reo)

            results += "\n" + result

            print(result)

with open('results.tsv', 'w') as f:
    f.write(results)