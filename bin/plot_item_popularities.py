import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser.parsers import eval_bias_parse_args, print_args
import src.config.configs as cfg

from src.evaluation.read_results import read_data, get_data_statistics
from src.evaluation.metrics import get_head_tail_split
from src.util.plot import plot_item_popularity


def run():
    args = eval_bias_parse_args()
    print_args(args)

    for dataset in ['movielens100k', 'amazon', 'lastfm', 'movielens', 'yelp']:

        print('Start PLOT item popularities on Dataset: {0}'.format(dataset))

        cfg.output_rec_plot_dir, cfg.output_rec_bias_dir = cfg.output_rec_plot_dir.format(
            dataset), cfg.output_rec_bias_dir.format(dataset)

        # manage_directories_evaluate_results(cfg.output_rec_plot_dir)
        # manage_directories_evaluate_results(cfg.output_rec_bias_dir)

        train, test = read_data(dataset)

        # Split in Short Head, Long Tail, and Distant Tail
        item_pop = train.groupby([cfg.item_field]).count().sort_values(cfg.user_field, ascending=False)[cfg.user_field]
        dict_item_pop = dict(zip(item_pop.index, item_pop.to_list()))

        num_users, num_items, num_ratings = get_data_statistics(train, test)

        # Add the items not in the training
        for item_id in set(range(num_items)).difference(set(item_pop.index)):
            dict_item_pop[item_id] = 0

        short_head_split = get_head_tail_split(item_pop, num_items)

        plot_item_popularity(item_pop, short_head_split,
                             os.path.join(cfg.output_rec_plot_dir, cfg.item_popularity_plot))


if __name__ == '__main__':
    run()
