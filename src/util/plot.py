import math
import pickle
import pandas as pd
import scipy as sp

import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import src.config.configs as cfg


def regress(x, y):
    """Return a tuple of predicted y values and parameters for linear regression."""
    p = sp.stats.linregress(x, y)
    b1, b0, r, p_val, stderr = p
    y_pred = sp.polyval([b1, b0], x)
    return y_pred, p


def preprocess_x_y(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    df = df.groupby(['x']).mean('y').reset_index()
    return df['x'].to_list(), df['y'].to_list()


def plot_item_popularity(item_pop, head_tail_split, path_name):
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(6, 3.5))

    plt.xlabel('Items ranked by popularity')
    plt.ylabel('Number of feedback')
    plt.plot(range(head_tail_split), item_pop.values[:head_tail_split], '-', lw=3, alpha=0.7, label='Short Head')
    plt.plot(range(head_tail_split, len(item_pop.index)), item_pop.values[head_tail_split:], '--', lw=3, label='Long Tail')
    plt.axvline(x=head_tail_split, linestyle='--', lw=1, c='grey')
    plt.xlim([-25, len(item_pop.index)])
    plt.ylim([-25, item_pop.values[0]])
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_name, format='png', bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()


def plot_item_popularity_by_recommendation_frequency(key, original, predictions, item_pop, dataset, num_users, k):
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 3.5))

    plt.xlabel('Average item popularity by users')
    plt.ylabel('Recommendation frequency')

    pred_item_pop = predictions.groupby([cfg.item_field]).count().sort_values(cfg.user_field, ascending=False)[
        cfg.user_field]
    dict_pred_item_pop = dict(zip(pred_item_pop.index, pred_item_pop.to_list()))

    x = sorted(item_pop.to_list())
    y = []
    for ele in item_pop.index[::-1]:
        try:
            y.append(dict_pred_item_pop[ele])
        except:
            y.append(0)
    x = np.array(x) / num_users
    plt.scatter(x, y, label='AMF' if dataset in ['movielens', 'lastfm'] else 'AMR', alpha=0.75)

    pred_item_pop = original.groupby([cfg.item_field]).count().sort_values(cfg.user_field, ascending=False)[
        cfg.user_field]
    dict_pred_item_pop = dict(zip(pred_item_pop.index, pred_item_pop.to_list()))

    x = sorted(item_pop.to_list())
    y = []
    for ele in item_pop.index[::-1]:
        try:
            y.append(dict_pred_item_pop[ele])
        except:
            y.append(0)
    x = np.array(x) / num_users
    plt.scatter(x, y, label='BPR-MF' if dataset in ['movielens', 'lastfm'] else 'VBPR', alpha=0.5)

    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.show()
    plt.savefig('../measures/{0}/{1}_{2}_item_popularity_by_rec_frequency_at_{3}.pdf'.format(dataset, dataset, key, k))
    plt.close()


def plot_embedding_norm_by_item_popularity(q_embedding_base, item_pop, q_embedding, num_users, dataset):
    plt.figure()

    reversed_item_pop = item_pop[::-1]
    y = []
    y2 = []
    y_delta = []

    for index in reversed_item_pop.index:
        y.append(np.linalg.norm(q_embedding[index]))
        y2.append(np.linalg.norm(q_embedding_base[index]))
        y_delta.append(sum(q_embedding[index] - q_embedding_base[index]))

    plt.xlabel('Items')
    plt.ylabel('Embedding Norm')

    plt.scatter(range(len(item_pop)), y, label='Adv. Trained')

    plt.scatter(range(len(item_pop)), y2, label='Original')

    plt.scatter(range(len(item_pop)), y_delta, label='DELTA')

    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.show()
    plt.savefig('../measures/{0}/embedding_norm_by_item_popularity.pdf'.format(dataset))
    plt.close()


def plot_item_popularity_by_recommendation_score(key, original, predictions, item_pop, dataset, num_users, k,
                                                 head_tail_items, long_tail_items, emb_original, emb_advers_eps10):
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 3.5))

    adv = np.dot(emb_advers_eps10[1], emb_advers_eps10[0].T)
    org = np.dot(emb_original[1], emb_original[0].T)
    delta = adv - org

    x = item_pop.index[::-1]
    y = []
    for item in x:
        y.append(np.mean(delta[:, item]))

    plt.xlabel('Item Popularity')
    plt.ylabel('Score Variation')

    plt.plot(range(len(x)), y)

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    plt.savefig('../measures/{0}/{1}_{2}_item_popularity_by_rec_scores_at_{3}.pdf'.format(dataset, dataset, key, k))
    plt.close()
