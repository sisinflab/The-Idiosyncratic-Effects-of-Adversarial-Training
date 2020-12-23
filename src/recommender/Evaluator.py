import heapq

import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import math
from time import time
import datetime

_feed_dict = None
_dataset = None
_model = None
_sess = None
_K = None


def _init_eval_model(data):
    global _dataset
    _dataset = data

    pool = Pool(cpu_count() - 1)
    # feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))
    feed_dicts = pool.map(_evaluate_input_list, range(_dataset.num_users))

    pool.close()
    pool.join()

    return feed_dicts


def _evaluate_input_list(user):
    test_items = _dataset.test_list[user]

    if len(test_items) > 0:
        item_input = set(range(_dataset.num_items)) - set(_dataset.train_list[user])

        for test_item in test_items:
            if test_item in item_input:
                item_input.remove(test_item)

        item_input = list(item_input)

        for test_item in test_items:
            item_input.append(test_item)

        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input
    else:
        print('User {} has no test list!'.format(user))
        return 0, 0


def _evaluate_input(user):
    # generate items_list
    try:
        test_item = _dataset.test_list[user][1]
        item_input = set(range(_dataset.num_items)) - set(_dataset.train_list[user])
        if test_item in item_input:
            item_input.remove(test_item)
        item_input = list(item_input)
        item_input.append(test_item)
        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input
    except:
        print('******' + user)
        return 0, 0


def _eval_by_user(user, curr_pred):
    # get predictions of data in testing set
    user_input, item_input = _feed_dicts[user]
    if type(user_input) != np.ndarray:
        return ()

    predictions = curr_pred[list(item_input.reshape(-1))]
    neg_predict, pos_predict = predictions[:-len(_dataset.test_list[user])], \
                               predictions[-len(_dataset.test_list[user]):]

    position = 0
    for t in range(len(_dataset.test_list[user])):
        position += (neg_predict >= pos_predict[t]).sum()

    # calculate from HR@1 to HR@10, and from NDCG@1 to NDCG@100, AUC
    hr, ndcg, auc, prec, rec = [], [], [], [], []
    # for k in range(1, _K + 1):
    k = _K

    # HIT RATIO (HR)
    item_score = {}
    for i in list(item_input.reshape(-1)):
        item_score[i] = curr_pred[i]

    k_max_item_score = heapq.nlargest(k, item_score, key=item_score.get)

    r = []
    for i in k_max_item_score:
        if i in item_input[-len(_dataset.test_list[user]):]:
            r.append(1)
        else:
            r.append(0)

    hr.append(1. if sum(r) > 0 else 0.)

    # NDCG
    ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)

    # AREA UNDER CURVE (AUC)
    auc.append(1 - (position / (len(neg_predict) * len(pos_predict))))

    # PRECISION (P)
    prec.append(sum(r) / len(r))

    # RECALL (R)
    rec.append(sum(r) / len(pos_predict))

    return hr, ndcg, auc, prec, rec


class Evaluator:
    def __init__(self, model, data, k):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.data = data
        self.k = k
        self.eval_feed_dicts = _init_eval_model(data)
        self.model = model

    def eval(self, epoch=0, results={}, epoch_text='', start_time=0):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        global _model
        global _K
        global _dataset
        global _feed_dicts
        _dataset = self.data
        _model = self.model
        _K = self.k
        _feed_dicts = self.eval_feed_dicts

        eval_start_time = time()
        all_predictions = self.model.predict_all().numpy()

        res = []
        for user in range(self.model.data.num_users):
            current_prediction = all_predictions[user, :]
            res.append(_eval_by_user(user, current_prediction))

        hr, ndcg, auc, prec, rec = (np.array(res).mean(axis=0)).tolist()
        print_results = "Train Time: %s \tInference Time: %s\n\tMetrics@%d\n\t\tHR\tnDCG\tAUC\tPrec\tRec\n\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
            datetime.timedelta(seconds=(time() - start_time)),
            datetime.timedelta(seconds=(time() - eval_start_time)),
            _K,
            # hr[_K - 1], ndcg[_K - 1], auc[_K - 1], prec[_K - 1], rec[_K - 1]
            hr[0], ndcg[0], auc[0], prec[0], rec[0]
        )

        print(print_results)

        if len(epoch_text) != '':
            # results is store in the results object passed as parameter
            results[epoch] = {'hr': hr, 'auc': auc, 'p': prec, 'r': rec, 'ndcg': ndcg}

        return print_results

    def store_recommendation(self, epoch=None, attack_name=""):
        """
        Store recommendation list (top-k) in order to be used for the ranksys framework (anonymized)
        attack_name: The name for the attack stored file
        :return:
        """
        results = self.model.predict_all().numpy()
        with open('{0}{1}ep_{2}_best{3}_top{4}_rec.tsv'.format(self.model.path_output_rec_list,
                                                               attack_name + self.model.path_output_rec_list.split('/')[-2],
                                                               epoch,
                                                               self.model.best,
                                                               self.k),
                  'w') as out:
            for u in range(results.shape[0]):
                results[u][self.data.train_list[u]] = -np.inf
                top_k_id = results[u].argsort()[-self.k:][::-1]
                top_k_score = results[u][top_k_id]
                for i, value in enumerate(top_k_id):
                    out.write(str(u) + '\t' + str(value) + '\t' + str(top_k_score[i]) + '\n')
