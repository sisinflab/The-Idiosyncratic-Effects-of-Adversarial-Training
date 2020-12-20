import warnings

import numpy as np


def get_head_tail_split(item_pop):
    """
    Get the index of the short head limit (80% of ratings)
    :param item_pop:
    :return:
    """
    eighty_percent = sum(np.array(item_pop)) * 0.8
    for i, cnt in enumerate(np.array(item_pop)):
        eighty_percent -= cnt
        if eighty_percent <= 0:
            return i


def catalog_coverage(predicted, catalog, k):
    """
    Computes the catalog coverage for k lists of recommendations
    Parameters
    ----------
    predicted : a list of predictions
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    k: integer
        The number of observed recommendation lists
        which randomly choosed in our offline setup
    Returns
    ----------
    catalog_coverage:
        Number of Recommended Items
    percentage_catalog_coverage:
        The catalog coverage of the recommendations as a percent
        rounded to 2 decimal places
    ----------
    """
    predicted_flattened = [p for sublist in predicted for p in sublist]
    L_predictions = len(set(predicted_flattened))
    catalog_coverage = round(L_predictions / (len(catalog) * 1.0) * 100, 2)
    return L_predictions, catalog_coverage


def compute_gini(x, w=None):
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def _ark(actual, predicted, k=10):
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : int
        The average recall at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / len(actual)


def mark(actual, predicted, k=10):
    """
    Computes the mean average recall at k.
    Parameters
    ----------
    actual : a list of test predictions
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of predictions
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: int
            The mean average recall at k (mar@k)
    """
    return np.mean([_ark(a, p, k) for a, p in zip(actual, predicted)])


def novelty(predicted, pop, u, n):
    """
    Computes the novelty for a list of recommendations
    Parameters
    ----------
    predicted : a list of predictions
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    pop: dictionary
        A dictionary of all items alongside of its occurrences counter in the training data
        example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
    u: integer
        The number of users in the training data
    n: integer
        The length of recommended lists per user
    Returns
    ----------
    novelty:
        The novelty of the recommendations in system level
    mean_self_information:
        The novelty of the recommendations in recommended top-N list level
    ----------
    Metric Defintion:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """
    mean_self_information = []
    k = 0
    for sublist in predicted:
        self_information = 0
        k += 1
        for i in sublist:
            self_information += np.sum(-np.log2(pop[i] / u))
        mean_self_information.append(self_information / n)
    novelty = sum(mean_self_information) / k
    return novelty, mean_self_information


def recommender_precision(predicted, actual):
    """
    Computes the precision of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of test predictions
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of predictions
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        precision: int
    """

    def calc_precision(predicted, actual):
        prec = [value for value in predicted if value in actual]
        prec = np.round(float(len(prec)) / float(len(predicted)), 4)
        return prec

    precision = np.mean(list(map(calc_precision, predicted, actual)))
    return precision


def recommender_recall(predicted, actual):
    """
    Computes the recall of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of test predictions
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of predictions
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        recall: int
    """

    def calc_recall(predicted, actual):
        reca = [value for value in predicted if value in actual]
        reca = np.round(float(len(reca)) / float(len(actual)), 4)
        return reca

    recall = np.mean(list(map(calc_recall, predicted, actual)))
    return recall


def average_recommendation_popularity(predicted, pop):
    """
        Computes the average popularity of the recommended items in each list
        ----------
        predicted : a list of predictions
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        pop: dictionary
            A dictionary of all items alongside of its occurrences counter in the training data
            example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
        Returns:
        -------
            arp: Average Recommendation Popularity (ARP)
        -------
        Metric Definition:
            Himan Abdollahpouri, Robin Burke, Bamshad Mobasher
            Managing Popularity Bias in Recommender Systems with Personalized Re-Ranking. FLAIRS Conference 2019
        """

    arp = []
    for L_u in predicted:
        phi_u = 0
        for i in L_u:
            phi_u += pop[i]
        arp.append(phi_u / len(L_u))
    arp = sum(arp) / len(predicted)  # len(predicted)  is the number of users in the test set.

    return arp


def average_percentage_of_long_tail_items(predicted, long_tail_items):
    """
        Computes  the average percentage of long tail items in the recommended lists
        ----------
        predicted : a list of predictions
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        long_tail_items: list
            A dictionary of all items alongside of its occurrences counter in the training data
            example: ['X', 'Y']
        Returns:
        -------
            aplt: Average Percentage of Long Tail Items (APLT)
        -------
        Metric Definition:
            Himan Abdollahpouri, Robin Burke, Bamshad Mobasher
            Managing Popularity Bias in Recommender Systems with Personalized Re-Ranking. FLAIRS Conference 2019
        """

    aplt = []
    for L_u in predicted:
        long_tail_items_in_u = len(list(set(L_u) & set(long_tail_items)))
        aplt.append(long_tail_items_in_u / len(L_u))
    aplt = sum(aplt) / len(predicted)  # len(predicted)  is the number of users in the test set.

    return aplt


def average_coverage_of_long_tail_items(predicted, long_tail_items):
    """
        Evaluate how much exposure long-tail items get in the entire recommendation
        ----------
        predicted : a list of predictions
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        long_tail_items: list
            A dictionary of all items alongside of its occurrences counter in the training data
            example: ['X', 'Y']
        Returns:
        -------
            aclt: Average Coverage of Long Tail items (ACLT)
        -------
        Metric Definition:
            Himan Abdollahpouri, Robin Burke, Bamshad Mobasher
            Managing Popularity Bias in Recommender Systems with Personalized Re-Ranking. FLAIRS Conference 2019
        """

    aclt = []
    for L_u in predicted:
        long_tail_items_in_u = len(list(set(L_u) & set(long_tail_items)))
        aclt.append(long_tail_items_in_u)
    aclt = sum(aclt) / len(predicted)  # len(predicted)  is the number of users in the test set.

    return aclt


def ranking_based_statistical_parity(list_of_predictions, list_of_training, head_tail_items, long_tail_items):
    """
        Evaluate how much exposure long-tail items get in the entire recommendation
        ----------
        list_of_predictions : a list of predictions
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        list_of_training : a list of predictions
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        long_tail_items: list
            A dictionary of all items alongside of its occurrences counter in the training data
            example: ['X', 'Y']
        Returns:
        -------
            rsp: Ranking-based Statistical Parity (RSP)
        -------
        Metric Definition:
            Ziwei Zhu, Jianling Wang, James Caverlee
            Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems. SIGIR 2020
        """

    numerators = [0, 0]
    for L_u in list_of_predictions:
        long_tail_items_in_u = len(list(set(L_u) & set(long_tail_items)))
        short_tail_items_in_u = len(list(set(L_u) & set(head_tail_items)))
        numerators[0] += short_tail_items_in_u
        numerators[1] += long_tail_items_in_u

    denominators = [0, 0]
    for T_u in list_of_training:
        long_tail_items_in_u = len(list(set(T_u) | set(long_tail_items)))
        short_tail_items_in_u = len(list(set(T_u) | set(head_tail_items)))
        denominators[0] += short_tail_items_in_u
        denominators[1] += long_tail_items_in_u

    ps = [n / d for n, d in zip(numerators, denominators)]

    rsp = np.std(ps) / np.mean(ps)

    return ps[0], ps[1], rsp


def ranking_based_equal_opportunity(list_of_predictions, list_of_test, head_tail_items, long_tail_items):
    """
        Evaluate the probability of being ranked in top-𝑘 given the ground-truth that the user likes the item
        ----------
        list_of_predictions : a list of predictions
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        list_of_training : a list of predictions
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        list_of_test : a list of predictions
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        long_tail_items: list
            A dictionary of all items alongside of its occurrences counter in the training data
            example: ['X', 'Y']
        Returns:
        -------
            reo: Ranking-based Equal Opportunity (REO)
        -------
        Metric Definition:
            Ziwei Zhu, Jianling Wang, James Caverlee
            Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems. SIGIR 2020
        """

    numerators = [0, 0]  # counts how many items in test set from group-a are ranked in top-𝑘 for user u
    for index, L_u in enumerate(list_of_predictions):
        long_tail_items_in_u = len(list(set(L_u) & set(long_tail_items) & set(list_of_test[index])))
        short_tail_items_in_u = len(list(set(L_u) & set(head_tail_items) & set(list_of_test[index])))
        numerators[0] += short_tail_items_in_u
        numerators[1] += long_tail_items_in_u

    denominators = [0, 0]  # counts the total number of items from group-a in test set for user u.
    for index, T_u in enumerate(list_of_test):
        long_tail_items_in_u = len(list(set(T_u) & set(long_tail_items)))
        short_tail_items_in_u = len(list(set(T_u) & set(head_tail_items)))
        denominators[0] += short_tail_items_in_u
        denominators[1] += long_tail_items_in_u

    ps = [n / d for n, d in zip(numerators, denominators)]

    reo = np.std(ps) / np.mean(ps)

    return ps[0], ps[1], reo


def _warn_for_empty_labels():
    """Helper for missing ground truth sets"""
    warnings.warn("Empty ground truth set! Check input data")
    return 0.


def _mean_ranking_metric(predictions, labels, metric):
    """Helper function for precision_at_k and mean_average_precision"""
    return np.mean([
        metric(np.asarray(prd), np.asarray(labels[i]))
        for i, prd in enumerate(predictions)  # lazy eval if generator
    ])


def _require_positive_k(k):
    """Helper function to avoid copy/pasted code for validating K"""
    if k <= 0:
        raise ValueError("ranking position k should be positive")


def ndcg_at(predictions, labels, k=10, assume_unique=True):
    """Compute the normalized discounted cumulative gain at K.
    Compute the average NDCG value of all the queries, truncated at ranking
    position k. The discounted cumulative gain at position k is computed as:
        sum,,i=1,,^k^ (2^{relevance of ''i''th item}^ - 1) / log(i + 1)
    and the NDCG is obtained by dividing the DCG value on the ground truth set.
    In the current implementation, the relevance value is binary.
    If a query has an empty ground truth set, zero will be used as
    NDCG together with a warning.
    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    k : int, optional (default=10)
        The rank at which to measure the NDCG.
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> ndcg_at(preds, labels, 3)
    0.3333333432674408
    >>> ndcg_at(preds, labels, 10)
    0.48791273434956867
    References
    ----------
    .. [1] K. Jarvelin and J. Kekalainen, "IR evaluation methods for
           retrieving highly relevant documents."
    """
    # validate K
    _require_positive_k(k)

    def _inner_ndcg(pred, lab):
        if lab.shape[0]:
            # if we do NOT assume uniqueness, the set is a bit different here
            if not assume_unique:
                lab = np.unique(lab)

            n_lab = lab.shape[0]
            n_pred = pred.shape[0]
            n = min(max(n_pred, n_lab), k)  # min(min(p, l), k)?

            # similar to mean_avg_prcsn, we need an arange, but this time +2
            # since python is zero-indexed, and the denom typically needs +1.
            # Also need the log base2...
            arange = np.arange(n, dtype=np.float32)  # length n

            # since we are only interested in the arange up to n_pred, truncate
            # if necessary
            arange = arange[:n_pred]
            denom = np.log2(arange + 2.)  # length n
            gains = 1. / denom  # length n

            # compute the gains where the prediction is present in the labels
            dcg_mask = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            dcg = gains[dcg_mask].sum()

            # the max DCG is sum of gains where the index < the label set size
            max_dcg = gains[arange < n_lab].sum()
            return dcg / max_dcg

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_ndcg)
