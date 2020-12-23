import pandas as pd


import src.config.configs as cfg


def read_prediction_lists(path_prediction_file):
    df_pl = pd.read_csv(path_prediction_file, header=None, sep='\t')
    df_pl.columns = [cfg.user_field, cfg.item_field, cfg.score_field]
    return df_pl


def read_data(dataset):
    train = pd.read_csv(cfg.InputTrainFile.format(dataset), sep='\t', header=None)
    train.columns = [cfg.user_field, cfg.item_field, cfg.score_field, cfg.time_field]
    test = pd.read_csv(cfg.InputTestFile.format(dataset), sep='\t', header=None)
    test.columns = [cfg.user_field, cfg.item_field, cfg.score_field, cfg.time_field]

    return train, test


def get_data_statistics(train, test):
    data = train.copy()
    data = data.append(test, ignore_index=True)
    return data[cfg.user_field].nunique(), data[cfg.item_field].nunique(), len(train)


def get_list_of_predictions(predictions, test_users, k):
    list_of_lists = []
    for u in sorted(predictions[cfg.user_field].unique()):
        if u in test_users:
            list_of_lists.append(predictions[predictions[cfg.user_field] == u][cfg.item_field].to_list()[:k])
    return list_of_lists


def get_list_of_predictions(predictions, test_users, k):
    list_of_lists = []
    for u in sorted(predictions[cfg.user_field].unique()):
        if u in test_users:
            list_of_lists.append(predictions[predictions[cfg.user_field] == u][cfg.item_field].to_list()[:k])
    return list_of_lists


def get_list_of_training(train, test_users):
    list_of_lists = []
    for u in sorted(train[cfg.user_field].unique()):
        if u in test_users:
            list_of_lists.append(train[train[cfg.user_field] == u][cfg.item_field].to_list())
    return list_of_lists


def get_list_of_test(test):
    list_of_lists = []
    for u in sorted(test[cfg.user_field].unique()):
        list_of_lists.append(test[test[cfg.user_field] == u][cfg.item_field].to_list())
    return list_of_lists


def read_embeddings():
    pass
