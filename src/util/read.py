import pandas as pd
import configparser
import pickle
import numpy as np
import os


def read_csv(filename):
    """
    Args:
        filename (str): csv file path

    Return:
         A pandas dataframe.
    """
    df = pd.read_csv(filename, index_col=False)
    return df


def read_np(filename):
    """
    Args:
        filename (str): filename of numpy to load
    Return:
        The loaded numpy.
    """
    return np.load(filename)


def read_imagenet_classes_txt(filename):
    """
    Args:
        filename (str): txt file path

    Return:
         A list with 1000 imagenet classes as strings.
    """
    with open(filename) as f:
        idx2label = eval(f.read())

    return idx2label


def read_config(sections_fields):
    """
    Args:
        sections_fields (list): list of fields to retrieve from configuration file

    Return:
         A list of configuration values.
    """
    config = configparser.ConfigParser()
    config.read('./../config/configs.ini')
    configs = []
    for s, f in sections_fields:
        configs.append(config[s][f])
    return configs


def load_obj(name):
    """
    Load the pkl object by name
    :param name: name of file
    :return:
    """
    with open(name, 'rb') as f:
        return pickle.load(f)


def find_checkpoint(dir, restore_epochs, epochs, rec, best=0, params=None):
    """

    :param dir: directory of the model where we start from the reading.
    :param restore_epochs: epoch from which we start from.
    :param epochs: epochs from which we restore (0 means that we have best)
    :param rec: recommender model
    :param best: 0 No Best - 1 Search for the Best
    :return:
    """
    if best:
        for r, d, f in os.walk(dir):
            for file in f:
                if 'best-weights-'.format(restore_epochs) in file:
                    return dir + file.split('.')[0]
        return ''

    if rec == 'amf' and restore_epochs < epochs:
        # We have to restore from an execution of bprmf
        dir_stored_models = os.walk('/'.join(dir.split('/')[:-2]))
        for dir_stored_model in dir_stored_models:
            embed_size, epochs, lr = params
            bprmf_path = '{0}{1}_{2}_{3}_{4}_{5}_{6}_{7}'.format('', 'bprmf', 'emb' + str(embed_size), 'ep' + str(epochs), 'lr' + str(lr), 'XX', 'XX', 'XX')
            if bprmf_path in dir_stored_model[0]:
                dir = dir_stored_model[0] + '/'
                break

    for r, d, f in os.walk(dir):
        for file in f:
            if 'weights-{0}-'.format(restore_epochs) in file:
                return dir + file.split('.')[0]
    return ''
