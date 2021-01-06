import scipy.sparse as sp
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
import tensorflow as tf

from src.util.timethis import timethis
np.random.seed(0)

_user_input = None
_item_input_pos = None
_batch_size = None
_index = None
_model = None
_train = None
_test = None
_num_items = None


def _get_train_batch(i):
    """
    Generation of a batch in multiprocessing
    :param i: index to control the batch generayion
    :return:
    """
    user_batch, item_pos_batch, item_neg_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_pos_batch.append(_item_input_pos[_index[idx]])
        j = np.random.randint(_num_items)
        while j in _train[_user_input[_index[idx]]]:
            j = np.random.randint(_num_items)
        item_neg_batch.append(j)
    return np.array(user_batch)[:, None], np.array(item_pos_batch)[:, None], np.array(item_neg_batch)[:, None]


class DataLoader(object):
    """
    Load train and test dataset
    """

    def __init__(self, path_train_data, path_test_data, args):
        """
        Constructor of DataLoader
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """
        self.params = args

        self.num_users, self.num_items = self.get_length(path_train_data, path_test_data)
        # self.load_train_file(path_train_data)

        self.path_train_data = path_train_data
        self.path_validation_data = None
        self.path_test_data = path_test_data

        # train
        self.train_list = []
        self.load_list('train')

        # Valid
        self.validation_list = []

        # test
        self.test_list = []
        self.load_list('test')

        # self.load_train_file_as_list(path_train_data)
        # self.load_test_file(path_test_data)

        # self._user_input, self._item_input_pos = self.sampling()

        print('{0} and {1} - Loaded'.format(self.path_train_data, self.path_test_data))

    def get_length(self, train_name, test_name):
        train = pd.read_csv(train_name, sep='\t', header=None)
        test = pd.read_csv(test_name, sep='\t', header=None)
        try:
            train.columns = ['user', 'item', 'r', 't']
            test.columns = ['user', 'item', 'r', 't']
            data = train.copy()
            data = data.append(test, ignore_index=True)
        except:
            train.columns = ['user', 'item', 'r']
            test.columns = ['user', 'item', 'r']
            data = train.copy()
            data = data.append(test, ignore_index=True)

        # return data['user'].nunique(), data['item'].nunique()
        return max(data['user'].unique())+1, max(data['item'].unique())+1

    def load_train_file(self, filename):
        """
        Read /data/dataset_name/train file and Return the matrix.
        """
        # Get number of users and items
        # self.num_users, self.num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                # self.num_users = max(self.num_users, u)
                # self.num_items = max(self.num_items, i)
                line = f.readline()

        # Construct URM
        self.train = sp.dok_matrix((self.num_users + 1, self.num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    self.train[user, item] = 1.0
                line = f.readline()

        # self.num_users = self.train.shape[0]
        # self.num_items = self.train.shape[1]

    def load_list(self, train_val_test):
        # Get number of users and items
        u_ = 0
        items = []
        read_path = self.path_train_data if train_val_test == 'train' else self.path_validation_data if train_val_test == 'val' else self.path_test_data
        with open(read_path, "r") as f:
            line = f.readline()
            index = 0
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    if train_val_test == 'train':
                        self.train_list.append(items)
                    elif train_val_test == 'val':
                        self.validation_list.append(items)
                    else:
                        self.test_list.append(items)
                    items = []
                    u_ += 1
                index += 1
                items.append(i)
                line = f.readline()
        if train_val_test == 'train':
            self.train_list.append(items)
        elif train_val_test == 'val':
            self.validation_list.append(items)
        else:
            self.test_list.append(items)

    def load_train_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        self.train_list, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    self.train_list.append(items)
                    items = []
                    u_ += 1
                index += 1
                items.append(i)
                line = f.readline()
        self.train_list.append(items)

    def load_test_file(self, filename):
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                self.test_list.append([user, item])
                line = f.readline()

    def sampling(self):
        _user_input, _item_input_pos = [], []
        for (u, i) in self.train.keys():
            # positive instance
            _user_input.append(u)
            _item_input_pos.append(i)
        return _user_input, _item_input_pos

    def shuffle(self, batch_size=512):
        """
        Shuffle dataset to create batch with batch size
        Variable are global to be faster!
        :param batch_size: default 512
        :return: set of all generated random batches
        """
        global _user_input
        global _item_input_pos
        global _batch_size
        global _index
        global _model
        global _train
        global _num_items

        _user_input, _item_input_pos = self.user_input, self.item_input_pos
        _batch_size = batch_size
        _index = list(range(len(_user_input)))
        _train = self.train_list
        _num_items = self.num_items

        np.random.shuffle(_index)
        _num_batches = len(_user_input) // _batch_size
        pool = Pool(cpu_count())
        res = pool.map(_get_train_batch, range(_num_batches))
        pool.close()
        pool.join()

        user_input = [r[0] for r in res]
        item_input_pos = [r[1] for r in res]
        item_input_neg = [r[2] for r in res]
        return user_input, item_input_pos, item_input_neg

    def all_triple_batches(self):
        r_int = np.random.randint
        user_input, pos_input, neg_input = [], [], []

        for ab in range(self.num_users):
            u = r_int(self.num_users)
            uis = set(self.train_list[u])

            for i in uis:
                j = r_int(self.num_items)
                while j in uis:
                    j = r_int(self.num_items)

                user_input.append(np.array(u))
                pos_input.append(np.array(i))
                neg_input.append(np.array(j))

        return user_input, pos_input, neg_input,

    @timethis
    def next_triple_batch(self):
        all_triples = self.all_triple_batches()
        data = tf.data.Dataset.from_tensor_slices(all_triples)
        data = data.shuffle(buffer_size=self.params.batch_size*10).batch(batch_size=self.params.batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data
