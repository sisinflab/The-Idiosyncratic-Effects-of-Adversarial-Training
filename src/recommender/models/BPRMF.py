import numpy as np
import os
import logging

from time import time
from copy import deepcopy

np.random.seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from src.recommender.Evaluator import Evaluator
from src.recommender.RecommenderModel import RecommenderModel
from src.util.timethis import timethis
from src.util.write import save_obj
from src.util.read import find_checkpoint
from src.util.timer import timer


class BPRMF(RecommenderModel):

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, path_output_rec_list, args):
        """
        Create a BPR-MF instance.
        (see https://doi.org/10.1145/3209978.3209981 for details about the algorithm design choices)
        :param data: data loader object
        :param path_output_rec_result: path to the directory rec. results
        :param path_output_rec_weight: path to the directory rec. model parameters
        :param args: parameters
        """
        super(BPRMF, self).__init__(data, path_output_rec_result, path_output_rec_weight, path_output_rec_list, args.rec)
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        self.reg = args.reg
        self.bias_reg = args.bias_reg
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.restore_epochs = args.restore_epochs
        self.best_metric = args.best_metric
        self.evaluator = Evaluator(self, data, args.k)
        self.best = args.best

        self.adv_eps = args.adv_eps

        # Initialize Model Parameters
        self.embedding_P = tf.Variable(
            tf.random.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
            name='embedding_P', dtype=tf.dtypes.float32)  # (users, embedding_size)
        self.embedding_Q = tf.Variable(
            tf.random.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
            name='embedding_Q', dtype=tf.dtypes.float32)  # (items, embedding_size)
        self.item_bias = tf.Variable(tf.zeros(self.num_items), name='Bi', dtype=tf.float32)

        # Method to create delta locations for the adv. perturbations
        self.set_delta()

        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)

    def set_delta(self, delta_init=0):
        """
        Set delta variables useful to store delta perturbations,
        :param delta_init: 0: zero-like initialization, 1 uniform random noise initialization
        :return:
        """
        if delta_init:
            self.delta_P = tf.random.uniform(shape=[self.num_users, self.embedding_size], minval=-0.05, maxval=0.05,
                                             dtype=tf.dtypes.float32, seed=0)
            self.delta_Q = tf.random.uniform(shape=[self.num_items, self.embedding_size], minval=-0.05, maxval=0.05,
                                             dtype=tf.dtypes.float32, seed=0)
        else:
            self.delta_P = tf.Variable(tf.zeros(shape=[self.num_users, self.embedding_size]), dtype=tf.dtypes.float32,
                                       trainable=False)
            self.delta_Q = tf.Variable(tf.zeros(shape=[self.num_items, self.embedding_size]), dtype=tf.dtypes.float32,
                                       trainable=False)

    def get_biased_inference(self, user_input, item_input_pos):
        """
        generate predicition matrix with respect to passed users' and items indices
        :param user_input: user indices
        :param item_input_pos: item indices
        :return:
        """
        self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P + self.delta_P, user_input), 1)
        self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q + self.delta_Q, item_input_pos), 1)

        return tf.matmul(self.embedding_p * self.embedding_q,
                         self.h), self.embedding_p, self.embedding_q  # (b, embedding_size) * (embedding_size, 1)

    def get_inference(self, inputs, training=None, mask=None):
        """
        Generates prediction for passed users and items indices

        Args:
            inputs: user, item (batch)
            training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

        Returns:
            prediction and extracted model parameters
        """
        user, item = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.item_bias, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.embedding_P + self.delta_P, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.embedding_Q + self.delta_Q, item))
        try:
            xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, 1)
            return xui, beta_i, gamma_u, gamma_i
        except:
            print(gamma_u.shape, gamma_i.shape, (gamma_u*gamma_i).shape)
            xui = beta_i + tf.reduce_sum(gamma_u * gamma_i,)
            return xui, beta_i, gamma_u, gamma_i

    def predict_all(self):
        """
        Get Full Predictions useful for Full Store of Predictions
        :return: The matrix of predicted values.
        """
        return self.item_bias + tf.matmul(self.embedding_P + self.delta_P,
                                          tf.transpose(self.embedding_Q + self.delta_Q))

    def _train_step(self, batches):
        """
        Apply a single training step (across all batched in the dataset).
        :param batches: set of batches used fr the training
        :return:
        """
        user_input, item_input_pos, item_input_neg = batches

        # for batch_idx in range(len(user_input)):
        with tf.GradientTape() as t:
            t.watch([self.item_bias, self.embedding_P, self.embedding_Q])

            # Clean Inference
            self.output_pos, beta_pos, embed_p_pos, embed_q_pos = self.get_inference(
                inputs=(user_input, item_input_pos))
            self.output_neg, beta_neg, _, embed_q_neg = self.get_inference(inputs=(user_input, item_input_neg))
            result = tf.clip_by_value(self.output_pos - self.output_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.reg * tf.reduce_sum([tf.nn.l2_loss(embed_p_pos),
                                                 tf.nn.l2_loss(embed_q_pos),
                                                 tf.nn.l2_loss(embed_q_neg)]) \
                       + self.bias_reg * tf.nn.l2_loss(beta_pos) \
                       + self.bias_reg * tf.nn.l2_loss(beta_neg) / 10
            # Loss to be optimized
            loss += reg_loss

        gradients = t.gradient(loss, [self.item_bias, self.embedding_P, self.embedding_Q])
        self.optimizer.apply_gradients(zip(gradients, [self.item_bias, self.embedding_P, self.embedding_Q]))

        return loss.numpy()

    @timethis
    def train(self):

        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

        if self.restore():
            self.restore_epochs += 1
        else:
            self.restore_epochs = 1
            print("*** Training from scratch ***")

        max_metrics = {'hr': 0, 'p': 0, 'r': 0, 'auc': 0, 'ndcg': 0}
        best_model = self
        best_epoch = self.restore_epochs
        results = {}

        print('Start training...')

        for epoch in range(self.restore_epochs, self.epochs + 1):

            start_ep = time()
            loss = 0
            steps = 0

            next_batch = self.data.next_triple_batch()

            for batch in next_batch:
                steps += 1
                loss_batch = self._train_step(batch)
                loss += loss_batch

            epoch_text = 'Epoch {0}/{1} \tLoss: {2:.3f} (Avg Batch Losses) in {3}'.format(epoch, self.epochs, loss / steps, timer(start_ep, time()))
            print(epoch_text)

            if epoch % self.verbose == 0:
                # Eval Model
                epoch_eval_print = self.evaluator.eval(epoch, results, epoch_text, start_ep)

                # Updated Best Metrics
                for metric in max_metrics.keys():
                    # if max_metrics[metric] <= results[epoch][metric][self.evaluator.k - 1]:
                    #     max_metrics[metric] = results[epoch][metric][self.evaluator.k - 1]
                    if max_metrics[metric] <= results[epoch][metric][0]:
                        max_metrics[metric] = results[epoch][metric][0]
                        if metric == self.best_metric:
                            best_epoch, best_model, best_epoch_print = epoch, deepcopy(self), epoch_eval_print

                # Save Model
                saver_ckpt.save('{0}/weights-{1}'.format(self.path_output_rec_weight, epoch))
                # Save Rec Lists
                self.evaluator.store_recommendation(epoch=epoch)

        print('Training Ended')

        print("Store The Results of the Training")
        save_obj(results,
                 '{0}/{1}-results'.format(self.path_output_rec_result, self.path_output_rec_result.split('/')[-2]))

        print("Store Best Model at Epoch {0}".format(best_epoch))
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=best_model)
        saver_ckpt.save('{0}/best-weights-{1}'.format(self.path_output_rec_weight, best_epoch))
        best_model.evaluator.store_recommendation(epoch=best_epoch)

    def restore(self):
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        if self.best:
            try:
                checkpoint_file = find_checkpoint(self.path_output_rec_weight, 0, 0, self.rec, self.best)
                saver_ckpt.restore(checkpoint_file)
                print("Best Model correctly Restored: {0}".format(checkpoint_file))
                return True
            except Exception as ex:
                print("Error in model restoring operation! {0}".format(ex.message))
                return False

        if self.restore_epochs > 1:
            try:
                checkpoint_file = find_checkpoint(self.path_output_rec_weight, self.restore_epochs, self.epochs,
                                                  self.rec)
                saver_ckpt.restore(checkpoint_file)
                print("Model correctly Restored at Epoch: {0}".format(self.restore_epochs))
                return True
            except Exception as ex:
                print("Error in model restoring operation! {0}".format(ex.message))
        else:
            print("Restore Epochs Not Specified")
        return False

    def fgsm_perturbation(self, user_input, item_input_pos, item_input_neg):
        """
        Evaluate Adversarial Perturbation with FGSM-like Approach
        :param user_input:
        :param item_input_pos:
        :param item_input_neg:
        :param batch_idx:
        :return:
        """
        with tf.GradientTape() as tape_adv:
            tape_adv.watch([self.embedding_P, self.embedding_Q])
            # Clean Inference
            output_pos, beta_pos, embed_p_pos, embed_q_pos = self.get_inference(inputs=(user_input, item_input_pos))
            output_neg, beta_neg, embed_p_neg, embed_q_neg = self.get_inference(inputs=(user_input, item_input_neg))

            result = tf.clip_by_value(output_pos - output_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            loss += self.reg * tf.reduce_mean(
                tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

        grad_P, grad_Q = tape_adv.gradient(loss, [self.embedding_P, self.embedding_Q])
        grad_P, grad_Q = tf.stop_gradient(grad_P), tf.stop_gradient(grad_Q)
        self.delta_P = tf.nn.l2_normalize(grad_P, 1) * self.adv_eps
        self.delta_Q = tf.nn.l2_normalize(grad_Q, 1) * self.adv_eps

    def attack_full_fgsm(self, attack_eps, attack_name=""):
        """
        Create FGSM ATTACK
        :param attack_eps:
        :param attack_name:
        :return:
        """
        # Set eps perturbation (budget)
        self.adv_eps = attack_eps
        # user_input, item_input_pos, item_input_neg = self.data.shuffle(self.data.num_users)
        self.data.user_input, self.data.item_input_pos = self.data.sampling()
        user_input, item_input_pos, item_input_neg = self.data.shuffle(len(self.data.user_input))

        print('Initial Performance.')
        results = {}
        self.evaluator.eval(self.restore_epochs, results, 'BEST MODEL ' if self.best else str(self.restore_epochs))

        # Calculate Adversarial Perturbations
        self.fgsm_perturbation(user_input, item_input_pos, item_input_neg)

        print('After Attack Performance.')
        attacked_results = {}
        self.evaluator.eval(self.restore_epochs, attacked_results,
                            'BEST MODEL ' if self.best else str(self.restore_epochs))
        self.evaluator.store_recommendation(attack_name=attack_name)
        results_tsv = '{0}/{1}-results.tsv'.format(self.path_output_rec_result,
                                                   attack_name + self.path_output_rec_result.split('/')[
                                                       -2] + '_best{0}'.format(self.best))

        with open(results_tsv, 'w') as r:
            r.write("Metric\tBefore\tAfter\tDelta\n")
            for k, v in results[list(results.keys())[0]].items():
                r.write("{}\t{}\t{}\t{}\n".format(k,
                                                  v[0], attacked_results[list(attacked_results.keys())[0]][k][0],
                                                  round((attacked_results[list(attacked_results.keys())[0]][k][
                                                      0] - v[0]) * 100 / v[0], 2))
                        )

        print('{0} - Completed!'.format(attack_name))