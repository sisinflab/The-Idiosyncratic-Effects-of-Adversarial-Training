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


class AMF(RecommenderModel):

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, path_output_rec_list, args):
        """
        Create a AMF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices)
        :param data: data loader object
        :param path_output_rec_result: path to the directory rec. results
        :param path_output_rec_weight: path to the directory rec. model parameters
        :param args: parameters
        """
        super(AMF, self).__init__(data, path_output_rec_result, path_output_rec_weight, path_output_rec_list, args.rec)
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

        self.adv_eps = args.adv_eps
        self.adv_type = args.adv_type
        self.adv_reg = args.adv_reg
        self.adv_iteration = args.adv_iteration
        self.adv_step_size = args.adv_step_size
        self.best = args.best

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
            # print(gamma_u.shape, gamma_i.shape, (gamma_u * gamma_i).shape)
            xui = beta_i + tf.reduce_sum(gamma_u * gamma_i, )
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

        # Restore Deltas for the Perturbation
        self.set_delta(delta_init=0)

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

            loss += reg_loss

            # Restore Deltas for the Perturbation
            self.set_delta(delta_init=self.adv_type == 'pgd')

            # Adversarial Training Component
            if self.adv_type == 'fgsm':
                self.fgsm_perturbation(user_input, item_input_pos, item_input_neg)

            # Adversarial Inference
            self.output_pos_adver, _, _, _ = self.get_inference(inputs=(user_input, item_input_pos))
            self.output_neg_adver, _, _, _ = self.get_inference(inputs=(user_input, item_input_neg))

            result_adver = tf.clip_by_value(self.output_pos_adver - self.output_neg_adver, -80.0, 1e8)
            loss_adver = tf.reduce_sum(tf.nn.softplus(-result_adver))

            # Loss to be optimized
            loss += self.adv_reg * loss_adver

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

            epoch_text = 'Epoch {0}/{1} \tLoss: {2:.3f} (Avg Batch Losses) in {3}'.format(epoch, self.epochs,
                                                                                          loss / steps,
                                                                                          timer(start_ep, time()))
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

    def train_to_build_plots(self):

        adv_eps, adv_reg = self.adv_eps, self.adv_reg
        self.adv_eps, self.adv_reg = 0, 0  # We need to disable the AMF for the first half of epochs

        print('Start training...')

        # Dictionary where for each item associates a list. In this list we store the gradient magnitude in each epoch
        positive_gradient_magnitudes, negative_gradient_magnitudes = {}, {}
        adv_positive_gradient_magnitudes, adv_negative_gradient_magnitudes = {}, {}
        start_epoch = time()

        adv_text = ''

        for epoch in range(1, self.epochs + 1):

            if epoch > self.epochs // 2:
                self.adv_eps = adv_eps
                self.adv_reg = adv_reg
                adv_text = '(Adv. alpha: {}\teps: {})'.format(self.adv_reg, self.adv_eps)

            print('\tEpoch: {}/{}\t{}'.format(epoch, self.epochs, adv_text))

            positive_gradient_magnitudes[epoch], negative_gradient_magnitudes[epoch] = {}, {}
            adv_positive_gradient_magnitudes[epoch], adv_negative_gradient_magnitudes[epoch] = {}, {}

            for item_id in range(self.num_items):
                positive_gradient_magnitudes[epoch][item_id], negative_gradient_magnitudes[epoch][item_id] = [], []
                adv_positive_gradient_magnitudes[epoch][item_id], adv_negative_gradient_magnitudes[epoch][
                    item_id] = [], []

            steps = 0

            next_batch = self.data.next_triple_batch()

            for batch in next_batch:
                steps += 1

                if steps % 10000 == 0:
                    print('\t\t{} in {}'.format(steps, timer(start_epoch, time())))
                    start_epoch = time()

                user_input, item_input_pos, item_input_neg = batch

                # Restore Deltas for the Perturbation
                self.set_delta(delta_init=0)

                with tf.GradientTape() as t:
                    t.watch([self.item_bias, self.embedding_P, self.embedding_Q])

                    # Clean Inference
                    output_pos, beta_pos, embed_p_pos, embed_q_pos = self.get_inference(
                        inputs=(user_input, item_input_pos))
                    output_neg, beta_neg, _, embed_q_neg = self.get_inference(inputs=(user_input, item_input_neg))
                    result = tf.clip_by_value(output_pos - output_neg, -80.0, 1e8)
                    loss = tf.reduce_sum(tf.nn.softplus(-result))

                    # Gradient Magnitude
                    gradient_magnitude = (1 - tf.math.sigmoid(result))
                    positive_gradient_magnitudes[epoch][item_input_pos.numpy()[0]].append(gradient_magnitude.numpy())
                    negative_gradient_magnitudes[epoch][item_input_neg.numpy()[0]].append(-gradient_magnitude.numpy())

                    # Regularization Component
                    reg_loss = self.reg * tf.reduce_sum([tf.nn.l2_loss(embed_p_pos),
                                                         tf.nn.l2_loss(embed_q_pos),
                                                         tf.nn.l2_loss(embed_q_neg)]) \
                               + self.bias_reg * tf.nn.l2_loss(beta_pos) \
                               + self.bias_reg * tf.nn.l2_loss(beta_neg) / 10

                    loss += reg_loss

                    if self.adv_reg != 0:

                        # Restore Deltas for the Perturbation
                        self.set_delta(delta_init=self.adv_type == 'pgd')

                        # Adversarial Training Component
                        if self.adv_type == 'fgsm':
                            self.fgsm_perturbation(user_input, item_input_pos, item_input_neg)

                        # Adversarial Inference
                        output_pos_adver, _, _, _ = self.get_inference(inputs=(user_input, item_input_pos))
                        output_neg_adver, _, _, _ = self.get_inference(inputs=(user_input, item_input_neg))

                        result_adver = tf.clip_by_value(output_pos_adver - output_neg_adver, -80.0, 1e8)
                        loss_adver = tf.reduce_sum(tf.nn.softplus(-result_adver))

                        # Adversarial Gradient Magnitude
                        adv_gradient_magnitude = (1 - tf.math.sigmoid(result_adver))
                        adv_positive_gradient_magnitudes[epoch][item_input_pos.numpy()[0]].append(adv_gradient_magnitude.numpy())
                        adv_negative_gradient_magnitudes[epoch][item_input_neg.numpy()[0]].append(-adv_gradient_magnitude.numpy())

                        # Loss to be optimized
                        loss += self.adv_reg * loss_adver

                gradients = t.gradient(loss, [self.item_bias, self.embedding_P, self.embedding_Q])
                self.optimizer.apply_gradients(zip(gradients, [self.item_bias, self.embedding_P, self.embedding_Q]))

            print('\t\t{} in {}'.format(steps, timer(start_epoch, time())))

        print('Training Ended')

        return positive_gradient_magnitudes, negative_gradient_magnitudes, adv_positive_gradient_magnitudes, adv_negative_gradient_magnitudes

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

        if self.restore_epochs >= 1:
            try:
                checkpoint_file = find_checkpoint(self.path_output_rec_weight, self.restore_epochs, self.epochs,
                                                  self.rec, 0, (self.embedding_size, self.epochs, self.learning_rate))
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
