import tensorflow as tf


class RecommenderModel(tf.keras.Model):
    """
    This class represents a recommender model.
    You can load a pretrained model by specifying its ckpt path
     and use it for training/testing purposes.

    Attributes:
        model:
        do_eval: True to use the model in inference-mode, otherwise False
        gpu (int): index of gpu to use (-1 for cpu)
        model_path (str): model path
    """

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, path_output_rec_list, rec):
        self.rec = rec
        self.data = data
        self.num_items = data.num_items
        self.num_users = data.num_users
        self.path_output_rec_result = path_output_rec_result
        self.path_output_rec_weight = path_output_rec_weight
        self.path_output_rec_list = path_output_rec_list

    def train(self):
        pass

    def restore(self):
        pass

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