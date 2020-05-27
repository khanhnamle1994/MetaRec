# Import packages
import math
import numpy as np
from collections import OrderedDict

# Import utility script
from Tools import RunningAverage as AVG


class Evaluator:
    """
    Class that defines the evaluator
    """
    def __init__(self, eval_pos, eval_target, item_popularity, top_k):
        """
        :param eval_pos: position of the evaluated item
        :param eval_target: the evaluation target
        :param item_popularity: the popularity of the item
        :param top_k: choice of top K
        """
        self.top_k = top_k if isinstance(top_k, list) else [top_k]
        self.max_k = max(self.top_k)
        self.eval_pos = eval_pos
        self.eval_target = eval_target
        self.item_popularity = item_popularity
        self.num_users, self.num_items = self.eval_pos.shape
        self.item_self_information = self.compute_item_self_info(item_popularity)

    def evaluate(self, model, dataset, test_batch_size):
        """
        Function to perform evaluation
        :param model: choice of model
        :param dataset: given dataset
        :param test_batch_size: choice of batch size in test set
        :return: dictionary that stores the evaluation metrics
        """
        # Step into evaluation mode
        model.eval()
        # Collect the evaluation users
        eval_users = np.array(list(self.eval_target.keys()))
        # Get the prediction matrix
        pred_matrix = model.predict(eval_users, self.eval_pos, test_batch_size)
        # Get the top-k predictions
        topk = self.predict_topk(pred_matrix, max(self.top_k))

        # Precision, Recall, NDCG @ k
        scores = self.prec_recall_ndcg(topk, self.eval_target)
        score_dict = OrderedDict()
        for metric in scores:
            score_by_ks = scores[metric]
            for k in score_by_ks:
                score_dict['%s@%d' % (metric, k)] = score_by_ks[k].mean

        # Novelty @ k
        novelty_dict = self.novelty(topk)
        for k, v in novelty_dict.items():
            score_dict[k] = v

        # Gini diversity
        score_dict['Gini-D'] = self.gini_diversity(topk)

        return score_dict

    def predict_topk(self, scores, k):
        """
        Function to get the top-k predictions
        :param scores: prediction matrix
        :param k: choice of k
        :return: top-k predictions
        """
        # top_k item index (not sorted)
        relevant_items_partition = (-scores).argpartition(k, 1)[:, 0:k]

        # top_k item score (not sorted)
        relevant_items_partition_original_value = np.take_along_axis(scores, relevant_items_partition, 1)

        # top_k item sorted index for partition
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, 1)

        # sort top_k index
        topk = np.take_along_axis(relevant_items_partition, relevant_items_partition_sorting, 1)

        return topk

    def prec_recall_ndcg(self, topk, target):
        """
        Function to get the precision, recall, and NDCG @ k
        :param topk: top-k predictions
        :param target: ground truth label
        :return: precision, recall, and NDCG @ k
        """
        # Initialize the precision, recall, and NDCG values as averages of top-k predictions
        prec = {k: AVG() for k in self.top_k}
        recall = {k: AVG() for k in self.top_k}
        ndcg = {k: AVG() for k in self.top_k}

        # Dictionary to store precision, recall, and NDCG scores
        scores = {'Prec': prec, 'Recall': recall, 'NDCG': ndcg}

        for idx, u in enumerate(target):
            pred_u = topk[idx]
            target_u = target[u]
            num_target_items = len(target_u)
            for k in self.top_k:
                pred_k = pred_u[:k]
                hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
                num_hits = len(hits_k)

                idcg_k = 0.0
                for i in range(1, min(num_target_items, k) + 1):
                    idcg_k += 1 / math.log(i + 1, 2)

                dcg_k = 0.0
                for idx, item in hits_k:
                    dcg_k += 1 / math.log(idx + 1, 2)

                # Calculate precision, recall, and NDCG @ k
                prec_k = num_hits / k
                recall_k = num_hits / min(num_target_items, k)
                ndcg_k = dcg_k / idcg_k

                # Update the metrics accordingly
                scores['Prec'][k].update(prec_k)
                scores['Recall'][k].update(recall_k)
                scores['NDCG'][k].update(ndcg_k)

        return scores

    def novelty(self, topk):
        """
        Function to get the novelty @ k
        :param topk: top-k predictions
        :return: novelty @ k
        """
        topk_info = np.take(self.item_self_information, topk)
        top_k_array = np.array(self.top_k)
        topk_info_sum = np.cumsum(topk_info, 1)[:, top_k_array - 1]
        novelty_all_users = topk_info_sum / np.atleast_2d(top_k_array)
        novelty = np.mean(novelty_all_users, axis=0)

        novelty_dict = {'Nov@%d' % self.top_k[i]: novelty[i] for i in range(len(self.top_k))}

        return novelty_dict

    def gini_diversity(self, topk):
        """
        Function to calculate Gini diversity index
        :param topk: top-k predictions
        :return: Gini diversity index
        """
        num_items = self.eval_pos.shape[1]
        item_recommend_counter = np.zeros(num_items, dtype=np.int)

        rec_item, rec_count = np.unique(topk, return_counts=True)
        item_recommend_counter[rec_item] += rec_count

        item_recommend_counter_mask = np.ones_like(item_recommend_counter, dtype=np.bool)
        item_recommend_counter_mask[item_recommend_counter == 0] = False
        item_recommend_counter = item_recommend_counter[item_recommend_counter_mask]
        num_eff_items = len(item_recommend_counter)

        item_recommend_counter_sorted = np.sort(item_recommend_counter)  # values must be sorted
        index = np.arange(1, num_eff_items + 1)  # index per array element

        gini_diversity = 2 * np.sum(
            (num_eff_items + 1 - index) / (num_eff_items + 1) * item_recommend_counter_sorted / np.sum(
                item_recommend_counter_sorted))
        return gini_diversity

    def compute_item_self_info(self, item_popularity):
        self_info = np.zeros(len(item_popularity))
        # total = 0
        for i in item_popularity:
            self_info[i] = item_popularity[i] / self.num_users
            # total += item_popularity[i]
        # self_info /= total
        self_info = -np.log2(self_info)
        return self_info
