import torch
import numpy as np


def evaluate(model, criterion, reader, hyper_params, is_train_set):
    """
    Function to evaluate the model
    :param model: The model choice
    :param criterion: The loss function choice
    :param reader: The Data Reader class
    :param hyper_params: The hyper-parameter dictionary
    :param is_train_set: Boolean value to check training setting
    :return: NDCG, Precision, and Recall metrics
    """
    # Step into evaluation mode
    model.eval()

    metrics = {'loss': 0.0}
    Ks = [10, 100]
    for k in Ks:
        metrics['NDCG@' + str(k)] = 0.0
        metrics['Rec@' + str(k)] = 0.0
        metrics['Prec@' + str(k)] = 0.0

    batch = 0
    total_users = 0.0

    # For plotting the results (seq length vs. NDCG@100)
    len_to_ndcg_at_100_map = {}

    for x, y_s, test_movies, test_movies_r in reader.iter_eval():
        batch += 1
        if is_train_set and batch > hyper_params['train_cp_users']:
            break

        decoder_output, z_mean, z_log_sigma = model(x)

        metrics['loss'] += criterion(decoder_output, z_mean, z_log_sigma, y_s, 0.2).data

        # Making the logits of previous items in the sequence to be "- infinity"
        decoder_output = decoder_output.data
        x_scattered = torch.zeros(decoder_output.shape[0], decoder_output.shape[2])
        x_scattered[0, :].scatter_(0, x[0].data, 1.0)
        last_predictions = decoder_output[:, -1, :] - (torch.abs(decoder_output[:, -1, :] * x_scattered) * 100000000)

        for batch_num in range(last_predictions.shape[0]):
            # batch_num is ideally only 0, since batch_size is enforced to be always 1
            predicted_scores = last_predictions[batch_num]
            actual_movies_watched = test_movies[batch_num]
            actual_movies_ratings = test_movies_r[batch_num]

            # Calculate NDCG
            _, argsorted = torch.sort(-1.0 * predicted_scores)
            for k in Ks:
                best, now_at, dcg, hits = 0.0, 0.0, 0.0, 0.0

                rec_list = list(argsorted[:k].cpu().numpy())
                for m in range(len(actual_movies_watched)):
                    movie = actual_movies_watched[m]
                    now_at += 1.0
                    if now_at <= k:
                        best += 1.0 / float(np.log2(now_at + 1))

                    if movie not in rec_list:
                        continue
                    hits += 1.0
                    dcg += 1.0 / float(np.log2(float(rec_list.index(movie) + 2)))

                metrics['NDCG@' + str(k)] += float(dcg) / float(best)
                metrics['Rec@' + str(k)] += float(hits) / float(len(actual_movies_watched))
                metrics['Prec@' + str(k)] += float(hits) / float(k)

                # Only for plotting the graph (seq length vs. NDCG@100)
                if k == 100:
                    seq_len = int(len(actual_movies_watched)) + int(x[batch_num].shape[0]) + 1
                    if seq_len not in len_to_ndcg_at_100_map:
                        len_to_ndcg_at_100_map[seq_len] = []
                    len_to_ndcg_at_100_map[seq_len].append(float(dcg) / float(best))

            total_users += 1.0

    metrics['loss'] = float(metrics['loss']) / float(batch)
    metrics['loss'] = round(metrics['loss'], 4)

    for k in Ks:
        metrics['NDCG@' + str(k)] = round((100.0 * metrics['NDCG@' + str(k)]) / float(total_users), 4)
        metrics['Rec@' + str(k)] = round((100.0 * metrics['Rec@' + str(k)]) / float(total_users), 4)
        metrics['Prec@' + str(k)] = round((100.0 * metrics['Prec@' + str(k)]) / float(total_users), 4)

    return metrics, len_to_ndcg_at_100_map
