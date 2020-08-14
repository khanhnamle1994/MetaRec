# Import packages
import numpy as np
import pandas as pd
import logging
import gensim

# Import utility scripts
from data_prep import playlist_formatted, playlist_length
from evaluate import test_HR_and_NDGC_one_task_per_playlist

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set random seed
np.random.seed(0)

# Train with 80 % of the data (with cross validation) and test the final model with 20% of the data
index = np.random.choice(range(playlist_length), playlist_length, replace=False)
train = np.array(playlist_formatted)[index[:int(playlist_length * 0.8)]].tolist()
test = np.array(playlist_formatted)[index[int(playlist_length * 0.8):]].tolist()
print("Length of train set:", len(train))
print("Length of test set:", len(test))


def grid_search(*iterables):
    """
    Grid search creates all combinations of input values
    """
    return np.array(np.meshgrid(*iterables)).T.reshape(-1, len(iterables))


workers = 12  # number of workers
k = 10  # predicted items
sg = 0  # 0 for cbow, 1 for skipgram

window_options = [5]  # window size L
it_options = [30]  # epochs n
sample_options = [0.01]  # sub-sampling t
negative_sampling_dist = [0.75]  # negative sampling distribution parameter a

size_embedding_options = [50]  # embedding size
neg_options = [5]  # negative samples amount
alpha = [0.025]  # learning rate
min_count_options = [5]  # words under this are ignored

# Arrays to keep track of metrics
hyperparameters = []
hit_ratios = []
NDCG = []
best_model = None

# Grid search with all the hyper-parameters
for (n_window, n_it, n_sample, n_neg_sample_dist, n_size, n_neg, n_alpha, n_min_count) in \
        grid_search(window_options, it_options, sample_options, negative_sampling_dist,
                    size_embedding_options, neg_options, alpha, min_count_options):

    # Keep track of grid-search hyper-parameters
    hyperparameters.append([n_window, n_it, n_sample, n_neg_sample_dist, n_size, n_neg, n_alpha, n_min_count])
    print('Using these following hyper-parameters: ', hyperparameters[-1])

    # Instantiate word2Vec model from gensim library
    model = gensim.models.Word2Vec(train, size=int(n_size), window=int(n_window),
                                   min_count=int(n_min_count), workers=workers, sg=sg, iter=int(n_it),
                                   sample=n_sample, negative=int(n_neg), ns_exponent=n_neg_sample_dist, alpha=n_alpha)

    # Train the model
    model.train(train, total_examples=len(train), epochs=model.iter)

    # Save word2vec model
    model.save("models/word2vec.model")

    # Evaluate the model
    hits, ndgc, tries, fails = test_HR_and_NDGC_one_task_per_playlist(test, k, model)

    # Calculate Hit Ratio and NDCG metrics
    hit_ratio = hits / tries
    if hit_ratios and hit_ratio > max(hit_ratios):
        best_model = model
    hit_ratios.append(hits / tries)
    NDCG.append(ndgc / tries)

    # Display metrics to console
    print('Accuracy (hit_ratio) values for parameters:', hits / tries)
    print('NDCG values for parameters:', ndgc / tries)

# Print the combinations and their accuracy
acc_and_hyper = list(map(list, zip(hit_ratios, NDCG, hyperparameters)))
acc_and_hyper = list(map(lambda x: [x[0], x[1]] + x[2], acc_and_hyper))
acc_and_hyper.sort(key=lambda x: x[0])
acc_and_hyper = pd.DataFrame(acc_and_hyper)
acc_and_hyper.columns = ['hit-ratio', 'ndcg', 'window-size', 'epochs', 'sub-sample', 'negative-sampling-dist',
                         'embedding-size', 'negative-samples-size', 'learning-rate', 'mininum-count']

# Spotify Playlists CBOW
print(acc_and_hyper.head())
print(acc_and_hyper.tail())

# Save playlist CBOW into CSV format
acc_and_hyper.to_csv('playlists/hyperparameters_spotify_playlist_cbow.csv', index=False)
