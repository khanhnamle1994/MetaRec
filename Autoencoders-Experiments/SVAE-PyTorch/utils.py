import torch
import os
import json
import pickle
import matplotlib.pyplot as plt

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor


def save_obj(obj, name):
    """Function to save Pickle object"""
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_obj_json(obj, name):
    """Function to save JSON object"""
    with open(name + '.json', 'w') as f:
        json.dump(obj, f)


def load_obj(name):
    """Function to load Pickle object"""
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_obj_json(name):
    """Function to load JSON object"""
    with open(name + '.json', 'r') as f:
        return json.load(f)


def file_write(log_file, s):
    """Function to write into log file"""
    print(s)
    f = open(log_file, 'a')
    f.write(s + '\n')
    f.close()


def clear_log_file(log_file):
    """Function to clear log file"""
    f = open(log_file, 'w')
    f.write('')
    f.close()


def pretty_print(h):
    """Function to pretty print"""
    print("{")
    for key in h:
        print(' ' * 4 + str(key) + ': ' + h[key])
    print('}\n')


def plot_len_vs_ndcg(len_to_ndcg_at_100_map):
    """Function to plot the number of items in fold-out set vs NDCG@100"""
    lens = list(len_to_ndcg_at_100_map.keys())
    lens.sort()
    X, Y = [], []

    for le in lens:
        X.append(le)
        ans = 0.0
        for i in len_to_ndcg_at_100_map[le]:
            ans += float(i)
        ans = ans / float(len(len_to_ndcg_at_100_map[le]))
        Y.append(ans * 100.0)

    # Smoothening
    Y_mine = []
    prev_5 = []
    for i in Y:
        prev_5.append(i)
        if len(prev_5) > 5:
            del prev_5[0]

        temp = 0.0
        for j in prev_5:
            temp += float(j)
        temp = float(temp) / float(len(prev_5))
        Y_mine.append(temp)

    # Use Matplotlib to plot the result
    plt.figure(figsize=(12, 5))
    plt.plot(X, Y_mine, label='Sequential Variational Auto-Encoder')
    plt.xlabel("Number of items in the fold-out set")
    plt.ylabel("Average NDCG@100")
    plt.title("SVAE_ML1M")
    if not os.path.isdir("saved_plots/"):
        os.mkdir("saved_plots/")
    plt.savefig("saved_plots/seq_len_vs_ndcg_" + "SVAE_ML1M" + ".pdf")
    plt.legend(loc='best', ncol=2)

    plt.show()
