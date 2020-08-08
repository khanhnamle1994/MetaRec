# Import packages
import random
import math


def test_HR_and_NDGC_single_item(playlist, index, k, model):
    """
    Test HR and NDCG metrics for a single item
    :param playlist: playlist dataframe
    :param index: index of the song item
    :param k: number of ranked items
    :param model: word2vec model
    :return: Hits, NDCG, tries, and fails
    """
    hits = 0
    ndgc = 0
    tries = 0
    fails = 0
    query = playlist[index]
    word = playlist[index + 1]
    try:
        res = model.wv.most_similar(positive=query, topn=k)
        for i in range(k):
            if res[i][0] == word:
                hits += 1
                ndgc += 1 / (math.log2(i + 1))
        tries += 1
    except:
        fails += 1
    return hits, ndgc, tries, fails


def test_HR_and_NDGC_one_task_per_playlist(test_set, k, model):
    """
    Test HR and NDCG metrics for one task per playlist
    :param test_set: playlist test dataframe
    :param k: number of ranked items
    :param model: word2vec model
    :return: Hits, NDCG, tries, and fails
    """
    hits = 0
    ndgc = 0
    tries = 0
    fails = 0
    for playlist in test_set:
        ind = random.randrange(len(playlist) - 1)
        hi, nd, tr, fa = test_HR_and_NDGC_single_item(playlist, ind, k, model)
        hits += hi
        ndgc += nd
        tries += tr
        fails += fa
    return hits, ndgc, tries, fails
