def get_similar_song(model, song_name):
    """
    Return top 30 most similar tracks to given song_name
    :param model: word2vec model
    :param song_name: given track
    :return: top 30 most similar tracks
    """
    return model.wv.most_similar(positive=song_name.lower(), topn=30)
