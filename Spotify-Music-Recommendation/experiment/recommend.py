# Import packages
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load word2vec model
model = Word2Vec.load("models/word2vec.model")

# Display top 20 most similar tracks to 'Eminem - Without Me'
w1 = "Eminem - Without Me"
print(model.wv.most_similar(positive=w1.lower(), topn=20))


def compare_similarity(song1, song2):
    """
    Compare the similarity score between two tracks
    :param song1: track 1
    :param song2: track 2
    :return: similarity score
    """
    similarity_score = model.wv.similarity(song1.lower(), song2.lower())
    return similarity_score


# Store word vector vocabularies in a list
vocab = list(model.wv.vocab)
X = model.wv[vocab]
print("Shape of word vector:", X.shape)

# Visualize word embeddings with T-SNE
index = np.random.choice(range(len(X)), 100, replace=False)
subset_x = np.array(X)[index]
labels = np.array(vocab)[index]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(subset_x)


def label_format(label, condition=0):
    """
    Format the track label
    """
    if condition == 1:
        return label.split("-", 1)[0]
    elif condition == 2:
        return label.split("-", 1)[1]
    else:
        return label


# Use Matplotlib for visualization
plt.figure(figsize=(20, 20))

for i in range(len(X_tsne)):
    plt.scatter(X_tsne[i, 0], X_tsne[i, 1])
    plt.annotate(label_format(labels[i], 0), xy=(X_tsne[i, 0], X_tsne[i, 1]), xytext=(8, 5),
                 textcoords='offset points', ha='right', va='bottom')

plt.savefig('pics/playlist2vec.png')
