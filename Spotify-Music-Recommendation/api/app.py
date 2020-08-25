# Import packages
import os

from flask import Blueprint, request, jsonify, Flask
import wget

import db
from gensim.models import Word2Vec
from ml.utils import get_similar_song

app = Flask(__name__)
api = Blueprint('api', __name__)

# Load word2vec model for inference
model_name = 'word2vec.model'
model_path = f'./ml/{model_name}'

# Download the trained word2vec model from GitHub and save it at src/api/ml/. This is done at the first run of the API
if model_name not in os.listdir('./ml'):
    print(f'downloading the trained model {model_name}')
    wget.download(
        "https://github.com/khanhnamle1994/transfer-rec/blob/master/Spotify-Music-Recommendation/experiment/models/word2vec.model",
        out=model_path
    )
else:
    print('model already saved to api/ml')

model = Word2Vec.load(model_path)
print('word2vec model loaded!')


@api.route('/recommend', methods=['POST'])
def recommend_songs():
    """
    Endpoint to recommend songs using the song name
    """
    if request.method == 'POST':
        if 'song_name' not in request.form:
            return jsonify({'error': 'no song name in body'}), 400
        else:
            song_name = request.form['song_name']
            output = get_similar_song(model, song_name)
            return jsonify(float(output))


@api.route('/song', methods=['POST'])
def post_song():
    """
    Save song to database
    """
    if request.method == 'POST':
        expected_fields = [
            'song_name',
            'artist_name',
            'playlist_name',
            'user_agent',
            'ip_address'
        ]
        if any(field not in request.form for field in expected_fields):
            return jsonify({'error': 'Missing field in body'}), 400

        query = db.Song.create(**request.form)
        return jsonify(query.serialize())


@api.route('/songs', methods=['GET'])
def get_songs():
    """
    Get all songs
    """
    if request.method == 'GET':
        query = db.Song.select()
        return jsonify([r.serialize() for r in query])
