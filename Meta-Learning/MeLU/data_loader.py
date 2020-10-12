# Import packages
import datetime
import pandas as pd


class movielens_1m(object):
    """
    Class to initialize MovieLens1M dataset
    """
    def __init__(self):
        """
        Initialize user, item, and ratings data
        """
        self.user_data, self.item_data, self.rating_data = self.load()

    def load(self):
        """
        Load MovieLens 1M
        :return: user, item, and ratings datasets
        """
        # Path to store the processed data
        path = "../../ml-1m"

        # Path to user, item, and ratings data
        user_data_path = "{}/users.dat".format(path)
        rating_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/movies_extrainfos.dat".format(path)  # additional movie contents from IMDB

        # Read user data CSV file
        user_data = pd.read_csv(
            user_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
            sep="::", engine='python'
        )

        # Read item data CSV file
        item_data = pd.read_csv(
            item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released',
                                   'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
            sep="::", engine='python', encoding="utf-8"
        )

        # Read rating data CSV file
        rating_data = pd.read_csv(
            rating_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )

        # Change 'timestamp' into 'time' with datetime format
        rating_data['time'] = rating_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        rating_data = rating_data.drop(["timestamp"], axis=1)
        return user_data, item_data, rating_data
