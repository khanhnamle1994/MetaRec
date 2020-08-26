# Import packages
import datetime
import pandas as pd


# Class object to store MovieLens1M
class movielens_1m(object):
    def __init__(self):
        self.user_data, self.item_data, self.score_data = self.load()

    def load(self):
        """
        Load MovieLens1M
        :return: user, item, and ratings information
        """
        # Path to data
        path = "../ml-1m"
        user_data_path = "{}/users.dat".format(path)
        rating_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/movies.dat".format(path)

        # Read into Pandas DataFrame
        user_data = pd.read_csv(user_data_path,
                                names=['user_id', 'gender', 'age', 'occupation_code', 'zipcode'],
                                sep="::", engine='python'
                                )
        item_data = pd.read_csv(item_data_path, names=['movie_id', 'title', 'genre'],
                                sep="::", engine='python', encoding="utf-8"
                                )
        rating_data = pd.read_csv(rating_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                  sep="::", engine='python'
                                  )

        # Convert 'time' column into 'datetime' data type
        rating_data['time'] = rating_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        rating_data = rating_data.drop(["timestamp"], axis=1)
        return user_data, item_data, rating_data
