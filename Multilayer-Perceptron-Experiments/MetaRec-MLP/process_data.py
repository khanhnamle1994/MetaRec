# Import libraries
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import re
import datetime
import pandas as pd
import json
from tqdm import tqdm


class movielens_1m(object):
    """
    Class to initialize MovieLens1M dataset
    """

    def __init__(self):
        """
        Initialize user, item, and ratings data
        """
        self.user_data, self.item_data, self.score_data = self.load()

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
            user_data_path,
            names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
            sep="::", engine='python'
        )

        # Read item data CSV file
        item_data = pd.read_csv(
            item_data_path,
            names=['movie_id', 'title', 'year', 'rate', 'released', 'genre',
                   'director', 'writer', 'actors', 'plot', 'poster'],
            sep="::", engine='python', encoding="utf-8"
        )

        # Read rating data CSV file
        rating_data = pd.read_csv(
            rating_data_path,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )

        # Change 'timestamp' into 'time' with datetime format
        rating_data['time'] = rating_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        rating_data = rating_data.drop(["timestamp"], axis=1)
        return user_data, item_data, rating_data


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    """
    Convert item data into suitable format
    :param row: current row
    :param rate_list: list of rate levels
    :param genre_list: list of movie genres
    :param director_list: list of directors
    :param actor_list: list of actors
    :return: PyTorch tensors storing item data
    """
    # Convert rate_list to PyTorch Tensor
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()

    # Convert genre_list to PyTorch Tensor
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1

    # Convert director_list to PyTorch Tensor
    director_idx = torch.zeros(1, 2186).long()
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1

    # Convert actor_list to PyTorch Tensor
    actor_idx = torch.zeros(1, 8030).long()
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1

    # Concatenate PyTorch tensors into one-dimensional tensor
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    """
    Convert user data into suitable format
    :param row: current row
    :param gender_list: list of genders
    :param age_list: list of ages
    :param occupation_list: list of occupations
    :param zipcode_list: list of zipcodes
    :return: PyTorch tensors storing user data
    """
    # Convert gender_list to PyTorch Tensor
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()

    # Convert age_list to PyTorch Tensor
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()

    # Convert occupation_list to PyTorch Tensor
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()

    # Convert zipcode_list to PyTorch Tensor
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()

    # Concatenate PyTorch tensors into one-dimensional tensor
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


def load_list(fname):
    """
    Return a list from a file
    :param fname: file name
    :return: Python list
    """
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


class DataPrep(Dataset):
    def __init__(self, args, partition='train', test_way=None, path=None):
        super(DataPrep, self).__init__()
        self.partition = partition

        # Path to MovieLens 1M
        self.dataset_path = args.data_root
        dataset_path = self.dataset_path

        # Load rate, genre, actor, director, gender, age, occupation, and zipcode lists
        rate_list = load_list("{}/m_rate.txt".format(dataset_path))
        genre_list = load_list("{}/m_genre.txt".format(dataset_path))
        actor_list = load_list("{}/m_actor.txt".format(dataset_path))
        director_list = load_list("{}/m_director.txt".format(dataset_path))
        gender_list = load_list("{}/m_gender.txt".format(dataset_path))
        age_list = load_list("{}/m_age.txt".format(dataset_path))
        occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
        zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

        # Initialize the MovieLens1M dataset class
        self.dataset = movielens_1m()

        # Use a hashmap called 'movie_dict' to store item information
        master_path = self.dataset_path
        if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
            self.movie_dict = {}
            # Iterate over indices and rows of item data
            for idx, row in self.dataset.item_data.iterrows():
                # Save rate levels, genres, directors, and actors into item_dict
                m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
                self.movie_dict[row['movie_id']] = m_info
            pickle.dump(self.movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
        else:
            self.movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))

        # Use a hashmap called 'user_dict' to store user information
        if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
            self.user_dict = {}
            # Iterate over indices and rows of user data
            for idx, row in self.dataset.user_data.iterrows():
                # Save genders, ages, occupations, and zipcodes into user_dict
                u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
                self.user_dict[row['user_id']] = u_info
            pickle.dump(self.user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
        else:
            self.user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

        if partition == 'train' or partition == 'valid':
            self.state = 'warm_state'
        else:
            if test_way is not None:
                if test_way == 'old':
                    self.state = 'warm_state'
                elif test_way == 'new_user':
                    self.state = 'user_cold_state'
                elif test_way == 'new_item':
                    self.state = 'item_cold_state'
                else:
                    self.state = 'user_and_item_cold_state'
        print(self.state)

        # Open the corresponding existing and new users (in JSON files) for the current experiment scenario
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())

        length = len(self.dataset_split.keys())
        self.final_index = []

        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])

            # Only include users whose item-consumption history length is between 13 and 100
            if seen_movie_len < 13 or seen_movie_len > 100:
                continue
            else:
                self.final_index.append(user_id)

    def __getitem__(self, item):
        user_id = self.final_index[item]
        u_id = int(user_id)
        seen_movie_len = len(self.dataset_split[str(u_id)])
        indices = list(range(seen_movie_len))
        # Shuffle the indices randomly
        random.shuffle(indices)
        # Existing users
        tmp_x = np.array(self.dataset_split[str(u_id)])
        # New users
        tmp_y = np.array(self.dataset_split_y[str(u_id)])

        # Use the remaining items as the support set to calculate test loss
        support_x_app = None
        for m_id in tmp_x[indices[:-10]]:
            m_id = int(m_id)
            tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted

        # Use 10 random items from the history as the query set to calculate training loss
        query_x_app = None
        for m_id in tmp_x[indices[-10:]]:
            m_id = int(m_id)
            u_id = int(user_id)
            tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted

        support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
        query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])
        return support_x_app, support_y_app.view(-1, 1), query_x_app, query_y_app.view(-1, 1)

    def __len__(self):
        return len(self.final_index)
