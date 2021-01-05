# Import packages
import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm

# Import utility scripts
from config import states
from data_loader import movielens_1m


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


def generate(master_path):
    """
    Prepare the dataset
    :param master_path: path to data directory
    """
    # Path to MovieLens 1M
    dataset_path = "../../ml-1m"

    # Load rate, genre, actor, director, gender, age, occupation, and zipcode lists
    rate_list = load_list("{}/m_rate.txt".format(dataset_path))
    genre_list = load_list("{}/m_genre.txt".format(dataset_path))
    actor_list = load_list("{}/m_actor.txt".format(dataset_path))
    director_list = load_list("{}/m_director.txt".format(dataset_path))
    gender_list = load_list("{}/m_gender.txt".format(dataset_path))
    age_list = load_list("{}/m_age.txt".format(dataset_path))
    occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
    zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

    # Make new directories if they do not already exist
    if not os.path.exists("{}/warm_state/".format(master_path)):
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    # Initialize the MovieLens1M dataset class
    dataset = movielens_1m()

    # Use a hashmap called 'item_dict' to store item information
    if not os.path.exists("{}/m_item_dict.pkl".format(master_path)):
        item_dict = {}
        # Iterate over indices and rows of item data
        for idx, row in dataset.item_data.iterrows():
            # Save rate levels, genres, directors, and actors into item_dict
            m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
            item_dict[row['movie_id']] = m_info
        pickle.dump(item_dict, open("{}/m_item_dict.pkl".format(master_path), "wb"))
    else:
        item_dict = pickle.load(open("{}/m_item_dict.pkl".format(master_path), "rb"))

    # Use a hashmap called 'user_dict' to store user information
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        # Iterate over indices and rows of user data
        for idx, row in dataset.user_data.iterrows():
            # Save genders, ages, occupations, and zipcodes into user_dict
            u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    # Loop through different experiment scenarios (there are 4)
    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))

        # Open the corresponding existing and new users (in JSON files) for the current experiment scenario
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())

        for _, user_id in tqdm(enumerate(dataset.keys())):
            u_id = int(user_id)
            seen_movie_len = len(dataset[str(u_id)])
            indices = list(range(seen_movie_len))

            # Only include users whose item-consumption history length is between 13 and 100
            if seen_movie_len < 13 or seen_movie_len > 100:
                continue

            # Shuffle the indices randomly
            random.shuffle(indices)
            # Existing users
            tmp_x = np.array(dataset[str(u_id)])
            # New users
            tmp_y = np.array(dataset_y[str(u_id)])

            # Use 10 random items from the history as the query set to calculate training loss
            query_x_app = None
            for m_id in tmp_x[indices[-10:]]:
                m_id = int(m_id)
                u_id = int(user_id)
                tmp_x_converted = torch.cat((item_dict[m_id], user_dict[u_id]), 1)
                try:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                except:
                    query_x_app = tmp_x_converted

            # Use the remaining items as the support set to calculate test loss
            support_x_app = None
            for m_id in tmp_x[indices[:-10]]:
                m_id = int(m_id)
                tmp_x_converted = torch.cat((item_dict[m_id], user_dict[u_id]), 1)
                try:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                except:
                    support_x_app = tmp_x_converted

            support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
            query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])

            # Dump the support and query sets into pickle files
            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))

            # Load the support and query set info into text files
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:-10]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[-10:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1
