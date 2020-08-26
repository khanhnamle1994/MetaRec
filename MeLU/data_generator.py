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


def item_converting(row, genre_list):
    """
    Convert item data into suitable format
    :param row: current row
    :param genre_list: list of movie genres
    """
    # There are 25 genres in total
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    return torch.cat(genre_idx, 1)


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    """
    Convert user data into suitable format
    :param row: current row
    :param gender_list: list of genders
    :param age_list: list of ages
    :param occupation_list: list of occupations
    :param zipcode_list: list of zipcodes
    """
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


def generate(master_path):
    # Path to MovieLens 1M
    dataset_path = "../ml-1m"

    # Load genre, gender, age, occupation, and zipcode lists
    genre_list = load_list("{}/m_genre.txt".format(dataset_path))
    gender_list = load_list("{}/m_gender.txt".format(dataset_path))
    age_list = load_list("{}/m_age.txt".format(dataset_path))
    occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
    zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

    if not os.path.exists("{}/warm_state/".format(master_path)):
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    # Initialize dataset class
    dataset = movielens_1m()

    # Use a hashmap to store movie items
    if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
        movie_dict = {}
        for idx, row in dataset.item_data.iterrows():
            m_info = item_converting(row, genre_list)
            movie_dict[row['movie_id']] = m_info
        pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
    else:
        movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))

    # Use a hashmap to store user profiles
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in dataset.user_data.iterrows():
            u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    # Loop through different model states
    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())
        for _, user_id in tqdm(enumerate(dataset.keys())):
            u_id = int(user_id)
            seen_movie_len = len(dataset[str(u_id)])
            indices = list(range(seen_movie_len))

            if seen_movie_len < 13 or seen_movie_len > 100:
                continue

            random.shuffle(indices)
            tmp_x = np.array(dataset[str(u_id)])
            tmp_y = np.array(dataset_y[str(u_id)])

            # Create a support set
            support_x_app = None
            for m_id in tmp_x[indices[:-10]]:
                m_id = int(m_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                except:
                    support_x_app = tmp_x_converted

            # Create a query set
            query_x_app = None
            for m_id in tmp_x[indices[-10:]]:
                m_id = int(m_id)
                u_id = int(user_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                except:
                    query_x_app = tmp_x_converted
            support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
            query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])

            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:-10]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[-10:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1
