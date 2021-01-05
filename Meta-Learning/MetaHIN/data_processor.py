# Import packages
import os
import json
import pandas as pd
import numpy as np
import torch
import re
import random
import pickle
import os
from tqdm import tqdm
import collections

random.seed(13)

# Data directories
input_dir = '../../ml-1m/original/'
output_dir = 'processed-data'

# List of possible states
states = ["warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing", "meta_training"]

if not os.path.exists("{}/meta_training/".format(output_dir)):
    os.mkdir("{}/log/".format(output_dir))
    for state in states:
        os.mkdir("{}/{}/".format(output_dir, state))
        if not os.path.exists("{}/{}/{}".format(output_dir, "log", state)):
            os.mkdir("{}/{}/{}".format(output_dir, "log", state))

# Load ratings data
ui_data = pd.read_csv(input_dir + 'ratings.dat', names=['user', 'item', 'rating', 'timestamp'],
                      sep="::", engine='python')
print("Number of ratings:", len(ui_data))

# Load user data
user_data = pd.read_csv(input_dir + 'users.dat', names=['user', 'gender', 'age', 'occupation_code', 'zip'],
                        sep="::", engine='python')

# Load item data
item_data = pd.read_csv(input_dir + 'movies_extrainfos.dat',
                        names=['item', 'title', 'year', 'rate', 'released', 'genre',
                               'director', 'writer', 'actors', 'plot', 'poster'],
                        sep="::", engine='python', encoding="utf-8")

user_list = list(set(ui_data.user.tolist()) | set(user_data.user))
item_list = list(set(ui_data.item.tolist()) | set(item_data.item))

user_num = len(user_list)
item_num = len(item_list)
print("Number of users:", user_num, "and Number of items:", item_num)

"""
1 - Code to process user and item features
"""


def load_list(fname):
    """
    Function to load a file into a Python list
    :param fname: file name
    :return: Python list
    """
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


rate_list = load_list("{}/m_rate.txt".format(input_dir))  # list of rate levels
genre_list = load_list("{}/m_genre.txt".format(input_dir))  # list of genres
actor_list = load_list("{}/m_actor.txt".format(input_dir))  # list of actors
director_list = load_list("{}/m_director.txt".format(input_dir))  # list of directors
gender_list = load_list("{}/m_gender.txt".format(input_dir))  # list of genders
age_list = load_list("{}/m_age.txt".format(input_dir))  # list of ages
occupation_list = load_list("{}/m_occupation.txt".format(input_dir))  # list of occupations
zipcode_list = load_list("{}/m_zipcode.txt".format(input_dir))  # list of zipcodes

# Verify the lists
print("Number of rate levels:", len(rate_list), "\n",
      "Number of genres:", len(genre_list), "\n",
      "Number of actors:", len(actor_list), "\n",
      "Number of directors:", len(director_list), "\n",
      "Number of gender:", len(gender_list), "\n",
      "Number of age:", len(age_list), "\n",
      "Number of occupation:", len(occupation_list), "\n",
      "Number of zipcodes:", len(zipcode_list))


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    """
    Convert item data into PyTorch tensor
    :param row: current row
    :param rate_list: list of rate levels
    :param genre_list: list of movie genres
    :param director_list: list of directors
    :param actor_list: list of actors
    """
    # Convert rate_list to PyTorch Tensor
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()

    # Convert genre_list to PyTorch Tensor
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1  # one-hot vector

    # Convert director_list to PyTorch Tensor
    director_idx = torch.zeros(1, 2186).long()
    director_id = []
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
        director_id.append(idx + 1)  # id starts from 1, not index

    # Convert actor_list to PyTorch Tensor
    actor_idx = torch.zeros(1, 8030).long()
    actor_id = []
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
        actor_id.append(idx + 1)

    # Concatenate PyTorch tensors into one-dimensional tensor
    return torch.cat((rate_idx, genre_idx), 1), \
           torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1), \
           director_id, actor_id


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    """
    Convert user data into PyTorch tensor
    :param row: current row
    :param gender_list: list of genders
    :param age_list: list of ages
    :param occupation_list: list of occupations
    :param zipcode_list: list of zipcodes
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
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)  # (1, 4)


# Create a hash map for item features
movie_fea_hete = {}
movie_fea_homo = {}
m_directors = {}
m_actors = {}
for idx, row in item_data.iterrows():
    m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
    movie_fea_hete[row['item']] = m_info[0]
    movie_fea_homo[row['item']] = m_info[1]
    m_directors[row['item']] = m_info[2]
    m_actors[row['item']] = m_info[3]

# Create a hash map for user features
user_fea = {}
for idx, row in user_data.iterrows():
    u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
    user_fea[row['user']] = u_info

"""
2 - Code to process meta-path features
"""


def reverse_dict(d):
    # {1:[a,b,c], 2:[a,f,g],...}
    re_d = collections.defaultdict(list)
    for k, v_list in d.items():
        for v in v_list:
            re_d[v].append(k)
    return dict(re_d)


a_movies = reverse_dict(m_actors)
d_movies = reverse_dict(m_directors)
print("Actor dictionary:", len(a_movies), " and Director dictionary:", len(d_movies))


def jsonKeys2int(x):
    """
    Turn JSON keys into integers
    """
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x


state = 'meta_training'

# Load user features support set
support_u_movies = json.load(open(output_dir + state + '/support_u_movies.json', 'r'), object_hook=jsonKeys2int)
# Load user features query set
query_u_movies = json.load(open(output_dir + state + '/query_u_movies.json', 'r'), object_hook=jsonKeys2int)
# Load user target support set
support_u_movies_y = json.load(open(output_dir + state + '/support_u_movies_y.json', 'r'), object_hook=jsonKeys2int)
# Load user target query set
query_u_movies_y = json.load(open(output_dir + state + '/query_u_movies_y.json', 'r'), object_hook=jsonKeys2int)

if support_u_movies.keys() == query_u_movies.keys():
    u_id_list = support_u_movies.keys()
print(len(u_id_list))

train_u_movies = {}
if support_u_movies.keys() == query_u_movies.keys():
    u_id_list = support_u_movies.keys()
print(len(u_id_list))

for idx, u_id in tqdm(enumerate(u_id_list)):
    train_u_movies[int(u_id)] = []
    train_u_movies[int(u_id)] += support_u_movies[u_id] + query_u_movies[u_id]
len(train_u_movies)
