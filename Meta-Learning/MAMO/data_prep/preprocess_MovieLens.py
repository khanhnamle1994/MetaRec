# ===========================================================================
# Data Preprocessing for MovieLens
# userIDs range from 1 to 6040
# itemIDs range from 1 to 3952
# User Info includes gender, age, occupation
# Item Info includes title, year, genres, director, and rate
# Rating Info includes ratings userID has given to itemID (mean = 3.58)
#
# Basic Info:
# User State: Warm users = 5400, Cold Users = 640
# Item State: Warm Items = 1683, Cold Items = 1645
# ===========================================================================

# Import libraries
import pandas as pd
import datetime
import os
import pickle
from tqdm import tqdm
import numpy as np

# Import utility script
from prepare_list import user_converting, item_converting, list_movielens


def load_movielens():
    """
    Load MovieLens 1M
    :return: user, item, and ratings information
    """
    # Path to store the processed data
    path = "../../ml-1m"

    # Path to user, item, and ratings data
    user_info_path = "{}/users.dat".format(path)
    rating_info_path = "{}/ratings.dat".format(path)
    item_info_path = "{}/movies_extrainfos.dat".format(path)  # additional movie contents from IMDB

    # Read user data CSV file
    user_info = pd.read_csv(
        user_info_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
        sep="::", engine='python')
    user_info = user_info.drop(columns=['zip'])

    # Read item data CSV file
    item_info = pd.read_csv(
        item_info_path, names=['item_id', 'title', 'year', 'rate', 'released',
                               'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
        sep="::", engine='python', encoding="utf-8")
    item_info = item_info.drop(columns=['released', 'writer', 'actors', 'plot', 'poster'])

    # Read rating data CSV file
    ratings = pd.read_csv(
        rating_info_path, names=['user_id', 'item_id', 'rating', 'timestamp'],
        sep="::", engine='python')

    # Change 'timestamp' into 'time' with datetime format
    ratings['time'] = ratings["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
    ratings = ratings.drop(["timestamp"], axis=1)
    return user_info, item_info, ratings


def store_index(max_count=20):
    """
    Store indices of processed data
    :param max_count: maximum number of counts
    """
    storing_path = 'processed_data'

    if not os.path.exists('{}/user_state_ids.p'.format(storing_path)):
        # Load the raw data
        _, _, ratings = load_movielens()

        # Sort the ratings by time
        sorted_time = ratings.sort_values(by='time', ascending=True).reset_index(drop=True)
        start_time, split_time, end_time = sorted_time['time'][0], sorted_time['time'][
            round(0.8 * len(ratings))], sorted_time['time'][len(ratings) - 1]
        print('Start time: %s, Split time: %s, End time: %s' % (start_time, split_time, end_time))

        # Sort the users by their timed ratings
        sorted_users = ratings.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)

        '''Collect user statistics'''
        user_warm_list, user_cold_list, user_counts = [], [], []
        new_df = pd.DataFrame()

        # Capture unique userIDs
        user_ids = ratings.user_id.unique()
        n_user_ids = ratings.user_id.nunique()

        for u_id in tqdm(user_ids):
            u_info = sorted_users.loc[sorted_users.user_id == u_id].reset_index(drop=True)
            u_count = len(u_info)

            if u_count > max_count - 1:
                new_u_info = u_info.iloc[:max_count, :]
                new_df = new_df.append(new_u_info, ignore_index=True)

                u_time = u_info['time'][0]
                if u_time < split_time:
                    user_warm_list.append(u_id)
                else:
                    user_cold_list.append(u_id)

            user_counts.append(u_count)

        print('Number of warm users: %d, Number of cold users: %d' % (len(user_warm_list), len(user_cold_list)))
        print('Mininum count: %d, Average count: %d, Maximum count: %d' % (min(user_counts),
                                                                           len(ratings) / n_user_ids, max(user_counts)))

        new_all_ids = new_df.user_id.unique()

        # Store user statistics in user_state_ids
        user_state_ids = {'user_all_ids': new_all_ids,
                          'user_warm_ids': user_warm_list,
                          'user_cold_ids': user_cold_list}

        '''Collect item statistics'''
        sorted_items = new_df.sort_values(by=['item_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        item_warm_list, item_cold_list, item_counts = [], [], []

        # Capture unique itemIDs
        item_ids = sorted_items.item_id.unique()
        n_item_ids = sorted_items.item_id.nunique()

        for i_id in tqdm(item_ids):
            i_info = sorted_items.loc[sorted_items.item_id == i_id].reset_index(drop=True)
            i_count = len(i_info)

            if i_count > 10:
                item_warm_list.append(i_id)
            else:
                item_cold_list.append(i_id)
            item_counts.append(i_id)

        print('Number of warm items: %d, Number of cold items: %d' % (len(item_warm_list), len(item_cold_list)))
        print('Minimum count: %d, Average count: %d, Maximum count: %d' % (min(item_counts),
                                                                           len(ratings) / n_item_ids, max(item_counts)))

        # Store item statistics in item_state_ids
        item_state_ids = {'item_all_ids': item_ids,
                          'item_warm_ids': item_warm_list,
                          'item_cold_ids': item_cold_list}

        # Dump the user and item statistics into pickle files
        pickle.dump(new_df, open('{}/ratings_sorted.p'.format(storing_path), 'wb'))
        pickle.dump(item_state_ids, open('{}/item_state_ids.p'.format(storing_path), 'wb'))
        pickle.dump(user_state_ids, open('{}/user_state_ids.p'.format(storing_path), 'wb'))

    else:
        print('User and item ID information has already been stored.')


def store_dict():
    """
    Store dictionaries of processed data
    """
    storing_path = 'processed_data'

    # Load pickle files of user and item statistics
    user_state_ids = pickle.load(open('{}/user_state_ids.p'.format(storing_path), 'rb'))
    item_state_ids = pickle.load(open('{}/item_state_ids.p'.format(storing_path), 'rb'))

    # Store user and item dictionaries
    user_info, item_info, _ = load_movielens()

    # User
    user_all_features = []
    user_all_ids = user_state_ids['user_all_ids']

    user_dict = {}
    for u_id in tqdm(user_all_ids):
        row = user_info.loc[user_info['user_id'] == u_id]
        # User features include age, gender, and occupation
        feature_vector = user_converting(user_row=row, age_list=list_movielens['list_age'],
                                         gender_list=list_movielens['list_gender'],
                                         occupation_list=list_movielens['list_occupation'])
        user_all_features.append(feature_vector)

    user_all_features = np.array(user_all_features)

    count = 0

    for u_id in tqdm(user_all_ids):
        u_info = user_all_features[count]
        user_dict[u_id] = u_info
        count += 1

    # Dump all user features into a Pickle file
    pickle.dump(user_dict, open('{}/user_dict.p'.format(storing_path), 'wb'))

    # Item
    item_all_features = []
    item_all_ids = item_state_ids['item_all_ids']
    item_dict = {}
    updated_i_id = []
    year_list = list(item_info.year.unique())

    for i_id in tqdm(item_all_ids):
        row = item_info.loc[item_info['item_id'] == i_id]
        if len(row) > 0:
            # Item features include rate, genres, directors, and year
            feature_vector = item_converting(item_row=row, rate_list=list_movielens['list_rate'],
                                             genre_list=list_movielens['list_genre'],
                                             director_list=list_movielens['list_director'],
                                             year_list=year_list)
            updated_i_id.append(i_id)
            item_all_features.append(feature_vector)

    item_all_features = np.array(item_all_features)

    count = 0

    for i_id in tqdm(updated_i_id):
        i_info = item_all_features[count]
        item_dict[i_id] = i_info
        count += 1

    # Dump all item features into a Pickle file
    pickle.dump(item_dict, open('{}/item_dict.p'.format(storing_path), 'wb'))
