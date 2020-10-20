# Import packages
import os
import pickle
from tqdm import tqdm
import numpy as np

# Import utility script
from preprocess_MovieLens import store_index, store_dict


def store_id(max_count=20):
    """
    Store the user and item IDs
    :param max_count: maximum number of count
    """
    storing_path = 'processed_data'
    if not os.path.exists(storing_path):
        os.mkdir(storing_path)
    if not os.path.exists('{}'.format(storing_path)):
        os.mkdir('{}'.format(storing_path))

    store_index(max_count=max_count)


def check_state(u_id, i_id, u_state_ids, i_state_ids):
    """
    Check the warm-cold states of the user and item data
    """
    if u_id in u_state_ids['user_warm_ids']:
        # Warm users and warm items
        if i_id in i_state_ids['item_warm_ids']:
            code = 0
        # Warm users and cold items
        else:
            code = 1

    else:
        # Cold users and warm items
        if i_id in i_state_ids['item_warm_ids']:
            code = 2
        # Cold users and cold items
        else:
            code = 3
    return code


def generate_data():
    """
    Generate train and test samples, where each sample is from a user and includes both support and query sets
    """
    storing_path = 'processed_data'

    if not os.path.exists('{}/raw'.format(storing_path)):
        os.mkdir('{}/raw/'.format(storing_path))
        processing_code = 1
    elif not os.path.exists('{}/raw/sample_1_x1.p'.format(storing_path)):
        processing_code = 1
    else:
        processing_code = 0
        print("Data already generated!")

    if processing_code == 1:
        sorted_ratings = pickle.load(open('{}/ratings_sorted.p'.format(storing_path), 'rb'))
        user_state_ids = pickle.load(open('{}/user_state_ids.p'.format(storing_path), 'rb'))
        item_state_ids = pickle.load(open('{}/item_state_ids.p'.format(storing_path), 'rb'))
        user_dict = pickle.load(open('{}/user_dict.p'.format(storing_path), 'rb'))
        item_dict = pickle.load(open('{}/item_dict.p'.format(storing_path), 'rb'))

        u_all_ids = user_state_ids['user_all_ids']

        file_index = 1

        for u_id in tqdm(u_all_ids):
            u_info = sorted_ratings.loc[sorted_ratings.user_id == u_id]
            ratings = np.array(u_info.rating)
            u_feature = user_dict[u_id]

            u_i_ids = u_info.item_id

            i_feature_file = []
            state_codes = []

            for i_id in u_i_ids:
                if i_id in item_dict.keys():
                    i_feature = item_dict[i_id]
                    state_code = check_state(u_id, i_id, user_state_ids, item_state_ids)
                    i_feature_file.append(i_feature)
                    state_codes.append(state_code)

            # Dump user and item features, state codes, and ratings into Pickle file
            pickle.dump(u_feature, open('{}/raw/'.format(storing_path) +
                                        'sample_' + str(file_index) + '_x1.p', 'wb'))
            pickle.dump(i_feature_file, open('{}/raw/'.format(storing_path) +
                                             'sample_' + str(file_index) + '_x2.p', 'wb'))
            pickle.dump(ratings, open('{}/raw/'.format(storing_path) +
                                      'sample_' + str(file_index) + '_y.p', 'wb'))
            pickle.dump(state_codes, open('{}/raw/'.format(storing_path) +
                                          'sample_' + str(file_index) + '_y0.p', 'wb'))

            file_index += 1


if __name__ == '__main__':
    # Execute the functions above
    store_id()
    store_dict()
    generate_data()
