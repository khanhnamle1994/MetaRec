# Import libraries
import re


def load_list(file_name):
    """
    Load the file as a Python list
    :param file_name: file name
    :return: Python list
    """
    list_ = []
    with open(file_name, encoding='utf-8') as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


# MovieLens Information
list_movielens = {
    'list_age': load_list('../../ml-1m/m_age.txt'),
    'list_gender': load_list('../../ml-1m/m_gender.txt'),
    'list_occupation': load_list('../../ml-1m/m_zipcode.txt'),
    'list_genre': load_list('../../ml-1m/m_genre.txt'),
    'list_rate': load_list('../../ml-1m/m_rate.txt'),
    'list_director': load_list('../../ml-1m/m_director.txt')
}


def user_converting(user_row, age_list, gender_list, occupation_list):
    """
    Convert user data into suitable format
    :param user_row: current user row
    :param age_list: list of ages
    :param gender_list: list of genders
    :param occupation_list: list of occupations
    :return: gender, age, and occupation information of the user in a list
    """
    # gender_dim = 2
    gender_idx = gender_list.index(user_row.iat[0, 1])
    # age_dim = 7
    age_idx = age_list.index(user_row.iat[0, 2])
    # occupation_dim = 21
    occupation_idx = occupation_list.index(user_row.iat[0, 3])
    return [gender_idx, age_idx, occupation_idx]


def item_converting(item_row, rate_list, genre_list, director_list, year_list):
    """
    Convert item data into suitable format
    :param item_row: current item row
    :param rate_list: list of rate levels
    :param genre_list: list of genres
    :param director_list: list of directors
    :param year_list: list of years
    :return: rate levels, genres, directors, and year information of the item in a list
    """
    # rate_dim = 6
    rate_idx = rate_list.index(item_row.iat[0, 1])
    # genre_dim = 25
    genre_idx = [0] * 25
    for genre in str(item_row.iat[0, 4]).split(", "):
        idx = genre_list.index(genre)
        genre_idx[idx] = 1
    # director_dim = 2186
    director_idx = [0] * 2186
    for director in str(item_row.iat[0, 5]).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[idx] = 1
    # year_dim = 1
    year_idx = year_list.index(item_row.iat[0, 2])

    out_list = list([rate_idx, year_idx])
    out_list.extend(genre_idx)
    out_list.extend(director_idx)
    return out_list
