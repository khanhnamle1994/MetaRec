# Import packages
import os
import pandas as pd

# Define constants to store path to data directory
MOVIELENS_DIR = '../../ml-1m/'
USER_DATA_FILE = 'users.dat'
MOVIE_DATA_FILE = 'movies.dat'
RATING_DATA_FILE = 'ratings.dat'

# Define constants to store age and occupation columns
AGES = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATIONS = {0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
               4: "college/grad student", 5: "customer service", 6: "doctor/health care",
               7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
               12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
               17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}

# Define constants to store data as CSV files
RATINGS_CSV_FILE = 'data/ml1m_ratings.csv'
USERS_CSV_FILE = 'data/ml1m_users.csv'
MOVIES_CSV_FILE = 'data/ml1m_movies.csv'

# Save the ratings data from .dat to .csv format
ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE),
                      sep='::',
                      engine='python',
                      encoding='latin-1',
                      names=['userid', 'movieid', 'rating', 'timestamp'])

max_userid = ratings['userid'].drop_duplicates().max()
max_movieid = ratings['movieid'].drop_duplicates().max()
ratings['user_emb_id'] = ratings['userid'] - 1
ratings['movie_emb_id'] = ratings['movieid'] - 1
print(len(ratings), 'ratings loaded')

ratings.to_csv(RATINGS_CSV_FILE,
               sep='\t',
               header=True,
               encoding='latin-1',
               columns=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])
print('Saved to', RATINGS_CSV_FILE)

# Save the users data from .dat to .csv format
users = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE),
                    sep='::',
                    engine='python',
                    encoding='latin-1',
                    names=['userid', 'gender', 'age', 'occupation', 'zipcode'])

users['age_desc'] = users['age'].apply(lambda x: AGES[x])
users['occ_desc'] = users['occupation'].apply(lambda x: OCCUPATIONS[x])
print(len(users), 'descriptions of', max_userid, 'users loaded.')
users['user_emb_id'] = users['userid'] - 1

users.to_csv(USERS_CSV_FILE,
             sep='\t',
             header=True,
             encoding='latin-1',
             columns=['user_emb_id', 'gender', 'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc'])
print('Saved to', USERS_CSV_FILE)

# Save the movies data from .dat to .csv format
movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE),
                     sep='::',
                     engine='python',
                     encoding='latin-1',
                     names=['movieid', 'title', 'genre'])
print(len(movies), 'descriptions of', max_movieid, 'movies loaded.')

movies['movie_emb_id'] = movies['movieid'] - 1
movies.to_csv(MOVIES_CSV_FILE,
              sep='\t',
              header=True,
              columns=['movie_emb_id', 'title', 'genre'])
print('Saved to', MOVIES_CSV_FILE)

print(len(ratings['userid'].drop_duplicates()), 'of the', max_userid, 'users rate at least one movie.')
print(len(ratings['movieid'].drop_duplicates()), 'of the', max_movieid, 'movies are rated.')
