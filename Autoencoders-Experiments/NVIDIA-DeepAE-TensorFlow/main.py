# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from data_processor import preprocess_data
import tensorflow as tf
from model import Deep_AE_model
from tensorflow.keras.optimizers import Adam
from utils import show_error, show_rmse, masked_rmse, masked_rmse_clip

# Load the ratings data
df = pd.read_csv('data/ml1m_ratings.csv', sep='\t', encoding='latin-1',
                 usecols=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])

# +1 is the real size, as they are zero based
num_users = df['user_emb_id'].unique().max() + 1
num_movies = df['movie_emb_id'].unique().max() + 1

# Split the train (80%), validation (10%), and test (10%) sets
train_df, test_df = train_test_split(df, stratify=df['user_emb_id'],
                                     test_size=0.1, random_state=999613182)
train_df, validate_df = train_test_split(train_df, stratify=train_df['user_emb_id'],
                                         test_size=0.1, random_state=999613182)

# Create sparse pivot tables with users in rows and items in columns
users_items_matrix_train_zero = preprocess_data(train_df, num_users, num_movies, 0)
users_items_matrix_train_one = preprocess_data(train_df, num_users, num_movies, 1)
users_items_matrix_train_two = preprocess_data(train_df, num_users, num_movies, 2)
users_items_matrix_train_three = preprocess_data(train_df, num_users, num_movies, 3)
users_items_matrix_train_four = preprocess_data(train_df, num_users, num_movies, 4)
users_items_matrix_train_five = preprocess_data(train_df, num_users, num_movies, 5)
users_items_matrix_train_average = preprocess_data(train_df, num_users, num_movies, average=True)
users_items_matrix_validate = preprocess_data(validate_df, num_users, num_movies, 0)
users_items_matrix_test = preprocess_data(test_df, num_users, num_movies, 0)

# Convert data types from int64 to float32 to use as tensor inputs for Keras model
users_items_matrix_train_zero = tf.convert_to_tensor(users_items_matrix_train_zero, dtype=tf.float32)
users_items_matrix_train_one = tf.convert_to_tensor(users_items_matrix_train_one, dtype=tf.float32)
users_items_matrix_train_two = tf.convert_to_tensor(users_items_matrix_train_two, dtype=tf.float32)
users_items_matrix_train_three = tf.convert_to_tensor(users_items_matrix_train_three, dtype=tf.float32)
users_items_matrix_train_four = tf.convert_to_tensor(users_items_matrix_train_four, dtype=tf.float32)
users_items_matrix_train_five = tf.convert_to_tensor(users_items_matrix_train_five, dtype=tf.float32)
users_items_matrix_train_average = tf.convert_to_tensor(users_items_matrix_train_average, dtype=tf.float32)
users_items_matrix_validate = tf.convert_to_tensor(users_items_matrix_validate, dtype=tf.float32)
users_items_matrix_test = tf.convert_to_tensor(users_items_matrix_test, dtype=tf.float32)

# Model hyper-parameters
layers = [256, 512, 256]
dropout = 0.5
activation = 'selu'
last_activation = 'selu'
regularizer_encode = 0.001
regularizer_decode = 0.001

# Build model
Deep_AE = Deep_AE_model(users_items_matrix_train_zero,
                        layers, activation, last_activation, dropout,
                        regularizer_encode, regularizer_decode)
# Compile model
Deep_AE.compile(optimizer=Adam(lr=0.0001), loss=masked_rmse, metrics=[masked_rmse_clip])
# Get model summary
Deep_AE.summary()
# Train the model and evaluate on validation set
hist_Deep_AE = Deep_AE.fit(
    x=users_items_matrix_train_zero, y=users_items_matrix_train_zero,
    epochs=500, batch_size=256,
    validation_data=[users_items_matrix_train_zero, users_items_matrix_validate], verbose=2)

# Show model loss
show_error(hist_Deep_AE, 100)
# Show RMSE metric
show_rmse(hist_Deep_AE, 100)

# Predict the model on test set
predict_deep = Deep_AE.predict(users_items_matrix_train_zero)
test_result_deep = Deep_AE.evaluate(users_items_matrix_train_zero, users_items_matrix_test)
