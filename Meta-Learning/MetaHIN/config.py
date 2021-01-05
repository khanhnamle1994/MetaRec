# MetaHIN configuration
config = {
    'dataset': 'movielens',  # specify MovieLens1M dataset
    'mp': ['um', 'umum', 'umam', 'umdm'],  # a set of meta-paths
    'file_num': 12,  # each task contains 12 files for movielens

    # item parameters
    'num_rate': 6,  # number of rate categories
    'num_genre': 25,  # number of genres
    'num_fea_item': 2,  # number of item features
    'item_fea_len': 26,  # length of item features

    # user parameters
    'num_gender': 2,  # number of genders
    'num_age': 7,  # number of age groups
    'num_occupation': 21,  # number of occupations
    'num_zipcode': 3402,  # number of zipcodes
    'num_fea_user': 4,  # number of user features

    # model setting
    'embedding_dim': 32,  # number of dimensions in the embedding layer
    'user_embedding_dim': 32*4,  # 4 features
    'item_embedding_dim': 32*2,  # 2 features

    'first_fc_hidden_dim': 64,  # number of dimensions in the first fully-connected hidden layer
    'second_fc_hidden_dim': 64,  # number of dimensions in the second fully-connected hidden layer
    'mp_update': 1,  # meta-path update
    'local_update': 1,  # local update
    'lr': 5e-4,  # step size Beta (global learning rate)
    'mp_lr': 5e-3,  # meta-path learning rate
    'local_lr': 5e-3,  # step size Alpha (local learning rate)
    'batch_size': 32,  # number of tasks for each batch
    'num_epoch': 100,  # number of epochs
    'neigh_agg': 'mean',  # neighborhood aggregation
    'mp_agg': 'mean'  # meta-path aggregation
}

'''
Experiment Scenarios:
1 - warm_up: recommendations of existing items for existing users
2 - user_cold_testing: recommendations of existing items for new users
3 - item_cold_testing: recommendations of new items for existing users
4 - user_and_item_cold_testing: recommendations of new items for new users
'''
states = ["meta_training", "warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing"]
