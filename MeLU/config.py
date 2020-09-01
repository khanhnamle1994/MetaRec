# Experiment configuration
config = {
    # item parameters
    'num_rate': 6,  # number of rate categories
    'num_genre': 25,  # number of genres
    'num_director': 2186,  # number of directors
    'num_actor': 8030,  # number of actors
    'embedding_dim': 32,  # dimension of embedding vectors
    'first_fc_hidden_dim': 64,  # dimension of first decision-making layer
    'second_fc_hidden_dim': 64,  # dimension of second decision-making layer

    # user parameters
    'num_gender': 2,  # number of genders
    'num_age': 7,  # number of age groups
    'num_occupation': 21,  # number of occupations
    'num_zipcode': 3402,  # number of zipcodes

    # model setting
    'inner': 1,
    'lr': 5e-5,  # step size Beta
    'local_lr': 5e-6,  # step size Alpha
    'batch_size': 32,  # batch size
    'num_epoch': 20,  # number of epochs

    # candidate selection
    'num_candidate': 20,  # number of selected candidates
}

'''
Experiment Scenarios:
1 - warm_state: non-cold-start scenario
2 - user_cold_state: recommendations of existing items for new users
3 - item_cold_state: recommendations of new items for existing users
4 - user_and_item_cold_state: recommendations of new items for new users
'''
states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]
