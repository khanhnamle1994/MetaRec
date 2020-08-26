# Model configuration
config = {
    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,
    'embedding_dim': 32,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,

    # model setting
    'inner': 1,
    'lr': 5e-5,
    'local_lr': 5e-6,
    'batch_size': 32,
    'num_epoch': 20,

    # candidate selection
    'num_candidate': 20,
}

# Model states
states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]