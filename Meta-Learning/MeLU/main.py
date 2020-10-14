# Import libraries
import os
import torch
import pickle

# Import utility scripts
from MeLU import MeLU
from config import config
from train import training
from data_generator import generate
from evidence_candidate import selection

if __name__ == "__main__":
    master_path = "./ml"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # Preparing the dataset. It needs about 22GB of your hard disk space.
        generate(master_path)

    # Training the model
    melu = MeLU(config)
    model_filename = "{}/models.pkl".format(master_path)
    if not os.path.exists(model_filename):
        # Load the training dataset
        training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
        # The support set is for local update
        supp_xs_s = []
        supp_ys_s = []
        # The query set is for global update
        query_xs_s = []
        query_ys_s = []

        # Populate the support and query sets with data from loaded pickle files
        for idx in range(training_set_size):
            supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))
        total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
        del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

        # Train MeLU
        training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'],
                 model_save=True, model_filename=model_filename)
    else:
        # Load pretrained MeLU model
        trained_state_dict = torch.load(model_filename)
        melu.load_state_dict(trained_state_dict)

    # Select evidence candidates
    evidence_candidate_list = selection(melu, master_path, config['num_candidate'])
    for movie, score in evidence_candidate_list:
        print(movie, score)
