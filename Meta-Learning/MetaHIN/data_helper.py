# Import packages
import gc
import glob
import os
import pickle
from tqdm import tqdm
from multiprocessing.pool import ThreadPool


class DataHelper:
    """
    Construct a data helper class
    """

    def __init__(self, input_dir, output_dir, config):
        """
        Initialize the class
        :param input_dir: input directory
        :param output_dir: output directory
        :param config: experiment configuration
        """
        self.input_dir = input_dir  # ../../ml-1m/original/
        self.output_dir = output_dir  # processed/
        self.config = config  # configuration setting
        self.mp_list = self.config['mp']  # list of meta-paths

    def load_data(self, data_set, state, load_from_file=True):
        """
        Load all data
        :param data_set: dataset
        :param state: experiment states
        """
        data_dir = os.path.join(self.output_dir, data_set)
        supp_xs_s = []
        supp_ys_s = []
        supp_mps_s = []
        query_xs_s = []
        query_ys_s = []
        query_mps_s = []

        training_set_size = int(
            len(glob.glob("{}/{}/*.pkl".format(data_dir, state))) / self.config['file_num'])  # support, query

        # load all data
        for idx in tqdm(range(training_set_size)):
            support_x = pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb"))
            if support_x.shape[0] > 5:
                continue
            del support_x
            supp_xs_s.append(pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/{}/support_y_{}.pkl".format(data_dir, state, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(data_dir, state, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(data_dir, state, idx), "rb")))

            supp_mp_data, query_mp_data = {}, {}
            for mp in self.mp_list:
                supp_mp_data[mp] = pickle.load(open("{}/{}/support_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))
                query_mp_data[mp] = pickle.load(open("{}/{}/query_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))
            supp_mps_s.append(supp_mp_data)
            query_mps_s.append(query_mp_data)

        print('#support set: {}, #query set: {}'.format(len(supp_xs_s), len(query_xs_s)))

        # all training tasks
        total_data = list(zip(supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s))
        del (supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s)
        gc.collect()
        return total_data

    def load_batch_data(self, data_set, state, batch_indices, load_from_file=True):
        """
        Load data in batches
        :param data_set: dataset
        :param state: experiment states
        :param batch_indices: batch indices
        """
        data_dir = os.path.join(self.output_dir, data_set)

        supp_xs_s = []
        supp_ys_s = []
        supp_mps_s = []
        query_xs_s = []
        query_ys_s = []
        query_mps_s = []

        for idx in batch_indices:
            supp_xs_s.append(pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/{}/support_y_{}.pkl".format(data_dir, state, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(data_dir, state, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(data_dir, state, idx), "rb")))

            supp_mp_data, query_mp_data = {}, {}
            for mp in self.mp_list:
                supp_mp_data[mp] = pickle.load(open("{}/{}/support_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))
                query_mp_data[mp] = pickle.load(open("{}/{}/query_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))

            supp_mps_s.append(supp_mp_data)
            query_mps_s.append(query_mp_data)

        return supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s

    def load_data_multiprocess(self, data_set, state, batch_indices, load_from_file=True):
        """
        Load data using multi-processor
        :param data_set: dataset
        :param state: experiment states
        :param batch_indices: batch indices
        """
        data_dir = os.path.join(self.output_dir, data_set)
        global cur_state
        cur_state = state

        supp_xs_s = []
        supp_ys_s = []
        supp_mps_s = []
        query_xs_s = []
        query_ys_s = []
        query_mps_s = []

        pool = ThreadPool(processes=20)
        res = pool.map(self.load_single_data, batch_indices)
        for r in res:
            supp_xs_s.append(r[0])
            supp_ys_s.append(r[1])
            supp_mps_s.append(r[2])
            query_xs_s.append(r[3])
            query_ys_s.append(r[4])
            query_mps_s.append(r[5])
        return supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s

    def load_single_data(self, idx):
        """
        Load a single data point
        :param idx: current index
        """
        # Specify data directory
        data_dir = os.path.join(self.output_dir, self.config['dataset'])

        # Load support sets
        supp_xs = pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, cur_state, idx), "rb"))
        supp_ys = pickle.load(open("{}/{}/support_y_{}.pkl".format(data_dir, cur_state, idx), "rb"))

        # Load query sets
        query_xs = pickle.load(open("{}/{}/query_x_{}.pkl".format(data_dir, cur_state, idx), "rb"))
        query_ys = pickle.load(open("{}/{}/query_y_{}.pkl".format(data_dir, cur_state, idx), "rb"))

        supp_mp_data = {}
        query_mp_data = {}
        for mp in self.config['mp']:
            supp_mp_data[mp] = pickle.load(open("{}/{}/support_{}_{}.pkl".format(data_dir, cur_state, mp, idx), "rb"))
            query_mp_data[mp] = pickle.load(open("{}/{}/query_{}_{}.pkl".format(data_dir, cur_state, mp, idx), "rb"))

        return supp_xs, supp_ys, supp_mp_data, query_xs, query_ys, query_mp_data


# if __name__ == "__main__":
#     data_set = 'ml-1m'
#     input_dir = '../data/'
#     output_dir = '../data/'
#
#     data_helper = DataHelper(input_dir, output_dir, config)
#
#     training_set_size = int(len(glob.glob("../data/{}/{}/*.pkl".format(
#         data_set, 'meta_training'))) / config['file_num'])
#
#     indices = list(range(training_set_size))
#     random.shuffle(indices)
#     num_batch = int(training_set_size / 32)
#     start_time = time.time()
#
#     for idx, i in tqdm(enumerate(range(num_batch))):
#         cur_indices = indices[32*i:32*(i+1)]
#         support_xs, support_ys, support_mps, query_xs, query_ys, query_mps = \
#             data_helper.load_data_multiprocess(data_set, 'meta_training', cur_indices)
#
#     print(time.time()-start_time)
