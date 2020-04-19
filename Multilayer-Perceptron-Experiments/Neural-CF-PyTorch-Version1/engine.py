# Import libraries
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# Import functions from metrics.py and utils.py scripts
from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK


class Engine(object):
    """
    Meta Engine for training & evaluating NCF model
    """

    def __init__(self, config):
        """
        Function to initialize the engine
        :param config: configuration dictionary
        """
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)  # Metrics for Top-10
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # Tensorboard Writer
        self._writer.add_text('config', str(config), 0) # String output for Tensorboard Writer
        self.opt = use_optimizer(self.model, config)  # set optimizer

        # self.crit = torch.nn.MSELoss() # mean squared error loss for explicit feedback
        self.crit = torch.nn.BCELoss()  # binary cross entropy loss for implicit feedback

    def train_single_batch(self, users, items, ratings):
        """
        Function to train a single batch with back-propagation
        :param users: user data
        :param items: item data
        :param ratings: rating data
        :return: Loss value
        """

        assert hasattr(self, 'model'), 'Please specify the exact model !'

        # if self.config['use_cuda'] is True:
        #     users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        self.opt.zero_grad()
        ratings_pred = self.model(users, items)

        # Get the loss with the choice of pre-defined loss function
        loss = self.crit(ratings_pred.view(-1), ratings)
        # Back-propagate the loss
        loss.backward()
        # Optimize the loss
        self.opt.step()
        # Get the final loss
        loss = loss.item()

        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """
        Function to train a single epoch
        :param train_loader: a Loader class for the training data
        :param epoch_id: current epoch
        :return:
        """
        assert hasattr(self, 'model'), 'Please specify the exact model !'

        # Initialize training mode for current model
        self.model.train()
        # Initialize total loss
        total_loss = 0

        # Loop through batches in the training data
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)

            # Get user, item, and rating data
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()

            # Train a single batch
            loss = self.train_single_batch(user, item, rating)

            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            # Add up total loss
            total_loss += loss

        # Save the loss values to be displayed on TensorBoard
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        """
        Function eo evaluate the model on test data
        :param evaluate_data: data array to be evaluated
        :param epoch_id: current epoch
        :return: values of Hit Ratio and NDCG metrics
        """
        assert hasattr(self, 'model'), 'Please specify the exact model !'

        # Initialize evaluation mode for current model
        self.model.eval()

        # Use 'no_grad' to reduce the memory usage and speed up computations (no Gradient Calculation)
        with torch.no_grad():
            # Get test user and test item data
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            # Get negative user and negative item data
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]

            # if self.config['use_cuda'] is True:
            #     test_users = test_users.cuda()
            #     test_items = test_items.cuda()
            #
            #     negative_users = negative_users.cuda()
            #     negative_items = negative_items.cuda()

            # Calculate test scores
            test_scores = self.model(test_users, test_items)
            # Calculate negative scores
            negative_scores = self.model(negative_users, negative_items)

            # if self.config['use_cuda'] is True:
            #
            #     test_users = test_users.cpu()
            #     test_items = test_items.cpu()
            #     test_scores = test_scores.cpu()
            #
            #     negative_users = negative_users.cpu()
            #     negative_items = negative_items.cpu()
            #     negative_scores = negative_scores.cpu()

            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                     test_items.data.view(-1).tolist(),
                                     test_scores.data.view(-1).tolist(),
                                     negative_users.data.view(-1).tolist(),
                                     negative_items.data.view(-1).tolist(),
                                     negative_scores.data.view(-1).tolist()]

        # Calculate Hit Ratio and NDCG values
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()

        # Save the HR and NDCG values to be displayed on TensorBoard writer
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)

        print('[Evaluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))

        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        """
        Function to save information for every run
        :param alias: alias info
        :param epoch_id: current epoch
        :param hit_ratio: value of Hit Ratio metric
        :param ndcg: value of NDCG metric
        """
        assert hasattr(self, 'model'), 'Please specify the exact model !'

        # Choose the model directory where the model will be saved
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        # Save the model
        save_checkpoint(self.model, model_dir)
