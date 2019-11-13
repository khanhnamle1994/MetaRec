import torch
from engine import Engine
from utils import use_cuda
from utils import resume_checkpoint

class RBM:
	'''
	Implementation of the Restricted Boltzmann Machine for collaborative filtering. The model is based on the paper of Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton: https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
	'''

	def __init__(self, num_visible, num_hidden, k, momentum_coefficient=0.5, config):
		'''Initialization of the model  '''
		super(RBM, self).__init__()

		self.config = config
		self.num_users = config['num_users']
		self.num_items = config['num_items']

		self.num_visible = config['num_visible']
		self.num_hidden = config['num_hidden']
		self.k = config['k']
		self.momentum_coefficient = config['momentum_coefficient']

		self.use_cuda = use_cuda

		self.weights = torch.randn(num_visible, num_hidden) * 0.1
		self.visible_bias = torch.ones(num_visible) * 0.5
		self.hidden_bias = torch.zeros(num_hidden)

		self.weights_momentum = torch.zeros(num_visible, num_hidden)
		self.visible_bias_momentum = torch.zeros(num_visible)
		self.hidden_bias_momentum = torch.zeros(num_hidden)

		if self.use_cuda:
			self.weights = self.weights.cuda()
			self.visible_bias = self.visible_bias.cuda()
			self.hidden_bias = self.hidden_bias.cuda()

			self.weights_momentum = self.weights_momentum.cuda()
			self.visible_bias_momentum = self.visible_bias_momentum.cuda()
			self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

	def sample_hidden(self, visible_probabilities):
		'''Uses the visible nodes for calculation of the probabilities that a hidden neuron is activated.'''

		hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
		hidden_probabilities = self._sigmoid(hidden_activations)
		return hidden_probabilities

	def sample_visible(self, hidden_probabilities):
		'''Uses the hidden nodes for calculation of  the probabilities that a visible neuron is activated.'''

		visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
		visible_probabilities = self._sigmoid(visible_activations)
		return visible_probabilities

	def contrastive_divergence(self, input_data):
		''' Computing the gradients of the weights and bias terms with Contrastive Divergence.'''

		# Positive phase
		positive_hidden_probabilities = self.sample_hidden(input_data)
		positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
		positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

		# Negative phase
		hidden_activations = positive_hidden_activations

		for step in range(self.k):
			visible_probabilities = self.sample_visible(hidden_activations)
			hidden_probabilities = self.sample_hidden(visible_probabilities)
			hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

		negative_visible_probabilities = visible_probabilities
		negative_hidden_probabilities = hidden_probabilities

		negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

		# Update parameters
		self.weights_momentum *= self.momentum_coefficient
		self.weights_momentum += (positive_associations - negative_associations)

		self.visible_bias_momentum *= self.momentum_coefficient
		self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

		self.hidden_bias_momentum *= self.momentum_coefficient
		self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

		batch_size = input_data.size(0)

		self.weights += self.weights_momentum * self.learning_rate / batch_size
		self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
		self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

		self.weights -= self.weights * self.weight_decay  # L2 weight decay

		# Compute reconstruction error
		error = torch.sum((input_data - negative_visible_probabilities)**2)

		return error

	def _sigmoid(self, x):
		'''Sigmoid function'''
		return 1 / (1 + torch.exp(-x))

	def _random_probabilities(self, num):
		'''Generate random probabilities'''
		random_probabilities = torch.rand(num)

		if self.use_cuda:
			random_probabilities = random_probabilities.cuda()

		return random_probabilities

class RBMEngine(Engine):
    """Engine for training & evaluating RBM model"""

	def __init__(self, config):
		self.model = RBM(config)

		if config['use_cuda'] is True:
			use_cuda(True, config['device_id'])
			self.model.cuda()

		super(RBMEngine, self).__init__(config)
		print(self.model)
