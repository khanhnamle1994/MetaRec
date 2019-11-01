import tensorflow as tf
import model_helper

class RBM:
	'''
	Implementation of the Restricted Boltzmann Machine for collaborative filtering. The model is based on the paper of Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton: https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
	'''

	def __init__(self, FLAGS):
		'''Initialization of the model  '''

		self.FLAGS = FLAGS
		self.weight_initializer = model_helper._get_weight_init()
		self.bias_initializer = model_helper._get_bias_init()
		self.init_parameter()

	def init_parameter(self):
		'''Initializes the weights and the bias parameters of the neural network.'''

		with tf.variable_scope('Network_parameter'):
			self.W = tf.get_variable('Weights', shape = (self.FLAGS.num_v, self.FLAGS.num_h),initializer = self.weight_initializer)
			self.bh = tf.get_variable('hidden_bias', shape = (self.FLAGS.num_h), initializer = self.bias_initializer)
			self.bv = tf.get_variable('visible_bias', shape = (self.FLAGS.num_v), initializer = self.bias_initializer)

	def _sample_h(self, v):
		'''Uses the visible nodes for calculation of  the probabilities that a hidden neuron is activated.
		After that Bernouille distribution is used to sample the hidden nodes.

		@param v: visible nodes
		@return probability that a hidden neuron is activated
		@return sampled hidden neurons (value 1 or 0 accroding to Bernouille distribution)
		'''

		with tf.name_scope('sampling_hidden_units'):

			a = tf.nn.bias_add(tf.matmul(v,self.W), self.bh)
			p_h_v = tf.nn.sigmoid(a)
			h_ = self._bernouille_sampling(p_h_v, shape = [self.FLAGS.batch_size, int(p_h_v.shape[-1])])

			return p_h_v, h_

	def _sample_v(self, h):
		'''Uses the hidden nodes for calculation of  the probabilities that a visible neuron is activated.
		After that Bernouille distribution is used to sample the visible nodes.

		@param h: hidden nodes
		@return probability that a visible neuron is activated
		@return sampled visible neurons (value 1 or 0 accroding to Bernouille distribution)
		'''

		with tf.name_scope('sampling_visible_units'):

			a = tf.nn.bias_add(tf.matmul(h, tf.transpose(self.W, [1,0])), self.bv)
			p_v_h = tf.nn.sigmoid(a)
			v_ = self._bernouille_sampling(p_v_h, shape=[self.FLAGS.batch_size, int(p_v_h.shape[-1])])

			return p_v_h, v_

	def optimize(self, v):
		'''Optimization step. Gibbs sampling, calculating of gradients and doing an update operation.

		@param v: visible nodes
		@return update operation
		@return accuracy
		'''

		with tf.name_scope('optimization'):
			v0, vk, ph0, phk, _ = self._gibbs_sampling(v)
			dW, db_h, db_v = self._compute_gradients(v0, vk, ph0, phk)
			update_op = self._update_parameter(dW, db_h, db_v)

		with tf.name_scope('accuracy'):

			mask = tf.where(tf.less(v0, 0.0), x = tf.zeros_like(v0), y = tf.ones_like(v0))
			bool_mask = tf.cast(tf.where(tf.less(v0, 0.0), x = tf.zeros_like(v0), y = tf.ones_like(v0)), dtype = tf.bool)
			acc = tf.where(bool_mask, x = tf.abs(tf.subtract(v0, vk)), y = tf.zeros_like(v0))
			n_values = tf.reduce_sum(mask)
			acc = tf.subtract(1.0, tf.div(tf.reduce_sum(acc), n_values))

		return update_op, acc

	def inference(self, v):
		 '''Inference step. Training samples are used to activate the hidden neurons which are used for calculation of input neuron values.
		This new input values are the prediction, for already rated movies as well as not yet rated movies

		@param v: visible nodes
		@return sampled visible neurons (value 1 or 0 accroding to Bernouille distribution)
		'''
		p_h_v = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(v, self.W), self.bh))
		h_ = self._bernouille_sampling(p_h_v, shape=[1, int(p_h_v.shape[-1])])

		p_v_h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(h_, tf.transpose(self.W, [1,0])), self.bv))
		v_ = self._bernouille_sampling(p_v_h, shape=[1,int(p_v_h.shape[-1])])

		return v_

	def _update_parameter(self,dW,db_h,db_v):
		''' Creating TF assign operations. Updated weight and bias values are replacing old parameter values.

		@return assign operations
		'''

		alpha = self.FLAGS.learning_rate

		update_op = [tf.assign(self.W, alpha * tf.add(self.W, dW)),
				   tf.assign(self.bh, alpha * tf.add(self.bh, db_h)),
				   tf.assign(self.bv, alpha * tf.add(self.bv, db_v))]

		return update_op

	def _compute_gradients(self,v0, vk, ph0, phk):
		''' Computing the gradients of the weights and bias terms with Contrastive Divergence.

		@param v0: visible neurons before gibbs sampling
		@param vk: visible neurons after gibbs sampling
		@param ph0: probability that hidden neurons are activated before gibbs sampling.
		@param phk: probability that hidden neurons are activated after gibbs sampling.

		@return gradients of the network parameters
		'''

		#end condition for the while loop
		def condition(i, v0, vk, ph0, phk, dW,db_h,db_v):
			r = tf.less(i,k)
			return r[0]

		#loop body
		def body(i, v0, vk, ph0, phk, dW,dbh,dbv):

			v0_ = v0[i]
			ph0_ = ph0[i]

			vk_ = vk[i]
			phk_ = phk[i]

			#reshaping for making the outer product possible
			ph0_ = tf.reshape(ph0_, [1, self.FLAGS.num_h])
			v0_ = tf.reshape(v0_, [self.FLAGS.num_v, 1])
			phk_ = tf.reshape(phk_, [1, self.FLAGS.num_h])
			vk_ = tf.reshape(vk_, [self.FLAGS.num_v, 1])

			#calculating the gradiends for weights and biases
			dw_ = tf.subtract(tf.multiply(ph0_, v0_), tf.multiply(phk_, vk_))
			dbh_ = tf.subtract(ph0_, phk_)
			dbv_ = tf.subtract(v0_, vk_)

			dbh_ = tf.reshape(dbh_,[self.FLAGS.num_h])
			dbv_ = tf.reshape(dbv_,[self.FLAGS.num_v])

			return [i + 1, v0, vk, ph0, phk, tf.add(dW, dw_),tf.add(dbh, dbh_), tf.add(dbv, dbv_)]

		i = 0 # start counter for the while loop
		k = tf.constant([self.FLAGS.batch_size]) # number for the end condition of the while loop

		#init empty placeholders wherer the gradients will be stored
		dW = tf.zeros((self.FLAGS.num_v, self.FLAGS.num_h))
		dbh = tf.zeros((self.FLAGS.num_h))
		dbv = tf.zeros((self.FLAGS.num_v))

		#iterate over the batch and compute for each sample a gradient
		[i, v0, vk, ph0, phk, dW,db_h,db_v] = tf.while_loop(condition, body,[i, v0, vk, ph0, phk, dW,dbh,dbv])

		#devide the summed gradiends by the batch size
		dW = tf.div(dW, self.FLAGS.batch_size)
		dbh = tf.div(dbh, self.FLAGS.batch_size)
		dbv = tf.div(dbv, self.FLAGS.batch_size)

		return dW,dbh,dbv

	def _gibbs_sampling(self, v):
		''' Perfroming the gibbs sampling.

		@param v: visible neurons
		@return visible neurons before gibbs sampling
		@return visible neurons before gibbs sampling
		@return probability that hidden neurons are activated before gibbs sampling.
		@return probability that hidden neurons are activated after gibbs sampling.
		'''

		#end condition for the while loop
		def condition(i, vk, hk, v):
			r = tf.less(i,k)
			return r[0]

		#loop body
		def body(i, vk, hk,v):

			_,hk = self._sample_h(vk)
			_,vk = self._sample_v(hk)

			vk = tf.where(tf.less(v,0),v,vk)

			return [i + 1, vk, hk,v]

		ph0, _ =self._sample_h(v)

		vk = v
		hk = tf.zeros_like(ph0)

		i = 0 # start counter for the while loop
		k = tf.constant([self.FLAGS.k]) # number for the end condition of the while loop

		[i, vk, hk, v] = tf.while_loop(condition, body,[i, vk, hk, v])

		phk, _ = self._sample_h(vk)

		return v, vk, ph0, phk, i

	def _bernouille_sampling(self,p, shape):
		'''Samples from the Bernoulli distribution

		@param p: probability
		@return samples from Bernoulli distribution
		'''
		return tf.where(
						tf.less(p, tf.random_uniform(shape, minval = 0.0, maxval = 1.0)),
						x = tf.zeros_like(p),
						y = tf.ones_like(p)
				)
