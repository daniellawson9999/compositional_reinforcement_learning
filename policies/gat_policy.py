from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable


from distributions import Normal 
from policies import NNPolicy2
from misc import tf_utils
from rllab.misc import tensor_utils
import time
dis=tf.contrib.distributions
EPS = 1e-6

def linear(X, dout, name, bias=True):
	with tf.variable_scope(name):
		dX = int(X.get_shape()[-1])
		W = tf.get_variable('W', shape=(dX, dout))
		if bias:
			b = tf.get_variable('b', initializer=tf.constant(np.zeros(dout).astype(np.float32)))
		else:
			b = 0
	return tf.matmul(X, W)+b

def relu_layer(X, dout, name):
	return tf.nn.relu(linear(X, dout, name))

def decoder(x, layers=2, d_hidden=32):
	out = x
	for i in range(layers):
		out = relu_layer(out, dout=d_hidden, name='l%d'%i)
	return out

def gat_layer(x, name, embed_size=10):
	with tf.variable_scope(name):
		W_input = tf.get_variable("W_input", [x.shape[-1], embed_size], initializer=tf.contrib.layers.xavier_initializer())
		features = tf.matmul(x, W_input)
		scores = tf.matmul(features, tf.transpose(features,perm=[0,2,1]))
		weights = tf.nn.softmax(scores, axis=-1)
		result = tf.matmul(weights, features)
		result = tf.nn.leaky_relu(result)
		return result

class GaussianGatPolicy(NNPolicy2, Serializable):
	def __init__(self, env_spec, hidden_layer_sizes=(100, 100), reg=1e-3, squash=True, reparameterize=False, name='gauss_gatpolicy', embed_size=10):
		Serializable.quick_init(self, locals())
		self._Da = env_spec.action_space.flat_dim
		self._Ds = env_spec.observation_space.flat_dim
		self.embed_size = embed_size
		self._squash = squash
		self._reg = reg
		self._is_deterministic = False
		self._reparameterize = reparameterize
		self.name=name
		self.n_hiddens = hidden_layer_sizes[0]
		self._scope_name = (tf.get_variable_scope().name + "/" + name).lstrip("/")
		self.initializer=tf.contrib.layers.xavier_initializer()
		self.build()
		super(NNPolicy2, self).__init__(env_spec)
		
	def actions_for(self, observations, sub_level_actions,name=None, reuse=tf.AUTO_REUSE,with_log_pis=False, regularize=False):
		name = name or self.name

		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			action_embeddings = gat_layer(sub_level_actions, name="gat_layer_1", embed_size=self.n_hiddens)
			action_embeddings = gat_layer(action_embeddings, name="gat_layer_2", embed_size=self.n_hiddens)
			input_embedding = tf.reduce_sum(action_embeddings, axis=[1])
			
			input_decoder=tf.concat([observations,input_embedding],axis=1)
			decoder_output=decoder(input_decoder,layers=2,d_hidden=self.n_hiddens)
			
			Wref_i=tf.get_variable("Wref_f",[1,self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			Wd=tf.get_variable("Wd",[self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			v=tf.get_variable("v",[self.n_hiddens], initializer=self.initializer)

			We_i=tf.nn.conv1d(action_embeddings,Wref_i,1,"VALID", name="We_i")
			Wd=tf.expand_dims(tf.matmul(decoder_output,Wd, name="Wd"),1)
			scores=tf.reduce_sum(v*tf.tanh(We_i + Wd),[-1],name="scores") #BatchXseq_len

			sm_scores=tf.nn.softmax(scores/0.5,name="sm_scores") #BatchXseq_len
			scores_index=tf.argmax(sm_scores,axis=1)	

		actions =tf.reduce_sum(tf.multiply(sub_level_actions,tf.expand_dims(sm_scores,2)),1)
		raw_actions =tf.reduce_sum(tf.multiply(sub_level_actions,tf.expand_dims(sm_scores,2)),1)
		# TODO: should always return same shape out
		# Figure out how to make the interface for `log_pis` cleaner
		if with_log_pis:
			# TODO.code_consolidation: should come from log_pis_for
			return actions,sm_scores,self._squash_correction(raw_actions)

		return actions

	
	def build(self):
		self._observations_ph= tf.placeholder(tf.float32,(None,self._Ds),name='observations',)
		self.sub_level_actions= tf.placeholder(tf.float32,(None,None,self._Da),name='sub_level_actions',)
		
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.action_embeddings = gat_layer(self.sub_level_actions, name="gat_layer_1", embed_size=self.n_hiddens)
			self.action_embeddings = gat_layer(self.action_embeddings, name="gat_layer_2", embed_size=self.n_hiddens)
			self.input_embedding = tf.reduce_sum(self.action_embeddings, axis=[1])
			
			self.input_decoder=tf.concat([self._observations_ph,self.input_embedding],axis=1)
			self.decoder_output=decoder(self.input_decoder,layers=2,d_hidden=self.n_hiddens)
			
			self.Wref_i=tf.get_variable("Wref_f",[1,self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			self.Wd=tf.get_variable("Wd",[self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			self.v=tf.get_variable("v",[self.n_hiddens], initializer=self.initializer)

			self.We_i=tf.nn.conv1d(self.action_embeddings,self.Wref_i,1,"VALID", name="We_i")
			self.Wd=tf.expand_dims(tf.matmul(self.decoder_output,self.Wd, name="Wd"),1)
			self.scores=tf.reduce_sum(self.v*tf.tanh(self.We_i + self.Wd),[-1],name="scores") #BatchXseq_len

			self.sm_scores=tf.nn.softmax(self.scores/0.5,name="sm_scores") #BatchXseq_len
			self.scores_index=tf.argmax(self.sm_scores,axis=1)	
			self.one_hot=tf.one_hot(self.scores_index,tf.shape(self.sub_level_actions)[1])
		
		self._actions =tf.reduce_sum(tf.multiply(self.sub_level_actions,tf.expand_dims(self.sm_scores,2)),1)
	@overrides
	def get_actions(self, observations,sub_level_actions):
		"""Sample actions based on the observations.

		If `self._is_deterministic` is True, returns the mean action for the 
		observations. If False, return stochastically sampled action.

		TODO.code_consolidation: This should be somewhat similar with
		`LatentSpacePolicy.get_actions`.
		"""
		feed_dict = {self._observations_ph: observations,self.sub_level_actions: sub_level_actions}
		if self._is_deterministic: # Handle the deterministic case separately
			mu = tf.get_default_session().run(self._actions, feed_dict)  # 1 x Da

			return mu
		return super(GaussianGatPolicy, self).get_actions(observations,sub_level_actions) 
	def _squash_correction(self, actions):
		if not self._squash: return 0
		return tf.reduce_sum(tf.log(1 - actions ** 2 + EPS), axis=1)

	@contextmanager
	def deterministic(self, set_deterministic=True, latent=None):
		"""Context manager for changing the determinism of the policy.

		See `self.get_action` for further information about the effect of
		self._is_deterministic.

		Args:
			set_deterministic (`bool`): Value to set the self._is_deterministic
				to during the context. The value will be reset back to the
				previous value when the context exits.
			latent (`Number`): Value to set the latent variable to over the
				deterministic context.
		"""
		was_deterministic = self._is_deterministic

		self._is_deterministic = set_deterministic

		yield

		self._is_deterministic = was_deterministic
if __name__ == "__main__":
	from rllab.envs.normalized_env import normalize
	from envs import CrossMazeAntEnv
	policy  = GausianGatPolicy(normalize(CrossMazeAntEnv()))		
