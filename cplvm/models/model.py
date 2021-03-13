from abc import ABC, abstractmethod, abstractproperty

class ContrastiveModel(ABC):
	"""A class representing a generic constrastive latent variable model"""
	def __init__(self, k_shared, k_foreground):
		self._k_shared = k_shared
		self._k_foreground = k_foreground

	def __call__(self, var_param):
		pass

	@abstractproperty
	def model(self, data_dim, num_datapoints_x, num_datapoints_y, counts_per_cell_X, counts_per_cell_Y, *args):
		"""The TFP model."""
		pass

	@abstractmethod
	def _fit_model_vi(self):
		"""Fit model with variational inference (VI)"""
		pass

	@abstractmethod
	def _fit_model_map(self):
		"""Fit model with maximum a posteriori (MAP) inference"""
		pass
