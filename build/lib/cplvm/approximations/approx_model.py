from abc import ABC, abstractmethod, abstractproperty


class ApproximateModel(ABC):
    """A class representing a generic constrastive latent variable model"""

    def __init__(self, k_shared, k_foreground):
        self._k_shared = k_shared
        self._k_foreground = k_foreground
