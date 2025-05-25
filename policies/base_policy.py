"""
READ-ONLY: Basic policy interface
"""
import abc
import numpy as np

class BasePolicy(object, metaclass=abc.ABCMeta):
    def get_action(self, image_seq: np.ndarray, height_seq: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, image_seq: np.ndarray, height_seq: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError
