from torch.utils.data.sampler import Sampler
import numpy as np
class MultiViewSampler(Sampler):
    def __init__(self, dataset_size, nview):
        """
        Custom sampler to shuffle and group multi-view data for DataLoader.
        Args:
            dataset_size (int): Total number of samples in the dataset.
            nview (int): Number of views per object.
        """
        self.dataset_size = dataset_size
        self.nview = nview
        self.num_objects = dataset_size // nview

    def __iter__(self):
        # Generate shuffled object indices
        object_indices = np.random.permutation(self.num_objects) * self.nview

        # Generate view indices for each object
        view_indices = np.zeros((self.nview, self.num_objects), dtype=int)
        for i in range(self.nview):
            view_indices[i] = object_indices + i

        # Flatten and return as an iterator
        shuffled_indices = view_indices.T.flatten()
        return iter(shuffled_indices)

    def __len__(self):
        return self.dataset_size
