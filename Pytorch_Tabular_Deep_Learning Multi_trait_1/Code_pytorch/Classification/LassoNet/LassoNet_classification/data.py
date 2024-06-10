import numpy as np
import pandas as pd

def in_place_shuffle(ndarray, seed=42):
    marker_value = ndarray.ravel()[0]
    np.random.default_rng(seed).shuffle(ndarray)
    marker_new_pos = np.argwhere(ndarray == marker_value).ravel()
    print(f'ndarray shuffled in place with seed {seed}.')
    print(f'Sanity checking: the original 0-th row has been moved to {marker_new_pos[0]}-th line.')


def in_place_normalize(_4darray):
    print(f'Normalizing across all rows...')
    mu = np.mean(_4darray, axis=0, keepdims=True)
    std = np.std(_4darray, axis=0, keepdims=True)
    std = np.where(std != 0, std, 1e-7)  # in case some columns have std 0
    _4darray -= mu
    _4darray /= std
    print(
        f'Sanity checking: The {0}-th column of size {_4darray[:, 0].shape} now has {_4darray[:, 0].mean()} mean and {_4darray[:, 0].std()} std.')


def train_test_indices(num_of_samples, test_ratio):
    num_test = int(num_of_samples * test_ratio)
    training_indices, testing_indices = np.split(np.arange(num_of_samples), [num_of_samples - num_test])
    print(
        f'Split {num_of_samples} samples in total into {training_indices.shape[0]} training samples and {testing_indices.shape[0]} testing samples.')
    return training_indices, testing_indices


def split_by_indices(features, labels, training_indices, testing_indices):
    train_feature = features[training_indices, :]
    train_label = labels[training_indices, :]
    test_feature = features[testing_indices, :]
    test_label = labels[testing_indices, :]
    print(f'train_feature.shape {train_feature.shape}')
    print(f'train_label.shape {train_label.shape}')
    print(f'test_feature.shape {test_feature.shape}')
    print(f'test_label.shape {test_label.shape}')
    return train_feature, train_label, test_feature, test_label


def load_wheat_csv_ds(dataset_path, testing_set_ratio=0.1):
    """
    Return the tuple (train_feature, train_label, test_feature, test_label)
    """
    pd_ds = pd.read_csv(dataset_path, header=None)
    npdata=pd_ds.to_numpy()
    in_place_shuffle(npdata)
    labels = npdata[:,:4] # first 4 columns
    features = npdata[:, 4:]
    in_place_normalize(features)
    num_of_samples = labels.shape[0]
    training_indices, testing_indices = train_test_indices(num_of_samples, testing_set_ratio)

    train_feature, train_label, test_feature, test_label = split_by_indices(features, labels, training_indices, testing_indices)

    return train_feature, train_label, test_feature, test_label