import torch

import numpy as np

from torch.utils.data import DataLoader, TensorDataset


def load_data(data, args):
    # Shuffle and extract the data
    idx = np.arange(0, data.shape[0], 1)
    np.random.shuffle(idx)

    num_train = int(np.ceil(data.shape[0] * args.train_test_ratio))
    num_test = int(data.shape[0] - num_train)

    # Extract training data from shuffled data
    train_feature = []
    train_label = []
    train_data = data[idx[:num_train], :, :]
    for i in range(num_train):
        sample_path = train_data[i, :, :].squeeze()
        for input_start in range(sample_path.shape[1] - 11):
            train_feature.append(sample_path[:, input_start].reshape(-1))
            train_label.append(sample_path[:, input_start + 1: input_start + 11].reshape(-1))
    train_feature = np.array(train_feature)
    train_label = np.array(train_label)

    # Extract testing data from shuffled data
    test_feature = []
    test_label = []
    test_data = data[idx[-num_test:], :, :]
    for i in range(num_test):
        sample_path = test_data[i, :, :].squeeze()
        for input_start in range(sample_path.shape[1] - 11):
            test_feature.append(sample_path[:, input_start].ravel())
            test_label.append(sample_path[:, input_start + 1: input_start + 11].ravel())
    test_feature = np.array(test_feature)
    test_label = np.array(test_label)

    # Create PyTorch datasets for training and testing
    train_dataset = TensorDataset(torch.Tensor(train_feature), torch.Tensor(train_label))
    test_dataset = TensorDataset(torch.Tensor(test_feature), torch.Tensor(test_label))

    # Create PyTorch data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader
