import getpass
import os

import numpy as np
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

from .fairface_dataset import FairfaceDataset

user = getpass.getuser()


def get_fairface_set(args,split, dataset_path=f'data/'):
    """
    Function that enables to load the set with respective split form the fairface dataset
    """
    full_path = args.data_dir
    #full_path = os.path.join(os.getcwd(), dataset_path)

    dataset_extractor = FairfaceDataset

    all_set = dataset_extractor(
        root=full_path,
        split=split,
        target_feat='gender',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
        download=False)
    return all_set


def get_fairface_train_set(args):
    return get_fairface_set(args,split='train')


def get_fairface_dev_set(args):
    return get_fairface_set(args,split='valid')


def get_fairface_test_set(args):
    return get_fairface_set(args,split='test')


def get_fairface_all_set(args):
    return get_fairface_set(args,split='all')


def get_fairface_balanced_private_data(args):
    ''' split the dataset into balanced subset'''
    if args.dataset != 'fairface':
        return None
    all_private_datasets = get_fairface_train_set(args)
    private_dataset_size = len(all_private_datasets) // args.num_models
    all_private_trainloaders = []
    sss = StratifiedShuffleSplit(n_splits=args.num_models, train_size=1.0, random_state=None)
    # using this as placeholder for the training data
    X = np.zeros(len(all_private_datasets.filename))
    for indices, _ in sss.split(X, all_private_datasets.gender):
        private_dataset = Subset(all_private_datasets, indices)
        kwargs = args.kwargs
        private_trainloader = DataLoader(
            dataset=private_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        all_private_trainloaders.append(private_trainloader)
    return all_private_trainloaders


def get_fairface_private_data(args):
    if args.dataset != 'fairface':
        return None
    all_private_datasets = get_fairface_train_set(args)
    # exclude the validation set
    start = 0
    end = len(all_private_datasets) - args.num_val_samples
    all_private_datasets =Subset(dataset=all_private_datasets, indices=list(range(start, end)))
    private_dataset_size = len(all_private_datasets) // args.num_models
    all_private_trainloaders = []
    for i in range(args.num_models):
        begin = i * private_dataset_size
        if i == args.num_models - 1:
            end = len(all_private_datasets)
        else:
            end = (i + 1) * private_dataset_size
        indices = list(range(begin, end))
        private_dataset = Subset(all_private_datasets, indices)
        kwargs = args.kwargs
        private_trainloader = DataLoader(
            dataset=private_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        all_private_trainloaders.append(private_trainloader)
    return all_private_trainloaders


def main():
    # ones_per_example()
    pass


if __name__ == "__main__":
    main()
