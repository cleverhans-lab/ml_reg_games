import getpass

import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from .celeba_dataset import CelebADataset, CelebASensitive

user = getpass.getuser()


def get_celeba_all_set(args, dataset_path=f"/home/{user}/data/celeba/"):
    celeba_dataset_type = 'custom'

    if args.dataset == "celebasensitive":
        print("======using celeba sensitive=====")
        dataset_extractor = CelebASensitive
    elif celeba_dataset_type == 'custom':
        dataset_extractor = CelebADataset
    else:
        dataset_extractor = datasets.CelebA
        
    all_set = dataset_extractor(
        root=dataset_path,
        split='all',
        target_type='attr',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
        download=False)
    return all_set


def get_celeba_set(args, type='train'):
    all_set = get_celeba_all_set(args, dataset_path=args.dataset_path)
    if type == 'train':
        # Only a part of the dataset set is used for training.
        start_index = 0
        last_index = args.num_train_samples
    elif type == 'dev':
        start_index = args.num_train_samples
        last_index = start_index + args.num_val_samples
    elif type == 'test':
        start_index = args.num_train_samples + args.num_val_samples
        last_index = start_index + args.num_unlabeled_samples + args.num_test_samples
        if last_index != args.num_all_samples:
            raise Exception(
                f"The indexes do not sum up to the total data size.")
    else:
        raise Exception(f"Unknown data set type for CelebA: {type}.")
    return Subset(all_set, list(range(start_index, last_index)))


def get_celeba_train_set(args):
    return get_celeba_set(args=args, type='train')


def get_celeba_dev_set(args):
    # ones_per_example()
    return get_celeba_set(args=args, type='dev')


def get_celeba_test_set(args):
    # ones_per_example()
    return get_celeba_set(args=args, type='test')


def get_celeba_private_data(args):
    if args.dataset not in ['celeba', 'celebasensitive']:
        return None
    all_private_datasets = get_celeba_train_set(args=args)
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


def ones_per_example():
    all_set = get_celeba_all_set()
    ones = []
    for i, (_, target) in enumerate(all_set):
        # print('target: ', target)
        ones.append(target.sum().item())
        if i % 5000 == 0:
            print(i, ',mean,', np.mean(ones), ',median,', np.median(ones),
                  ',std,', np.std(ones))
    print(',mean,', np.mean(ones), ',median,', np.median(ones),
          ',std,', np.std(ones))


def main():
    ones_per_example()


if __name__ == "__main__":
    main()
