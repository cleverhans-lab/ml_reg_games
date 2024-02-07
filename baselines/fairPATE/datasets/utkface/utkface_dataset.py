from functools import partial
import torch
import os
import numpy as np
import PIL

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive
from .utkface_labels import infer_information_from_filename

base_folder = "utkface"


class UTKfaceDataset(VisionDataset):
    """`UTKface Dataset https://susanqq.github.io/UTKFace/`_ Dataset.
    Args:
    root (string): Root directory where images are downloaded to.
    split (string): One of {'train', 'valid', 'test', 'all'}.
        Accordingly dataset is selected.
    target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
        or ``landmarks``. Can also be a list to output a tuple with all specified target types.
        The targets represent:
            ``age``
            ``gender``
            ``race``
        Defaults to ``age``. If empty, ``None`` will be returned as target.
    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.ToTensor``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "utkface"

    file_list = [
        # File ID                                                    Filename
        ('0BxYys69jI14kU0I1YUQyY1ZDRUE', 'UTKFace.tar.gz'),
    ]

    def __init__(self, args, root, split="train", target_feat="gender", sensitive_feat="race", transform=None,
                 target_transform=None, download=False):
        import pandas
        super(UTKfaceDataset, self).__init__(root, transform=transform,
                                             target_transform=target_transform)
        self.split = split

        if download:
            self.download()

        fn = partial(os.path.join, self.root, self.base_folder)

        # todo make consistent with fairface where all data is used (here we leave the "unlabeled out")
        # but we might not leave them out at the right place....
        self.num_all_samples = args.num_all_samples
        self.num_train_samples = args.num_train_samples
        self.num_test_samples = args.num_test_samples

        # define indices for different sets to start
        train_start = 0
        test_start = train_start + self.num_train_samples
        end = test_start + self.num_test_samples + args.num_unlabeled_samples

        rng = np.random.default_rng(42)  

        # unfortunately, 3 files suffer from missing information
        # we need to sort them out
        #all_files = [file for file in all_files if file.count('_') == 3]
        #rng.shuffle(all_files)  # shuffle this list -> then we can sample to the sets by indices
        all_files = np.load("./baselines/fairPATE/datasets/utkface/utkface_files.npy")

        if split == 'train':
            self.filename = all_files[train_start:test_start]
        elif split == 'test':
            self.filename = all_files[test_start:end]
        elif split == 'all':
            self.filename = all_files
        else:
            raise ValueError("Please provide a valid split {'train', 'valid', 'test', 'all'}")

        assert len(self.filename) == len(
            set(self.filename)), "Your sampling has sampled the same datapoint more than once"

        information = [infer_information_from_filename(fn) for fn in
                       self.filename]  # get the three attributes from the file name
        information = np.array(information, dtype=np.float32)  # convert to an array

        self.target_feat = target_feat
        self.sensitive_feat = sensitive_feat

        self.age = torch.as_tensor(information[:, 0])
        self.gender = torch.as_tensor(information[:, 1])
        self.race = torch.as_tensor(information[:, 2])

    def download(self):
        # todo: attention: download function cannot be used. Data must be added manually once.
        import zipfile

        # does not work because of issue: https://github.com/pytorch/vision/issues/2992
        # but also does not work with: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
        for (file_id, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root,
                                                                  self.base_folder),
                                            filename)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder,
                                          "UTKFace.tar.gz"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    # get one data point
    def __getitem__(self, index):
        X = PIL.Image.open(
            os.path.join(self.root, self.base_folder, "UTKFace",
                         self.filename[index])).convert('RGB')
        targets = []
        for t in [self.target_feat, self.sensitive_feat]:
            if t == "age":
                targets.append(self.age[index])
            elif t == "gender":
                targets.append(self.gender[index])
            elif t == "race":
                targets.append(self.race[index])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        target = targets[0]
        sensitive_attribute = targets[1]

        if self.target_transform is not None:
            target = self.target_transform(target)
            # convert the target to longTensor
            target = target.type(torch.LongTensor)

        sensitive_attribute = sensitive_attribute.type(torch.LongTensor)

        return X, target, sensitive_attribute

    def __len__(self):
        return len(self.age)  # just take a random one of all the lists: all have same length

    # todo: see what it does and it I need to adapt it
    def extra_repr(self):
        lines = ["Target variable: {target_feat}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    pass

