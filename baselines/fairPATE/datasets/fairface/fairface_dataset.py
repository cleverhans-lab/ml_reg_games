from functools import partial
import torch
import os
import PIL
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, \
    check_integrity, verify_str_arg
from .fairface_labels import race_dict_inv, gender_dict_inv, age_dict_inv

base_folder = "fairface"


class FairfaceDataset(VisionDataset):
    """`Fairface Dataset <https://github.com/joojs/fairface>`_ Dataset.
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

    base_folder = "fairface/FairFace"

    file_list = [
        # File ID                                                    Filename
        ('1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH', 'fairface_label_train.csv'),
        ('1wOdja-ezstMEp81tX1a-EYkFebev4h7D', 'fairface_label_val.csv'),
        ('1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86', 'fairface-img-margin025-trainval.zip'),
    ]

    def __init__(self, root, split="train", target_feat="gender", sensitive_feat="race", transform=None,
                 target_transform=None, download=False):
        import pandas
        super(FairfaceDataset, self).__init__(root, transform=transform,
                                              target_transform=target_transform)
        self.split = split

        if download:
            self.download()

        fn = partial(os.path.join, self.root, self.base_folder)

        # return the whole "validation set". Splitting it into test and unlabeled is taken 
        # care of by the util functions
        if split == 'test' or split == 'valid':
            information = pandas.read_csv(fn("val_labels.csv"),
                                          header=0)
        elif split == 'train':  # I misuse what the dataset had as validation data as our test data
            information = pandas.read_csv(fn("train_labels.csv"),
                                          header=0)
        elif split == 'all':  # In case of all, I return the concatenation of train + test data
            information_train = pandas.read_csv(fn("train_labels.csv"),
                                                header=0)
            information_test = pandas.read_csv(fn("val_labels.csv"),
                                               header=0)
            information = pandas.concat([information_train, information_test])
        else:
            raise ValueError("Please provide a valid split {'train', 'valid', 'test', 'all'}")

        # do the data processing according to the dictionaries for labels
        information.replace({"age": age_dict_inv}, inplace=True)
        information.replace({"gender": gender_dict_inv}, inplace=True)
        information.replace({"race": race_dict_inv}, inplace=True)

        self.target_feat = target_feat
        self.sensitive_feat = sensitive_feat
        self.target_feat_col_id = information.columns.get_loc(target_feat)
        self.sensitive_feat_col_id = information.columns.get_loc(sensitive_feat)

        self.filename = information['file']
        self.age = torch.as_tensor(information['age'].values)
        self.gender = torch.as_tensor(information['gender'].values)
        self.race = torch.as_tensor(information['race'].values)

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
                                          "fairface-img-margin025-trainval.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    # get one data point
    def __getitem__(self, index):  
        X = PIL.Image.open(
            os.path.join(self.root, self.base_folder,
                         self.filename[index]))

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



if __name__ == "__main__":
    import getpass

    user = getpass.getuser()
    root = os.path.join(os.getcwd(), f'../../data/')
    fairface = FairfaceDataset(root=root, split='train')
    X, target, sens = fairface.__getitem__(0)
    breakpoint()
