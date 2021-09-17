import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

import dense_transforms
import csv
import pdb 

class ClassificationDataset(Dataset):
    def __init__(self, dataset_path, transform=transforms.ToTensor()):

        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*.png')):
            self.files.append(im_f.replace('.png', ''))

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        img = Image.open(b + '.png')
        lbl = int(b.split('_c')[-1])
        return self.transform(img), lbl


class RegressionDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from glob import glob
        from os import path
        self.files = []
        self.csv_data = []
        for im_f in glob(path.join(dataset_path, '*_c1.png')):
            self.files.append(im_f.replace('.png', ''))
            reader = csv.reader(open(im_f.replace('png', 'csv'), 'r'))
            self.csv_data.append([float(i) for i in next(reader)])

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '.png')
        lbl = self.csv_data[idx]

        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = ClassificationDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_regression_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = RegressionDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return (self.matrix / (self.matrix.sum(1, keepdim=True) + 1e-5)).cpu()


if __name__ == '__main__':
    dataset = DenseSuperTuxDataset('dense_data/train', transform=dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor()]))
    from pylab import show, imshow, subplot, axis

    for i in range(15):
        im, lbl = dataset[i]
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(dense_transforms.label_to_pil_image(lbl))
        axis('off')
    show()
    import numpy as np

    c = np.zeros(5)
    for im, lbl in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))
