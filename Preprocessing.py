import os
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def default_loader(path):
    img = Image.open(path).convert("RGB")
    return img


class BirdDataset(Dataset):
    def __init__(
        self, path, img_list, train=True, transform=None, dataloader=default_loader
    ):
        self.meta = {}
        self.transform = transform
        self.dataloader = dataloader
        self.data_path = path
        if train:
            with open("dataset/training_labels.txt", "r") as file:
                for line in file.readlines():
                    img, label = line.split(" ")
                    img = int(img[:-4])
                    label = int(label[:3])
                    self.meta[img] = label
        self.train = train
        self.img_list = img_list

    def __getitem__(self, index):
        image_id = self.img_list[index]
        image_path = os.path.join(self.data_path, image_id)
        image_id = int(image_id[:-4])
        img = self.dataloader(image_path)
        img = self.transform(img)

        if not self.train:
            return [img, image_id]

        label = int(self.meta[image_id])
        label = torch.LongTensor([label])

        return [img, label]

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    training_data = BirdDataset(
        "dataset\\training_images",
        train=True,
        img_list=os.listdir(r"dataset\\training_images"),
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    training_data, val_data = random_split(training_data, [2800, 200])

    testing_data = BirdDataset(
        "dataset\\testing_images",
        img_list=os.listdir(r"dataset\\testing_images"),
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    # print(len(training_data))
    # print(len(val_data))
    image = training_data[1][0].numpy().transpose((1, 2, 0))
    plt.imshow(image / np.max(image))
    plt.show()
