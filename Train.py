import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.models.resnet import resnext50_32x4d
import matplotlib.pyplot as plt
import os
from Preprocessing import BirdDataset


def plt_loss(epoch, train_all_loss, val_all_loss):
    plt.plot(epoch, train_all_loss, label="training loss")
    plt.plot(epoch, val_all_loss, label="validation loss")
    plt.title("LOSS")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["training set", "validation set"], loc="center right")
    plt.show()


def plt_accuracy(epoch, train_all_accuracy, val_all_accuracy):
    plt.plot(epoch, train_all_accuracy, label="training accuracy")
    plt.plot(epoch, val_all_accuracy, label="validation accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("acc")
    plt.legend(["training set", "validation set"], loc="lower right")
    plt.show()


def train(model, criterion, optimizer):
    epoch = []
    train_all_loss = []
    train_all_accuracy = []
    val_all_loss = []
    val_all_accuracy = []
    epoch_num = 80
    batch_size = 16

    training_data = BirdDataset(
        "dataset\\training_images",
        train=True,
        img_list=os.listdir(r"dataset\\training_images"),
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    training_data, val_data = random_split(training_data, [2800, 200])

    train_dataloader = data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    val_dataloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for i in range(epoch_num):

        train_running_loss = 0.0
        count_train = 0
        train_correct = 0

        print("-----epoch", i + 1, "-----")
        # train
        model.train()
        for num, image in enumerate(train_dataloader):
            x_train, y_train = image
            x_train = torch.autograd.Variable(x_train.cuda())
            y_train = torch.autograd.Variable(y_train.squeeze().cuda())
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            count_train += len(y_train)
            train_correct += (pred == y_train).float().sum()
            # print(loss_correct)
            del x_train
            del y_train
            torch.cuda.empty_cache()

        # validation
        model.eval()
        val_running_loss = 0.0
        count_val = 0
        val_correct = 0
        with torch.no_grad():
            for num, image in enumerate(val_dataloader):
                x_train, y_train = image
                x_train = x_train.cuda()
                y_train = y_train.squeeze().cuda()
                optimizer.zero_grad()
                output = model(x_train)
                loss = criterion(output, y_train)
                val_running_loss += loss.item()
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                count_val += len(y_train)
                val_correct += (pred == y_train).float().sum()
                # print(val_correct)
                del x_train
                del y_train
                torch.cuda.empty_cache()

        print(
            "train_loss：",
            train_running_loss,
            " train_acc：",
            100 * train_correct / float(count_train),
        )
        print(
            "val_loss：  ",
            val_running_loss,
            " val_acc：  ",
            100 * val_correct / float(count_val),
        )

        epoch.append(i + 1)
        train_all_loss.append(train_running_loss)
        train_all_accuracy.append(100 * train_correct / float(count_train))
        val_all_loss.append(val_running_loss)
        val_all_accuracy.append(100 * val_correct / float(count_val))

    torch.save(model.state_dict(), "model_weights.pth")
    torch.save(model, "model.pth")
    # del model
    torch.cuda.empty_cache()
    plt_loss(epoch, train_all_loss, val_all_loss)
    plt_accuracy(epoch, train_all_accuracy, val_all_accuracy)


if __name__ == "__main__":

    train_model = resnext50_32x4d(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model.to(device)
    # define loss function (criterion) and optimizer
    loss_f = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(train_model.parameters(), lr=0.00002, weight_decay=0.003)
    train(train_model, loss_f, opt)
