from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from Prepocessing import BirdDataset
import torchvision.transforms as transforms
import os


# 讀取classes
def load_classes():
    f = open("dataset/classes.txt")
    classes = {}
    for line in f.readlines():
        class_number, class_name = line.replace("\n", "").split(".")
        classes[int(class_number)] = class_name
    # print(classes)
    f.close()
    return classes


# 讀取testing順序
def load_testing_order():
    f = open("dataset/testing_img_order.txt")
    test_images = [x.strip() for x in f.readlines()]  # all the testing images
    f.close()
    # print(test_images)

    f = open("dataset/testing_img_order.txt")
    testing_img_order = []
    for line in f.readlines():
        testing_img_order.append(int(line[:4]))

    f.close()
    # print(testing_img_order)
    return test_images, testing_img_order


def predict_submission(classes, test_images, testing_img_order):
    model = torch.load("model.pth")
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predict_id = []
    predict_label = []
    predict_class = []

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
    test_dataloader = DataLoader(testing_data, batch_size=16, shuffle=False)

    with torch.no_grad():
        for num, image in enumerate(test_dataloader):
            x_train, y_train = image
            x_train = Variable(x_train.cuda())
            y_train = Variable(y_train.cuda())
            # print (y_train)
            # print (x_train)
            y_pred = model(x_train)
            _, pred = torch.max(y_pred, 1)
            # print(pred)
            predict_id = predict_id + list(y_train.cpu().numpy())
            predict_label = predict_label + list(pred.cpu().numpy())
        # print(predict_id)
        # print (predict_label)

    for i in predict_label:
        predict_class.append(classes.get(i))
    # print(predict_class)

    path = "answer.txt"
    with open(path, "w") as f:
        j = 0
        for i in testing_img_order:
            index = predict_id.index(i)
            if predict_label[index] < 10:
                predict = "00" + str(predict_label[index])
            elif predict_label[index] < 100:
                predict = "0" + str(predict_label[index])
            else:
                predict = str(predict_label[index])
            print(test_images[j] + " " + predict + "." + predict_class[index])
            f.write(test_images[j] + " " + predict + "." + predict_class[index])
            f.write("\n")
            j += 1


if __name__ == "__main__":
    cls = load_classes()
    test_img, testing_order = load_testing_order()
    predict_submission(cls, test_img, testing_order)
