import os, skimage, cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets.mnist as mnist

os.system("OPENCV_PYTHON")

path_file = os.path.dirname(os.path.realpath(__file__))
root = os.path.join(path_file, "datum")

# train data
train_set = (
    mnist.read_image_file(os.path.join(root, "train-images-idx3-ubyte")),
    mnist.read_label_file(os.path.join(root, "train-labels-idx1-ubyte"))
)

# test data
test_set = (
    mnist.read_image_file(os.path.join(root, "t10k-images-idx3-ubyte")),
    mnist.read_label_file(os.path.join(root, "t10k-labels-idx1-ubyte"))
)

print(">>> train set: ", train_set[0].size())
print(">>> test set: ",  test_set[0].size())

def convert_to_image(train=True):
    if train:
        f = open(os.path.join(root, "train.txt"), "w")
        image_path = os.path.join(root, "train")
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        for i, (image, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = os.path.join(image_path, str(i)+".jpg")
            # skimage.io.imsave(img_path, ima90ge.numpy())
            cv2.imwrite(img_path, image.numpy())
            f.write(img_path+" "+str(label)+"\n")
        f.close()
    else:
        f = open(os.path.join(root, "test.txt"), "w")
        image_path = os.path.join(root, "test")
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        for i, (image, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = os.path.join(image_path, str(i)+".jpg")
            # skimage.io.imsave(img_path, image.numpy())
            cv2.imwrite(img_path, image.numpy())
            f.write(img_path+" "+str(label)+"\n")
        f.close()


if __name__ == "__main__":
    convert_to_image(True)
    convert_to_image(False)

    os.system("OPENCV_ROS")