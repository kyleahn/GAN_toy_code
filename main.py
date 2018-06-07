import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as datasets

# dimension of image (in this case, 28 & 28)
image_h = 28
image_w = 28

try:
    # open MNIST data if we can
    with open('pixels.pkl', 'rb') as f:
        pixels = pickle.load(f)

except FileNotFoundError as e:
    # download MNIST data from web
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    # convert image to 1-D pixel data
    pixels = []
    for idx, (im, label) in enumerate(mnist):
        if label != 3:
            continue
        px = list(im.getdata())
        w, h = im.size
        px = [px[i * w:(i + 1) * w] for i in range(h)]
        pixels.append([x for sublist in px for x in sublist])
        if idx % (len(mnist) / 10) == 0:
            print(int(idx / len(mnist) * 100), "%")

    # write to file
    with open('pixels.pkl', 'wb') as f:
        pickle.dump(pixels, f)

# function that returns random noise (0 ~ 1)
def Z():
    return torch.rand(image_h, image_w)

# number of data
N = len(pixels)
# number of Hidden dimension
H = 1000

# define Generator
G = nn.Sequential(
    nn.Linear(image_h * image_w, H),
    nn.ReLU(),
    nn.Linear(H, image_h * image_w),
)

# define Discriminator
D = nn.Sequential(
    nn.Linear(image_h * image_w, H),
    nn.ReLU(),
    nn.Linear(H, 1),
)

# loss function (Mean Squared Error)
loss_fn = nn.MSELoss(size_average=False)

# number of iterations
iteration = 100

for _ in range(iteration):

    # number of k (train D for k times in a row)
    k = 10
    for _ in range(k):
        #train D
        pass

    #train G
    pass
