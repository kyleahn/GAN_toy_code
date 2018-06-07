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
    raise FileNotFoundError
    # open MNIST data if we can
    with open('pixels.pkl', 'rb') as f:
        real_data = pickle.load(f)

except FileNotFoundError as e:
    # download MNIST data from web
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    print(mnist[0])
    print(mnist[1])
    print(mnist[2])

    # convert image to 1-D pixel data
    real_data = []
    for idx, (im, label) in enumerate(mnist):
        if label != 3:
            continue
        temp = list(im.getdata())
        w, h = im.size
        temp = [temp[i * w:(i + 1) * w] for i in range(h)]
        temp = [x for sublist in temp for x in sublist]
        temp = [x / 256.0 for x in temp]
        real_data.append((torch.Tensor(temp), 1.0))

        if idx % (len(mnist) / 10) == 0:
            print(int(idx / len(mnist) * 100), "%")

    # write to file
    with open('pixels.pkl', 'wb') as f:
        pickle.dump(real_data, f)

# function that returns random noise (0 ~ 1)
def Z():
    return torch.rand(image_h * image_w)

# number of data
N = len(real_data)
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
    nn.Sigmoid(),
)

# loss function (Mean Squared Error)
loss_fn = nn.MSELoss(size_average=False)

# SGD optimizer
optimizer = optim.SGD(D.parameters(), lr=1e-4, momentum=0.5)

# number of iterations
iteration = 100

for _ in range(iteration):

    # create fake data
    fake_data = []
    for _ in range(N):
        fake_data.append((G(Z()), 0.0))

    # combine real data and fake data
    combined_data = real_data + fake_data

    #print(real_data[0])
    #print(fake_data[-1])

    # split data
    train_loader = torch.utils.data.DataLoader(combined_data, batch_size=50, shuffle=True, num_workers=2)
    print(train_loader[0])

    optimizer.zero_grad()

    # number of k (train D for k times in a row)
    k = 10
    for d_idx in range(k):
        for idx, data in enumerate(train_loader):

            train_data, train_label = data

            optimizer.zero_grad()

            print(train_label)
            # compute predicted Y
            D_pred = D(train_data)

            # compute loss
            loss = loss_fn(D_pred, train_label)
            print(d_idx, loss.item())

            loss.backward()

            optimizer.step()

    #train G
    pass
