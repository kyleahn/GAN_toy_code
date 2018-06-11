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
    #raise FileNotFoundError
    # open MNIST data if we can
    print('Finding MNIST data in local...')
    with open('pixels.pkl', 'rb') as f:
        real_data = pickle.load(f)

except FileNotFoundError as e:
    # download MNIST data from web
    print('Download data from web...')
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    # convert image to 1-D pixel data
    print('Convert PIL image to 1-Dim list...')
    real_data = []
    for idx, (im, label) in enumerate(mnist):
        if idx % (len(mnist) / 10) == 0:
            print(int(idx / len(mnist) * 100), "%")
        if label != 3:
            continue
        temp = list(im.getdata())
        w, h = im.size
        temp = [temp[i * w:(i + 1) * w] for i in range(h)]
        temp = [x for sublist in temp for x in sublist]
        temp = [x / 256.0 for x in temp]
        real_data.append((temp, [1.0]))

    # write to file
    print('Save data to local...')
    with open('pixels.pkl', 'wb') as f:
        pickle.dump(real_data, f)

# number of data
N = len(real_data)
# number of Hidden dimension
H = 1000

# define Generator
print('Declare G...')
G = nn.Sequential(
    nn.Linear(image_h * image_w, H),
    nn.ReLU(),
    nn.Linear(H, image_h * image_w),
    nn.Sigmoid(),
)

# define Discriminator
print('Declare D...')
D = nn.Sequential(
    nn.Linear(image_h * image_w, H),
    nn.ReLU(),
    nn.Linear(H, 1),
    nn.Sigmoid(),
)

# loss function (Mean Squared Error)
print('Declare Loss function...')

# declare loss function (V)
loss_fn = nn.BCELoss()

# number of iterations
iteration = 100

print('Training Start...')
for idx in range(iteration):

    if idx < iteration/2:
        learning_rate = 0.01
    elif idx < iteration/4:
        learning_rate = 0.005
    else:
        learning_rate = 0.001

    print('@ Iteration', idx)

    # function that returns random noise (0 ~ 1)
    def Z():
        return torch.rand(image_h * image_w)

    # create fake data
    print('Creating fake data...')
    fake_data = []

    for _ in range(N):
        fake_data.append((G(Z()).tolist(), [0.0]))

    # combine real data and fake data
    combined_data = real_data + fake_data

    # number of k (train D for k times in a row)
    k = 10
    for d_idx in range(k):
        print('@@ Iteration(D)', d_idx)
        train_data = torch.Tensor([x[0] for x in combined_data])
        train_label = torch.Tensor([x[1] for x in combined_data])

        # compute predicted Y
        print('Computing Y...')
        D_pred = D(train_data)

        # compute loss
        print('Computing loss...')
        loss = loss_fn(D_pred, train_label)
        print('loss =', loss.item())

        D.zero_grad()

        print('Updating gradient...')
        loss.backward()

        with torch.no_grad():
            for param in D.parameters():
                param -= learning_rate * param.grad
        print('\n')
    print('@@ Train G')
