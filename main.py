import pickle
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.datasets as datasets

#######################################################
#                    Preprocessing                    #
#######################################################

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
        temp = [x / 255.0 for x in temp]
        real_data.append(temp)

    # write to file
    print('Save data to local...')
    with open('pixels.pkl', 'wb') as f:
        pickle.dump(real_data, f)


#######################################################
#                Variable declaration                 #
#######################################################

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
iteration = 1000
learning_rate = 0.002

D_optimizer = optim.SGD(D.parameters(), lr=learning_rate)
G_optimizer = optim.SGD(G.parameters(), lr=learning_rate)

#######################################################
#                      Training                       #
#######################################################

if not os.path.exists('output/'):
    os.makedirs('output/')

print('Training Start...')
for idx in range(iteration):

    print('@ Iteration', idx)

    # function that returns random noise (0 ~ 1)
    def Z():
        return torch.rand(image_h * image_w)


    # number of k (train D for k times in a row)
    k = 5
    # number of element in each dataset
    m = 64

    # create fake data
    print('Creating fake data...')
    fake_data = []

    for _ in range(k * m):
        fake_data.append(G(Z()).tolist())

    real_data_loader = DataLoader(real_data, batch_size = m, shuffle = False, num_workers = 0)
    fake_data_loader = DataLoader(fake_data, batch_size = m, shuffle = False, num_workers = 0)

    def preprocess(data):
        # list of tensor to 2-D tensor
        data = torch.stack(data)
        # convert axis
        return data.t()

    #######################################################
    #                      Train D                        #
    #######################################################

    for d_idx in range(k):
        # train with REAL data
        train_data = next(iter(real_data_loader))
        train_data = preprocess(train_data)

        D_optimizer.zero_grad()
        D_pred = D(train_data.float())
        D_loss = loss_fn(D_pred, torch.Tensor([[1.0]]*m))
        print('loss =', D_loss.item())

        D_loss.backward()
        D_optimizer.step()

        # train with FAKE data
        train_data = next(iter(fake_data_loader))
        train_data = preprocess(train_data)

        D_optimizer.zero_grad()
        D_pred = D(train_data.float().detach())
        D_loss = loss_fn(D_pred, torch.Tensor([[0.0]]*m))
        print('loss =', D_loss.item())

        D_loss.backward()
        D_optimizer.step()

    #######################################################
    #                      Train G                        #
    #######################################################

    fake_data_loader = DataLoader(fake_data, batch_size = m, shuffle = False, num_workers = 0)

    # train with FAKE data
    train_data = next(iter(fake_data_loader))
    train_data = preprocess(train_data)

    G_optimizer.zero_grad()
    G_pred = D(train_data.float())
    G_loss = loss_fn(G_pred, torch.Tensor([[1.0]] * m))
    print('loss =', G_loss.item())

    G_loss.backward()
    G_optimizer.step()

    print('@@ Iteration(G)')

    for param in G.parameters():
        print(param)

    #######################################################
    #                     Write Image                     #
    #######################################################

    write_image = True
    if write_image:
        print_img = Image.new('L', (image_w, image_h))
        print_img.putdata(G(Z()).detach().numpy()*255)
        print_img.save('output/G_'+str(idx).zfill(4)+'.png')

    print('\n')
