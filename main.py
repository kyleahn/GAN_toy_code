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
        real_data.append((temp, [1.0]))

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

    # create fake data
    print('Creating fake data...')
    fake_data = []

    for _ in range(N):
        fake_data.append((G(Z()).tolist(), [0.0]))

    # combine real data and fake data
    combined_data = real_data + fake_data

    # number of k (train D for k times in a row)
    k = 10
    # number of element in each dataset
    m = 64
    D_data_loader = DataLoader(combined_data, batch_size = m, shuffle = True, num_workers = 0)

    for d_idx, (train_data, train_label) in enumerate(D_data_loader):
        if k <= d_idx:
            break

        # list of tensor to 2-D tensor
        train_data = torch.stack(train_data)
        train_label = torch.stack(train_label)

        # convert axis
        train_data = torch.from_numpy(np.swapaxes(train_data.numpy(), 0, 1).copy())
        train_label = torch.from_numpy(np.swapaxes(train_label.numpy(), 0, 1).copy())

        print('@@ Iteration(D)', d_idx)

        D_optimizer.zero_grad()

        # compute predicted Y
        print('Computing Y...')
        D_pred = D(train_data.float())

        # compute loss (lower is better for D)
        print('Computing loss...')
        D_loss = loss_fn(D_pred, train_label.float())
        print('loss =', D_loss.item())

        print('Updating gradient...')
        D_loss.backward()

        D_optimizer.step()

        D_optimizer.zero_grad()

    # create fake data
    print('Creating fake data...')
    fake_data = []

    for _ in range(m):
        fake_data.append((G(Z()).tolist(), [1.0]))

    G_data_loader = DataLoader(fake_data, batch_size=m, shuffle=False, num_workers=0)

    (train_data, train_label) = next(iter(G_data_loader))

    # list of tensor to 2-D tensor
    train_data = torch.stack(train_data)
    train_label = torch.stack(train_label)

    # convert axis
    train_data = torch.from_numpy(np.swapaxes(train_data.numpy(), 0, 1).copy())
    train_label = torch.from_numpy(np.swapaxes(train_label.numpy(), 0, 1).copy())

    print('@@ Iteration(G)')

    #for param in G.parameters():
    #    print(param)

    G_optimizer.zero_grad()

    # compute predicted Y
    print('Computing Y...')
    G_pred = D(train_data.float())

    # compute loss (higher is better for G)
    print('Computing loss...')
    G_loss = loss_fn(G_pred, train_label.float())

    print('loss =', G_loss.item())

    print('Updating gradient...')
    G_loss.backward()

    G_optimizer.step()

    G_optimizer.zero_grad()

    print_img = Image.new('L', (image_w, image_h))
    print_img.putdata(G(Z()).detach().numpy()*255)
    print_img.save('output/G_'+str(idx).zfill(4)+'.png')

    print('\n')
