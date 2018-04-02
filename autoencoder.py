import torch    # computational library saves us time and effort in building models
from torch.autograd import Variable     # for computational graphs
import matplotlib.pyplot as plt     # for plotting
from torchvision import datasets, transforms    # to get the dataset and then transform it into a tensor
import torch.nn.functional as F # functional stuff like our activation function
import numpy as np

epochs = 100    # for how many runs through of the dataset?
lr = 0.0001       # proportionality constant controlling parameter update size
batch_size = 64 # how large are the training batches?
latent_dim = 3  # how many dimensions does our latent variable have? we can plot it in 3

data = datasets.MNIST(root='data/',     # where to save/look for it
                         train=True,    # this is for training
                         transform=transforms.ToTensor(),   # transform it into a tensor of data
                         download=True) # yes, download it

# make a dataloader to generate us samples for training
training_samples = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=batch_size,   # train on small batches of inputs
                                               shuffle=True)    # make sure to shuffle the data to avoid overfitting

class VAE(torch.nn.Module): # create a class for our variational autoencoder

    def __init__(self):
        super().__init__()

        # encoder
        self.e1 = torch.nn.Linear(784, 20)
        self.e2 = torch.nn.Linear(20, latent_dim)

        # decoder
        self.d1 = torch.nn.Linear(latent_dim, 20)
        self.d2 = torch.nn.Linear(20, 784)

        self.a = F.relu
        self.sigmoid = F.sigmoid

    def encode(self, x): # compress our input into the latent space
        x = x.view(-1, 784)             # unroll
        out1 = self.a(self.e1(x))       # feature extraction
        z = self.a(self.e2(out1))     # batch_size x latent_dim matrix
        print(z.shape)

        return z # return latent vector

    def forward(self, z):   # decode
        out1 = self.d1(z)
        out2 = self.sigmoid(self.d2(out1))
        print(out2.shape)
        return out2#.view(28, 28)

myvae = VAE()
optimiser = torch.optim.Adam(myvae.parameters(), lr=lr)

def loss(x_hat, x):
    reconstruction_loss = F.binary_cross_entropy(x_hat, x.view(-1, 784))
    print(reconstruction_loss)
    return reconstruction_loss

def train():
    myvae.train()   # put in training mode
    costs = []
    for batch_index, (x, _) in enumerate(training_samples):

        x = Variable(x)
        print('x shape', x.shape)

        z = myvae.encode(x)   #
        x_hat = myvae(z)    # (calls myvae.forward) generate an output

        cost = loss(x_hat, x)
        costs.append(cost.data)
        cost.backward()
        optimiser.zero_grad()
        optimiser.step()
        print('batch', batch_index, 'cost', cost.data)

        ax.plot(costs)
        fig.canvas.draw()

fig = plt.figure()
ax = fig.add_subplot(111)

plt.ion()
plt.show()

train()

