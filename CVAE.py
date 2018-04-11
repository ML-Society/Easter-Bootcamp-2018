import torch    # computational library saves us time and effort in building models
from torch.autograd import Variable     # for computational graphs
import matplotlib.pyplot as plt     # for plotting
from torchvision import datasets, transforms    # to get the dataset and then transform it into a tensor
import torch.nn.functional as F # functional stuff like our activation function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

epochs = 1    # for how many runs through of the dataset?
lr = 0.1      # proportionality constant controlling parameter update size
batch_size = 64 # how large are the training batches?
latent_dim = 2  # how many dimensions does our latent variable have? we can plot it in 3

train_data = datasets.MNIST(root='data/',     # where to save/look for it
                         train=True,    # this is for training
                         transform=transforms.ToTensor(),   # transform it into a tensor of data
                         download=True) # yes, download it

test_data = datasets.MNIST(root='data/',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

# make a dataloader to generate us samples for training
training_samples = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,   # train on small batches of inputs
                                               shuffle=True)    # make sure to shuffle the data to avoid overfitting

testing_samples = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size,
                                           shuffle=True)

class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # encoder to means and standard deviations
        self.to_mu1 = torch.nn.Conv2d(1, 5, kernel_size=5, stride=2)  # 64x1x28x28 -> 64x5x12x12
        self.to_mu2 = torch.nn.Conv2d(5, 10, kernel_size=5, stride=2)  # 64x5x24x24 -> 64x10x20x20
        self.to_mu3 = torch.nn.Linear(4000, latent_dim)   # 4000 -> 2

        self.to_logvar1 = torch.nn.Conv2d(1, 5, kernel_size=5, stride=2)
        self.to_logvar2 = torch.nn.Conv2d(5, 10, kernel_size=5, stride=2)
        self.to_logvar3 = torch.nn.Linear(4000, latent_dim) # 4000 -> 2

        # decoder
        self.d1 = torch.nn.Linear(latent_dim, 4000)
        self.d2 = torch.nn.ConvTranspose2d(10, 5, kernel_size=5)
        self.d3 = torch.nn.ConvTranspose2d(5, 1, kernel_size=5)

    def encode(self, x):
        #x = x.view(-1, 784)

        mu = F.relu(self.to_mu1(x))
        print(mu.shape)
        mu = F.relu(self.to_mu2(mu))#.view(-1, 10*20*20)
        print(mu.shape)
        mu = self.to_mu3(mu)
        print(mu.shape)

        logvar = F.relu(self.to_logvar1(x))
        logvar = F.relu(self.to_logvar2(logvar)).view(-1, 10*20*20)
        print(logvar.shape)
        logvar = self.to_logvar3(logvar)
        print(logvar.shape)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        epsilon = Variable(torch.Tensor(np.random.randn(batch_size, latent_dim)))
        z = mu + epsilon * (0.5*logvar).exp()
        return z

    def decode(self, z):
        x_pred = F.relu(self.d1(z)).view(-1, 10, 20, 20)
        x_pred = F.relu(self.d2(x_pred))
        x_pred = F.sigmoid(self.d3(x_pred))
        print(x_pred.shape)
        return x_pred

    def forward(self, x):
        mu, logvar = self.encode(x) # the output for the sd does not make sense to be negative, so predict log(var)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z)
        return x_pred, z, mu, logvar

def VAEloss(x_hat, x, mu, logvar):
    print(mu, logvar)
    reconstruction_loss = F.binary_cross_entropy(x_hat, x, size_average=False)
    print(reconstruction_loss.data[0])
    #print(reconstruction_loss)
    KL_divergence = - 0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    print(KL_divergence.data[0])
    return reconstruction_loss + KL_divergence

fig = plt.figure(figsize=(10, 20))
ax1 = fig.add_subplot(121)
ax1.set_title('Costs vs epoch')
#ax1.set_ylim(0, 5)
ax2 = fig.add_subplot(122)
ax2.set_title('Latent representation')
plt.ion()
plt.show()
#ax2.set_xlim(-3, 3)
#ax2.set_ylim(-3, 3)

myVAE = VAE()
optimiser = torch.optim.Adam(myVAE.parameters(), lr=lr)

def trainVAE():
    myVAE.train()   # put in training mode
    costs = []
    for i in range(epochs):
        for batch_index, (x, y) in enumerate(training_samples):

            x = Variable(x)
            #print('x shape', x.shape)
            optimiser.zero_grad()
            x_pred, z, mu, logvar = myVAE(x)  # (calls myvae.forward) generate an output

            cost = VAEloss(x_pred, x, mu, logvar)
            costs.append(cost.data)
            cost.backward()
            optimiser.step()
            #optimiser.zero_grad()
            print('batch', batch_index, 'cost', cost.data[0])

            z = np.array(z.data) # for plotting
            colordict = {0:'blue', 1:'orange', 2:'green', 3:'red', 4:'purple', 5:'brown', 6:'pink', 7:'gray', 8:'olive', 9:'cyan'}
            colorlist = [colordict[i] for i in y]
            ax2.scatter(z[:, 0], z[:, 1], c=colorlist, s=5)
            ax1.plot(costs, 'b')
            fig.canvas.draw()

            if batch_index == 50:
                ax2.clear()

            if batch_index == 200:
                break

trainVAE()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ioff()

def testVAE():
    for batch_index, (x, y) in enumerate(training_samples):

        x, y = Variable(x), Variable(y)
        print('x shape:', x.shape)

        x_pred, z, mu, logvar = myVAE(x)
        print('x_pred shape:', x_pred.shape)

        n = int(np.min([8, x_pred.shape[0]]))
        print('n', n)
        comparison = torch.cat([x[:n].view(28*n, 28), x_pred[:n].view(28*n, 28)], dim=1)
        print(comparison.shape)
        #comparison= comparison.view(28*n, 28)
        print(comparison.data.shape)
        ax.imshow(comparison.data)
        plt.show()
        break

testVAE()