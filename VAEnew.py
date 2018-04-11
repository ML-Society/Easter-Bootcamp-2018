import torch    # computational library saves us time and effort in building models
from torch.autograd import Variable     # for computational graphs
import matplotlib.pyplot as plt     # for plotting
from torchvision import datasets, transforms    # to get the dataset and then transform it into a tensor
import torch.nn.functional as F # functional stuff like our activation function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

epochs = 1    # for how many runs through of the dataset?
lr = 0.02      # proportionality constant controlling parameter update size
batch_size = 128 # how large are the training batches?
latent_dim = 40  # how many dimensions does our latent variable have? we can plot it in 3
beta = 1

train_data = datasets.FashionMNIST(root='fashiondata/',     # where to save/look for it
                         train=True,    # this is for training
                         transform=transforms.ToTensor(),   # transform it into a tensor of data
                         download=True) # yes, download it

test_data = datasets.FashionMNIST(root='fashiondata/',
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
        self.to_mu1 = torch.nn.Linear(784, 256)
        self.to_mu2 = torch.nn.Linear(256, 64)
        self.to_mu3 = torch.nn.Linear(64, latent_dim)

        self.to_logvar1 = torch.nn.Linear(784, 256)
        self.to_logvar2 = torch.nn.Linear(256, 64)
        self.to_logvar3 = torch.nn.Linear(64, latent_dim)

        # decoder
        self.d1 = torch.nn.Linear(latent_dim, 64)
        self.d2 = torch.nn.Linear(64, 256)
        self.d3 = torch.nn.Linear(256, 784)

    def encode(self, x):
        x = x.view(-1, 784)

        mu = F.relu(self.to_mu1(x))
        mu = F.relu(self.to_mu2(mu))
        mu = self.to_mu3(mu)

        logvar = F.relu(self.to_logvar1(x))
        logvar = F.relu(self.to_logvar2(logvar))
        logvar = self.to_logvar3(logvar)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        epsilon = Variable(torch.Tensor(np.random.randn(batch_size, latent_dim)))
        z = mu + epsilon * (0.5*logvar).exp()
        return z

    def decode(self, z):
        x_pred = F.relu(self.d1(z))
        x_pred = F.relu(self.d2(x_pred))
        x_pred = F.sigmoid(self.d3(x_pred))
        #print(x_pred.shape)
        return x_pred

    def forward(self, x):
        mu, logvar = self.encode(x) # the output for the sd does not make sense to be negative, so predict log(var)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z)
        return x_pred, z, mu, logvar

def VAEloss(x_hat, x, mu, logvar):
    reconstruction_loss = F.binary_cross_entropy(x_hat, x.view(-1, 784), size_average=False)
    #print(reconstruction_loss.data[0])
    KL_divergence = - 0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    #print(KL_divergence.data[0])
    return reconstruction_loss + beta * KL_divergence

fig1 = plt.figure(figsize=(10, 20))
ax1 = fig1.add_subplot(121)
ax1.set_title('Costs vs epoch')
#ax1.set_ylim(0, 500)
ax2 = fig1.add_subplot(122)
ax2.set_title('Latent representation')

fig2 = plt.figure(figsize=(10, 20))
ax3 = fig2.add_subplot(121)
ax3.set_title('Input')
ax4 = fig2.add_subplot(122)
ax4.set_title('Reconstruction')

plt.ion()
plt.show()

myVAE = VAE()
optimiser = torch.optim.Adam(myVAE.parameters(), lr=lr)

def trainVAE():
    myVAE.train()   # put in training mode
    costs = []
    for epoch in range(epochs):
        for batch_index, (x, y) in enumerate(training_samples):

            x = Variable(x)
            #print('x shape', x.shape)
            #optimiser.zero_grad()
            x_pred, z, mu, logvar = myVAE(x)  # (calls myvae.forward) generate an output

            cost = VAEloss(x_pred, x, mu, logvar)
            costs.append(cost.data)
            cost.backward()
            optimiser.step()
            optimiser.zero_grad()
            print('Epoch', epoch, 'batch', batch_index, 'cost', cost.data[0])

            z = np.array(z.data) # for plotting
            colordict = {0:'blue', 1:'orange', 2:'green', 3:'red', 4:'purple', 5:'brown', 6:'pink', 7:'gray', 8:'olive', 9:'cyan'}
            colorlist = [colordict[i] for i in y]
            ax2.scatter(z[:, 0], z[:, 1], c=colorlist, s=5)
            ax1.plot(costs, 'b')
            fig1.canvas.draw()

            if batch_index % 20:
                x = x.view(-1, 28, 28)
                x_pred = x_pred.view(-1, 28, 28)

                ax3.imshow(x.data[0])
                ax4.imshow(x_pred.data[0])

                fig2.canvas.draw()

            if batch_index == 30:
                ax2.clear()

            if batch_index == 200:
                #return ax2
                break


trainVAE()
torch.save(myVAE, 'trainedVAE.pt')

fig = plt.figure()
ax = fig.add_subplot(111)
#plt.ioff()

def testVAE():
    for batch_index, (x, y) in enumerate(training_samples):

        x, y = Variable(x), Variable(y)
        print('x shape:', x.shape)

        x_pred, z, mu, logvar = myVAE(x)
        print('x_pred shape:', x_pred.shape)


        plt.show()
        break

testVAE()

myVAE = torch.load('trainedVAE.pt')

newfig = plt.figure()
axis = newfig.add_subplot(111)
plt.show()

def generate():
    z = np.random.randn(1, latent_dim)
    print(z.shape)
    ax2.scatter(z[:, 0], z[:, 1], marker='x', c='k', s=100)
    fig.canvas.draw()

    z = Variable(torch.Tensor(z))
    x_new = myVAE.decode(z).data
    print(x_new)
    axis.imshow(x_new.view(28, 28))
    sleep(2)
    '''
    for i in range(100):
        ax2.scatter(z[:,0], z[:,1], marker='x', c='k', s=50)
        x_new = myVAE.decode(Variable(torch.Tensor(z)))
        print(x_new.data)
        plt.imshow(x_new.data.view(28, 28))
        z += 0.1*np.random.randn(1, latent_dim)
        sleep(2)
    '''
generate()
