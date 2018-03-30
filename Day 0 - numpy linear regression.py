import numpy as np                      # for mathematical computation
import matplotlib.pyplot as plt         # for plotting

m = 100     # specify the number of datapoints we want to use

def makedata(numdatapoints):            # make some fake data fitted to an arbitrary polynomial
    x = np.linspace(-10, 10, numdatapoints)     # create a vector of numdatapoints(=m) numbers evenly spaced between -10 and 10
    x = x.reshape(-1, 1)    # make it into a column vector (each datapoint is a row)

    coeffs = [2, -30, 0.5, 5]   # make some polynomial coefficients for the fake data

    y = np.polyval(coeffs, x) + 2 * np.random.rand(numdatapoints, 1)    # evaluate a polynomial with the coefficients we
                                                                        # specified to create the labels for our data
    y = y.reshape(-1, 1) # reshape into column vector (each datapoint is a row)

    return x, y # return column vectors of single inputs and outputs

powers = [2, 3]     # a list of the powers of the inputs which we want to include as features for our model
n = len(powers)     # n = number of features of each training datapoint

def makefeatures(powers):
    features = np.ones((inputs.shape[0], len(powers)))  # initialise a design matrix with the right shape (mxn)
    for i in range(len(powers)):    # for each power in the list powers
        features[:, i] = (inputs**powers[i])[:, 0] # set a column of the design matrix = inputs raised to that power
    print(features)

    return features

def scale_features(features):   # center all features around their mean and divide by their range

    avg = np.mean(features, axis=0) # calculate the mean of each of our features over the whole dataset
    print(avg)
    ranges = np.ptp(features, axis=0)   #  calculate the peak-to-peak values for each feature (ptp of the rows)

    scaled = features - avg     # center all features about their mean
    scaled = np.divide(scaled, ranges)  # divide features by their range so that they are all between 0 and 1
    print(scaled)

    return scaled, avg, ranges  # return our scaled features and the averages and ranges of each feature

# hyperparameters
epochs = 100    # number of times we want to pass through the whole dataset when training
lr = 0.5    # proportion of the gradient by which we want to move our parameters each optimisation step
batch_size=32   # number of data points to train on each optimisation step

class LinearModel():    # create a class as a framework for our linear model

    def __init__(self): # class initialiser
        self.weights = np.random.rand(n, 1)     # randomly initialise a nx1 (in x out) matrix of weights
        print(self.weights)
        self.bias = np.random.rand(1) # initialise a random initial bias (offset)
        print(self.bias)

    def forward(self, x):   # define what happens when data is passed forward through our model
        out = np.matmul(x, self.weights) + self.bias # output is the biased, linear combination of our inputs
        return out

def MSE(h, y):  # mean squared error loss as our model criterion
    diff = h - y    # vector of raw differences between prediction hypothesis and labels
    J = 0.5 * np.matmul(diff.T, diff) # cost = 0.5 diff^2
    return float(J) # return the cost as a float rather than a 1x1 matrix

def SGD(datain, batch_size):    # stochastic gradient descent
    sample = np.random.randint(m, size=batch_size)  # create a list of batch_size=32 random indices to include in the
                                                    # training batch

    batchfeatures, batchlabels = datain[sample], labels[sample] # index the randomly selected batch from the training data

    prediction = mymodel.forward(batchfeatures) # make a prediction for the training batch using the current parameters
    cost = MSE(prediction, batchlabels) # determine how bad the model performs on this batch by calculating the mena square error

    dJdw = np.matmul(batchfeatures.T, (prediction - batchlabels))/m # rate of change of cost wrt weights
    mymodel.weights -= lr * dJdw    # move the weights a small step in the right direction

    dJdb = np.sum(prediction - batchlabels)/m # rate of change of cost wrt bias
    mymodel.bias -= lr * dJdb# move the bias a small step in the right direction

    return cost # return how bad the model is

def train(datain):
    costs=[]    # initialise an empty list of costs for visualisation
    for e in range(epochs): # for however many epochs specialises

        print('b', mymodel.bias[0]) # print bias
        print('w', mymodel.weights) # print weights

        cost = SGD(datain, batch_size)  # calculate cost for a specified batch size
        costs.append(cost) # add cost to list of history
        print('Epoch', e, 'Cost', cost)

        line1.set_ydata(np.matmul(datain, mymodel.weights) + mymodel.bias)  # set the y values of our line to show the
                                                                            # current prediction for that x input
        fig.canvas.draw()   # update ('draw') our latest plot
        ax2.plot(costs)     # replot the new full list of costs


# setting up data--------------------------------------
inputs, labels = makedata(m) # call the above function to make the data
features = makefeatures(powers) # make our features with the powers we specified
scaled, avgs, ranges = scale_features(features)     # scale our features (try and see what happens when you don't scale)
print(avgs, ranges)

# setting up visualisation ----------------------------------------
fig = plt.figure(figsize=(10, 20))  # create a figure

ax1 = fig.add_subplot(121)  # add an axis called ax1 to our figure (height=1, width=2, axis number=1)
ax1.set_xlabel('Input')
ax1.set_ylabel('Output')
ax1.scatter(inputs, labels, s=5)    # scatter plot our labels against our inputs with size=5 markers
ax1.grid()  # add a grid to our axis

ax2 = fig.add_subplot(122)     # add another axis called ax2 to our figure (height=1, width=2, axis number=2)
ax2.set_title('Error vs epoch')
ax2.grid()

line1, = ax1.plot(inputs, inputs)   # THIS DOES NOT MEAN ANYTHING IMPORTANT, WE ARE ONLY CREATING THE LINE OBJECT
# WITH THE CORRECTLY SIZED DATA SO THAT WE CAN CHANGE THE OUTPUT TO
# PLOT OUR HYPOTHESIS FUNCTION

plt.ion()   # turn on interactive mode so that the plots can update whilst the program runs in the background
plt.show()  # show the plots


mymodel = LinearModel() # create an instance of a linear model

datain = scaled # set the data going in to our model as our scaled data (not necessary really)

train(datain)     # finally, call the function to train our model