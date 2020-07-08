####################
# Importation of packages.

import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
from os import listdir, path
from scipy.fftpack import fft, dct, idct

# Timer
start_time = time.time()

def NetworkInitialization(inSize, resSize, rho, dens, shift, topology, control):
    # Win = np.zeros(shape=(numParticles,resSize, 1 + inSize))
    Win = np.random.rand(resSize, 1 + inSize) - shift
    if control == 0:
        if topology == 0:  # classic ESN, full connected
            W = np.random.rand(resSize, resSize) - shiftr

        elif topology == 1:  # classic ESN with dens density
            print("Topo 1")
            W = np.random.rand(resSize, resSize) - shift
            for i in range(resSize):
                for j in range(resSize):
                    if np.random.rand() >= dens:
                        W[i, j] = 0

        if topology == 2:  # Delay Line Reservoir (DLR)
            print("Topo 2")
            W = np.zeros(shape=(resSize, resSize))
            for i in np.arange(resSize - 1):
                W[i, i + 1] = np.random.rand()
                W[i + 1, i] = np.random.rand()

        elif topology == 3:  # Simple Cycle Reservoir (SCR)
            print("Topo 3")
            W = np.zeros(shape=(resSize, resSize))
            for i in np.arange(resSize - 1):
                W[i, i + 1] = np.random.rand()
            W[resSize - 1, 0] = np.random.rand()
    else:
        if topology == 0:  # classic ESN, full connected
            W = np.random.rand(resSize, resSize) - shift
            rhoW = max(abs(np.linalg.eig(W)[0]))
            W *= rho / rhoW

        elif topology == 1:  # classic ESN with dens density
            print("Topo 1")
            W = np.random.rand(resSize, resSize) - shift
            for i in range(resSize):
                for j in range(resSize):
                    if np.random.rand() >= dens:
                         W[i, j] = 0
            rhoW = max(abs(np.linalg.eig(W)[0]))
            W *= rho / rhoW

        if topology == 2:  # Delay Line Reservoir (DLR)
            print("Topo 2")
            W = np.zeros(shape=(resSize, resSize))
            for i in np.arange(resSize - 1):
                W[i, i + 1] = np.random.rand()
                W[i + 1, i] = np.random.rand()
            rhoW = max(abs(np.linalg.eig(W)[0]))
            W *= rho / rhoW

        elif topology == 3:  # Simple Cycle Reservoir (SCR)
            print("Topo 3")
            W = np.zeros(shape=(resSize, resSize))
            for i in np.arange(resSize - 1):
                W[i, i + 1] = np.random.rand()
            W[resSize - 1, 0] = np.random.rand()
            # rhoW = max(abs(np.linalg.eig(W)[0]))
            # W *= rho / rhoW

    res = []
    res.append(Win)
    res.append(W)
    return res


# Output: Win, W, Wout, TestingPred
# ESN type is: 0, 1, 2
def ESNmodel(ESNparam, ConfigLearning, data):
    #  Reservoir metaparameters
    resSize = ESNparam[0]  # reservoir size
    rho = ESNparam[1]  # spectral radius, rho=-1 then no rho control
    dens = ESNparam[2]
    leaky = ESNparam[3]  # leaking rate
    reg = ESNparam[4]
    shift = ESNparam[5]  # the weights are scaled in [-shift,shift]
    # 0 (ESN full connected), 1 (ESN sparse), 2 (DLR), 3 (SCR)
    topology = ESNparam[6]
    # 0 (ESN with neuron leaky), 1 ESN (with reservoir leaky), 2 (ESN with noise)
    ESNtype = ESNparam[7]
    noise = np.exp(-15)
    #################
    # Config learning
    #################
    trainInit = ConfigLearning[0]  # from what point starts the training
    trainEnd = ConfigLearning[1]
    testLen = ConfigLearning[2]
    learningMode = ConfigLearning[3]
    timeAhead = ConfigLearning[4]
    inSize = outSize = 1
    initialTest = trainEnd
    # Target data
    Yt = data[None, timeAhead:(data.shape[0])]  # shift one time-step.

    #################
    # Network initialization
    #################
    control = 1
    weights = NetworkInitialization(
        inSize, resSize, rho, dens, shift, topology, control)
    Win = weights[0]
    W = weights[1]
    # x state, evaluate each particle in a small batch.
    x = np.random.rand(resSize, 1)
    X = np.zeros(shape=(1 + inSize + resSize, trainEnd))
    # Y = np.zeros(shape=(1,testLen))
    # print("p: ",p, " aux: ",aux)
    for t in list(range(trainEnd)):  # for each sample
        u = data[t]  # input pattern
        # compute state.
        if ESNtype == 0:
            x = (1 - leaky) * x + leaky * \
                np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        elif ESNtype == 1:
            x = (1 - leaky) * x + np.tanh(np.dot(Win,
                np.vstack((1, u))) + leaky * np.dot(W, x))
        else:
            aux = leaky * np.dot(W, x)[:, 0]+noise*np.random.rand(1, resSize)
            x = (1 - leaky) * x + \
                np.tanh(np.dot(Win, np.vstack((1, u)))+aux.transpose())

        # store state in large matrix of states
        X[:, t] = np.vstack((1, u, x))[:, 0]

     # Compute Wout
    Xrange = X.transpose()
    Wout = np.dot(np.dot(Yt[0, trainInit:trainEnd], Xrange[trainInit:trainEnd, :]), np.linalg.inv(np.dot( X[:, trainInit:trainEnd], Xrange[trainInit:trainEnd, :]) + reg * np.identity(1 + inSize + resSize)))
     # to test over data in range test
     # run the trained ESN in a generative mode. no need to initialize here,
     # because x is initialized with training data and we continue from there.

    Ytest = data[None, trainEnd:(trainEnd + testLen)]
    pred = np.zeros(shape=(1, testLen))
    u = data[initialTest]
    for t in list(range(testLen)):
        # compute state.
        if ESNtype == 0:
            x = (1 - leaky) * x + leaky * \
                np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        elif ESNtype == 1:
            x = (1 - leaky) * x + np.tanh(np.dot(Win,
                np.vstack((1, u))) + leaky * np.dot(W, x))
        else:
            aux = leaky * np.dot(W, x)[:, 0]+noise*np.random.rand(1, resSize)
            x = (1 - leaky) * x + \
                np.tanh(np.dot(Win, np.vstack((1, u)))+aux.transpose())

        y = np.dot(Wout, np.vstack((1, u, x)))
        pred[0, t] = y
    if learningMode == 0:  # generative mode
        # generative mode:
        u = y
        # this would be a predictive mode:
    else:
        u = data[initialTest + t]

    res = [Win, W, Wout, pred]
    return res


#####
my_dir = dir()  # Define a variable which holds all variables before you start programming
#####
#################
# Load the data
#################
# import LoadData
#################
np.random.seed(2)  # seed random number generator
#################
#  Reservoir metaparameters
#################
ESNparam = [100, 1.2, 0.9, 0.3, 1e-8, 0.5, 0,0]  # [resSize, rho, dens, leaky, regularization parameter, weightshift, topology]
#################
# Config learning
#################
trainInit= 20  # from what point starts the training
trainEnd= 300
testLen= 6
inSize= outSize = 1
initialTest= trainEnd
learningMode= 0  # generative mode
timeAhead= 1
ConfigLearning= [trainInit, trainEnd, testLen, learningMode, timeAhead]
K= 50

# # pred = np.zeros(shape=(K, testLen))
# predESNRhoO5 = np.zeros(shape=(3, testLen))
# predESNRho7 = np.zeros(shape=(3, testLen))
# predESNRho12= np.zeros(shape=(30, 3, testLen))
# ESNparam[1]= 0.05 # spectral radius
# for i in range(3):
#     print("First for ", i)
#     ESNparam[7] = i # ESN type
#     resESN= ESNmodel(ESNparam, ConfigLearning, data)  # return Win,W,Wout, pred
#     predESNRhoO5[i, : ] = resESN[3]
# #####################

# Minh matrix contains the file that you sent me by
root_dir = path.dirname(path.dirname(path.abspath(__file__)))
curr_path = path.dirname(__file__)

pca_matrix=np.load(path.join(root_dir, 'data', 'pca.npy'))
pca_elements=pca_matrix.shape[0] # your file with the time series.
predOz=np.zeros(shape=(pca_elements,testLen)) # predOz will have the predicted values
for i in range(pca_elements):
    data=pca_matrix[i,:]
    resESN = ESNmodel(ESNparam, ConfigLearning, data)
    predOz[i, :] = resESN[3]
    print('Iteration',i)

predOz = np.asarray(predOz)
print(predOz)

np.save(path.join(curr_path, 'pca_pred'), predOz)

print('Time: %s seconds.' % (time.time() - start_time))