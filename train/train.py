####################
# Importation of packages.

import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
import cv2
from os import listdir, path
from scipy.fftpack import fft, dct, idct
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


# Timer
curr_time = time.time()

def NetworkInitialization(inSize, resSize, rho, dens, shift, topology, control):
    # Win = np.zeros(shape=(numParticles,resSize, 1 + inSize))
    Win = np.random.rand(resSize, 1 + inSize) - shift
    if control == 0:
        if topology == 0:  # classic ESN, full connected
            W = np.random.rand(resSize, resSize) - shift

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
    topology = ESNparam[6]  # 0 (ESN full connected), 1 (ESN sparse), 2 (DLR), 3 (SCR)
    ESNtype = ESNparam[7]   # 0 (ESN with neuron leaky), 1 ESN (with reservoir leaky), 2 (ESN with noise)
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
    weights = NetworkInitialization(inSize, resSize, rho, dens, shift, topology, control)
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
            x = (1 - leaky) * x + leaky * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W,x))
        elif ESNtype == 1:
            x = (1 - leaky) * x + np.tanh(np.dot(Win,np.vstack((1, u))) + leaky * np.dot(W, x))
        else:
            aux = leaky * np.dot(W, x)[:, 0] + noise*np.random.rand(1, resSize)
            x = (1 - leaky) * x + np.tanh(np.dot(Win, np.vstack((1, u)))+aux.transpose())

        # store state in large matrix of states
        X[:, t] = np.vstack((1, u, x))[:, 0]

     # Compute Wout
    Xrange = X.transpose()
    Wout = np.dot(np.dot(Yt[0, trainInit:trainEnd], Xrange[trainInit:trainEnd, :]), np.linalg.inv(np.dot( X[:, trainInit:trainEnd], Xrange[trainInit:trainEnd, :]) + reg * np.identity(1 + inSize + resSize)))
     # to test over data in range test
     # run the trained ESN in a generative mode. no need to initialize here,
     # because x is initialized with training data and we continue from there.
    # X_train = x

    Ytest = data[None, trainEnd:(trainEnd + testLen)]
    pred = np.zeros(shape=(1, testLen))
    u = data[initialTest]
    for t in list(range(testLen)):
        # generative mode
        if learningMode == 0:  
            # generative mode:
            if t != 0:
                u = y
        # this would be a predictive mode:
        else:
            u = data[initialTest + t]

        # compute state.
        if ESNtype == 0:
            x = (1 - leaky) * x + leaky * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        elif ESNtype == 1:
            x = (1 - leaky) * x + np.tanh(np.dot(Win,np.vstack((1, u))) + leaky * np.dot(W, x))
        else:
            aux = leaky * np.dot(W, x)[:, 0] + noise*np.random.rand(1, resSize)
            x = (1 - leaky) * x + np.tanh(np.dot(Win, np.vstack((1, u)))+aux.transpose())

        y = np.dot(Wout, np.vstack((1, u, x)))

        pred[0, t] = y

    res = [Win, W, Wout, pred, x]
    return res

# # Custom predict
# def pred_new_img(ESNparam, W_param, x, y):
#     leaky = ESNparam[3]  # leaking rate
#     ESNtype = ESNparam[7]
#     Win = W_param[0]
#     W = W_param[1]
#     Wout = W_param[2]

#     pred = np.zeros(shape=(1, 1))
#     u = y

#     if ESNtype == 0:
#         x = (1 - leaky) * x + leaky * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
#     elif ESNtype == 1:
#         x = (1 - leaky) * x + np.tanh(np.dot(Win,np.vstack((1, u))) + leaky * np.dot(W, x))
#     else:
#         aux = leaky * np.dot(W, x)[:, 0] + noise*np.random.rand(1, resSize)
#         x = (1 - leaky) * x + np.tanh(np.dot(Win, np.vstack((1, u)))+aux.transpose())

#     y = np.dot(Wout, np.vstack((1, u, x)))
#     pred[0, 0] = y
#     res = [pred, x]
#     return res

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
ESNparam = [100, 0.1, 0.9, 0.5, 1e-6, 0.5, 0, 0]  # [resSize, rho, dens, leaky, regularization parameter, weightshift, topology]
# [100, 1, 0.9, 0.5, 1e-8, 0.5, 0, 1]
#################
# Config learning
#################
trainInit = 20  # from what point starts the training
trainEnd = 331
testLen = 10
inSize = outSize = 1
initialTest = trainEnd
learningMode = 1  # generative mode
timeAhead = 1
ConfigLearning = [trainInit, trainEnd, testLen, learningMode, timeAhead]
K = 50

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
curr_dir = path.dirname(__file__)
root_dir = path.dirname(curr_dir)

# North prediction
north_matrix = np.load(path.join(root_dir, 'data', 'reconstructed_gray', 'north', 'gray_north.npy'))
pixel_num = north_matrix.shape[0] # your file with the time series.
pred_gray_north = np.zeros(shape=(pixel_num, testLen)) # pred_gray_north will have the predicted values
# total_resESN = []

# Train image
for i in range(pixel_num):
    data = north_matrix[i,:]
    resESN = ESNmodel(ESNparam, ConfigLearning, data)
    pred_gray_north[i, :] = resESN[3]
    # total_resESN.append(resESN)
    print('Pixel',i,'/',pixel_num)

pred_gray_north = np.asarray(pred_gray_north)
# total_resESN = np.asarray(total_resESN)

# # Predict custom image
# new_img = np.zeros(shape=(pixel_num, 1))
# for i in range(pixel_num):
#     resESN = total_resESN[i]
#     W_param = [resESN[0], resESN[1], resESN[2]]
#     pixel_temp = north_matrix[i,:]
#     new_pixel = pred_new_img(ESNparam, W_param, resESN[4], pixel_temp[305])
#     new_img[i,:] = new_pixel[0]
#     print('Pixel',i,'/',pixel_num)

pred_gray_north = pred_gray_north.transpose().reshape(testLen, 64, 512)
for i in range(pred_gray_north.shape[0]):
    temp = pred_gray_north[i] * 255
    name = str(i) + '.png'
    cv2.imwrite(path.join(curr_dir, 'test_pred_gray', 'north', name), temp)

# new_img = new_img.transpose().reshape(1, 64, 512)
# new_img = new_img * 255
# new_img_name = 'new.png'
# cv2.imwrite(path.join(curr_dir, 'pred_gray', new_img_name), new_img[0])


""" # South prediction
south_matrix = np.load(path.join(root_dir, 'data', 'reconstructed_gray', 'south', 'gray_south.npy'))
pixel_num = south_matrix.shape[0] # your file with the time series.
pred_gray_south = np.zeros(shape=(pixel_num, testLen)) # pred_gray_south will have the predicted values
for i in range(pixel_num):
    data = south_matrix[i,:]
    resESN = ESNmodel(ESNparam, ConfigLearning, data)
    pred_gray_south[i, :] = resESN[3]
    print('Pixel',i,'/',pixel_num)

pred_gray_south = np.asarray(pred_gray_south)


pred_gray_south = pred_gray_south.transpose().reshape(testLen, 64, 512)

for i in range(pred_gray_south.shape[0]):
    temp = pred_gray_south[i] * 255

    name = str(i) + '.png'
    cv2.imwrite(path.join(curr_dir, 'test_pred_gray', 'south', name), temp) """

##############################################
# Calculating RMSE
for i in range(pred_gray_north.shape[0]):
    real = north_matrix[:,(342 - testLen + i)].transpose() * 255
    pred = pred_gray_north[i].flatten() * 255
    testScore_RMSE = mean_squared_error(real, pred, squared = False)

    real = real.reshape(64,512)
    pred = pred.reshape(64,512)
    testScore_SSIM = ssim(real, pred, data_range = (pred.max() - pred.min()))

    print('RMSE',(i+1),'=',testScore_RMSE)
    print('SSIM',(i+1),'=',testScore_SSIM)
    

curr_time = time.time() - curr_time
print('Time: %s seconds.' % (curr_time))