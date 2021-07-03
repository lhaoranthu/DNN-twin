import os
os.environ['PYTHONHASHSEED']=str(1)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import ntpath
import random
from numpy import linalg as LA
import sys
import scipy.io as sio
from numpy.linalg import inv
# import shap
##
from sympy import sin, cos, pi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression

def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(0)
reset_random_seeds()


# data generation power system
def Data_gen_power(Num_data = 1500, Num_bus = 4):
    Xtrain = np.zeros((0, Num_data))
    Topo = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
    for n in range(Num_bus):
        Xtrain = np.vstack((Xtrain, np.random.uniform(0.95, 1.05, Num_data)))
    Ytrain = np.zeros((0, Num_data))
    for n in range(Num_bus):
        neigh = Topo[n]
        p = 0
        for m in neigh:
            x1 = Xtrain[n, :]
            x2 = Xtrain[m, :]
            p = p + 5 * (x1 - x2) * (x1 - x2)
        Ytrain = np.vstack((Ytrain, p))
    Xtrain = Xtrain.T
    Ytrain = Ytrain.T
    # do normalization
    Xtrain = -1 + ((Xtrain - np.min(Xtrain, 0)) * 2) / (np.max(Xtrain, 0) - np.min(Xtrain, 0))
    Ytrain = -1 + ((Ytrain - np.min(Ytrain, 0)) * 2) / (np.max(Ytrain, 0) - np.min(Ytrain, 0))
    # reobtain coefficients for each function
    coeff = np.zeros((0, 9)) #c , x1, x1^2, x2, x2^2, x1x2, x4, x4^2, x1x4
    score = []
    for n in range(Num_bus):
        xt = Xtrain[:, n]
        Xp = np.vstack((xt ** 0, xt ** 1, xt ** 2)) # c, x1, x2^2
        neigh = Topo[n]
        for k1 in neigh:
            xt2 = Xtrain[:, k1]
            Xp = np.vstack((Xp, xt2, xt2 ** 2, xt * xt2)) # x2, x2^2, x1x2
        Xp = Xp.T
        # conduct regression to find the coefficients
        reg = LinearRegression(fit_intercept=False).fit(Xp, Ytrain[:, n])
        score.append(reg.score(Xp, Ytrain[:, n]))
        coeff = np.vstack((coeff, reg.coef_))
    return Xtrain, Ytrain, coeff, score



# data generation synthetic
def Data_gen_sythe(Num_data = 1000, D = 3):
    x1 = np.random.uniform(2, 5, Num_data)
    x2 = np.random.uniform(2, 5, Num_data)
    x3 = np.random.uniform(2, 5, Num_data)

    z = np.vstack((x1**2, np.cos(x2), np.sin(x3), x1**2 * np.cos(x2)))
    Xtrain = np.vstack((x1, x2, x3))
    Xtrain = Xtrain.T
    ytrain = z.T
    # do normalization
    Xtrain = -1 + ((Xtrain - np.min(Xtrain, 0)) * 2) / (np.max(Xtrain, 0) - np.min(Xtrain, 0))
    ytrain = -1 + ((ytrain - np.min(ytrain, 0)) * 2) / (np.max(ytrain, 0) - np.min(ytrain, 0))

    # Xtrain = 0 + ((Xtrain - np.min(Xtrain, 0)) * 1) / (np.max(Xtrain, 0) - np.min(Xtrain, 0))
    # ytrain = 0 + ((ytrain - np.min(ytrain, 0)) * 1) / (np.max(ytrain, 0) - np.min(ytrain, 0))

    # Xtrain = torch.tensor(Xtrain)
    # ytrain = torch.tensor(ytrain)
    return Xtrain, ytrain

def Gen_Fourier_base(x, N=10, L=1):
    CosB = np.zeros((0, len(x)))
    SinB = np.zeros((0, len(x)))
    for n in range(N):
        cosb = np.cos((n * np.pi * x)/L)
        CosB = np.vstack((CosB, cosb))
        sinb = np.sin((n * np.pi * x)/L)
        SinB = np.vstack((SinB, sinb))
    return CosB.T, SinB.T

def Base_eval(Xtrain, h2, h3, Gp, W4, PB):
    # Xtrain: training dataset
    # h2: unit base set
    # Gp: gate for activation of h2
    # PB: ground-truth physical base
    eplison = 0.01
    eplison2 = 0.05
    # find the activated learned base
    gmean = np.mean(Gp, 0)
    h2_act_ind = np.where(gmean > 0.8)
    h2_act_ind = h2_act_ind[0]
    h3_act_ind = np.where(abs(W4) > 0)
    h3_act_ind = h3_act_ind[1]
    h3_act_ind = np.unique(h3_act_ind)
    h32h2_act_ind = np.zeros(0)
    H = int(h2.shape[1]/Xtrain.shape[1])
    for ind in h3_act_ind:
        h32h2_act_ind = np.hstack((h32h2_act_ind, np.array(range(ind * h2.shape[1], (ind + 1) * h2.shape[1]))))
    h2_act_ind = np.intersect1d(h2_act_ind, h32h2_act_ind)
    act_ind = np.zeros(int(0))
    for ind in h2_act_ind:
        ind2 = int(ind % h2.shape[1])
        act_ind = np.hstack((act_ind, ind2))
    act_ind = np.unique(act_ind)
    act_ind = act_ind.astype(int)
    # compare the learned base and the ground-truth base
    h2_PB_neuron = []
    h2_coeff_list = []
    degree_record = np.zeros(0)
    num_act_base = 0
    for ind in range(h2.shape[1]):
        if ind in act_ind:
            ht = h2[:, int(ind)]
            ind1 = int(ind % H)
            ind2 = int(ind // H)
            xt = Xtrain[:, ind2]
            count = 0
            for key in PB.keys():
                count = count + len(PB[key])
                for n in PB[key]:
                    if key == 'poly':  # [1, xt^1, xt^2, ...]
                        xt2 = np.zeros((0, len(xt)))
                        for nn in range(n + 1):
                            xt2 = np.vstack((xt2, xt ** nn))
                        xt2 = xt2.T
                    elif key == 'sinu':
                        xt2 = np.sin(xt * n * np.pi)
                    # xt2 = xt2.reshape((len(xt2), 1))
                    reg = LinearRegression(fit_intercept=False).fit(xt2, ht)
                    ht2 = reg.predict(xt2)
                    loss = np.mean(np.square(ht - ht2))
                    score = reg.score(xt2, ht)
                    if loss <= eplison and score > 0.9:
                        num_act_base = num_act_base + 1
                        h2_PB_neuron.append('h2: hidden neuron ' + str(int(ind)) + ', ' + key + ' degree ' + str(n))
                        h2_coeff_list.append([ind2, n, reg.coef_])
                        degree_record = np.append(degree_record, n)
                        break
                else:
                    continue
                break
        else:
            h2_PB_neuron.append('h2: hidden neuron ' + str(int(ind)) + ', ' + 'not activated')
            h2_coeff_list.append([])
    if num_act_base < len(act_ind): # if not all the activated h2 neurons are well-captured by the physical base
        J2 = 1 + count
    else:
        J2 = count
    degree_record = np.unique(degree_record)
    J1 = len(degree_record)
    J = J1/J2 # Jaccard index

    # symbolic multiplication to
    h3_PB_neuron = []
    h3_coeff_list = []
    for k in range(h3.shape[1]):
        yt = h3[:, k]
        PB_dic_h3_ele = {}
        if k in h3_act_ind:
            # find the index of the corresponding h2 elements
            h2ele_ind = np.array(range(k * h2.shape[1], (k + 1) * h2.shape[1]))
            h2ele_ind = np.intersect1d(h2_act_ind, h2ele_ind)
            h2ele_ind = h2ele_ind % 12
            # obtain the physical base for element k in h3
            for ind in h2ele_ind:
                ind = int(ind)
                h2ele = h2_coeff_list[ind]
                ind2 = h2ele[0]
                degree = h2ele[1]
                if ind2 not in PB_dic_h3_ele.keys():
                    PB_dic_h3_ele[ind2] = degree
                else:
                    if degree > PB_dic_h3_ele[ind2]:
                        PB_dic_h3_ele[ind2] = degree
            # rearrange the key to be in order
            h3_ele_key = np.array(list(PB_dic_h3_ele.keys()))
            h3_ele_key = np.sort(h3_ele_key)
            xt = np.ones((1, Xtrain.shape[0]))
            fn = 'h3: hidden neuron ' + str(int(k))
            for key in h3_ele_key:   # NOTE: Here we should do x1 * x2 !!! haoran
                degree = PB_dic_h3_ele[key]
                fn = fn + ', input x' + str(key) + ', degree ' + str(degree)
                for n in range(1, degree + 1):
                    xt = np.vstack((xt, Xtrain[:, key] ** n))
            xt = xt.T
            reg = LinearRegression(fit_intercept=False).fit(xt, yt)
            yt2 = reg.predict(xt)
            loss = np.mean(np.square(yt - yt2))
            score = reg.score(xt, yt)
            if loss <= eplison2 and score > 0.9:
                h3_PB_neuron.append(fn)
                h3_coeff_list.append([k, reg.coef_])
            else:
                h3_PB_neuron.append('h3: hidden neuron ' + str(int(k)) + ', ' + 'not well-formalized')
                h3_coeff_list.append([])



        else:
            h3_PB_neuron.append('h3: hidden neuron ' + str(int(k)) + ', ' + 'not activated')
            h3_coeff_list.append([])
    return J, h2_PB_neuron






class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.data)

# four activation
def four(input):
    '''
    Applies the Fourier base activation functions element-wise:
    '''
    output = torch.empty(0)
    cosN = int(input.size()[1]/2)
    for n in range(cosN): # first half using cos activation
        ele = input[:, n]
        cosb = torch.cos((n * np.pi * ele))
        cosb = cosb.view(cosb.size()[0], 1)
        output = torch.cat((output, cosb), 1)
    for n in range(cosN): # second half using sin activation
        ele = input[:, n + cosN]
        sinb = torch.sin((n * np.pi * ele))
        sinb = sinb.view(sinb.size()[0], 1)
        output = torch.cat((output, sinb), 1)
    output = output.float()
    return output


# four-polynomial activation
def four_poly(input):
    '''
    Applies the Fourier base activation functions element-wise:
    '''
    output = torch.empty(0)
    cosN = int(input.size()[1]/3)
    for n in range(cosN): # first 1/3 using cos activation
        ele = input[:, n]
        cosb = torch.cos((n * np.pi * ele))
        cosb = cosb.view(cosb.size()[0], 1)
        output = torch.cat((output, cosb), 1)
    for n in range(cosN): # second 1/3 using sin activation
        ele = input[:, n + cosN]
        sinb = torch.sin((n * np.pi * ele))
        sinb = sinb.view(sinb.size()[0], 1)
        output = torch.cat((output, sinb), 1)
    for n in range(cosN): # third 1/3 using polynomial activation
        ele = input[:, n + 2 * cosN]
        polb = ele ** n
        polb = polb.view(polb.size()[0], 1)
        output = torch.cat((output, polb), 1)
    output = output.float()
    return output


def sig_log(input, x0):
    '''
    Sig: sigmoid*6 such that the input range is 0-6
    Log: logistic activation function: f= 1/(1+exp(-k(x-x0))), x0 =5, means 83% to be 0
    '''
    k = 10
    output = 6 * torch.sigmoid(input)
    output = 1/(1+torch.exp(-k*(output - x0)))
    output = output.float()
    return output

def logis(input, x0):
    '''
    Sig: sigmoid*6 such that the input range is 0-6
    Log: logistic activation function: f= 1/(1+exp(-k(x-x0))), x0 =5, means 83% to be 0
    '''
    k = 10
    output = 1/(1+torch.exp(-k*(input - x0)))
    output = output.float()
    return output

# class binstep(torch.autograd.Function):
#     '''
#     Implementation of Binary step activation function.
#     If mean(input) > 0.9, output =1. Otherwise, output = 0. We enourage 0s as sparsity.
#     '''
#     #both forward and backward are @staticmethods
#     @staticmethod
#     def forward(ctx, input):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         ctx.save_for_backward(input) # save input for backward pass
#
#         # get lists of odd and even indices
#         one_indices = torch.where(torch.mean(input, 0) > 0.9)[0]
#         zero_indices = torch.where(torch.mean(input, 0) <= 0.9)[0]
#
#         # clone the input tensor
#         output = input.clone()
#
#         # apply ReLU to elements where i mod 2 == 0
#         output[:, zero_indices] = 0
#         output[:, one_indices] = 1
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         grad_input = None # set output to None
#
#         input, = ctx.saved_tensors # restore input from context
#
#         # check that input requires grad
#         # if not requires grad we will return None to speed up computation
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.clone()
#
#             # get lists of odd and even indices
#             one_indices = torch.where(torch.mean(input, 0) > 0.9)[0]
#             zero_indices = torch.where(torch.mean(input, 0) <= 0.9)[0]
#
#             # set grad_input for even_indices
#             grad_input[one_indices] = (input[even_indices] >= 0).float() * grad_input[even_indices]
#
#             # set grad_input for odd_indices
#             grad_input[odd_indices] = (input[odd_indices] < 0).float() * grad_input[odd_indices]
#
#         return grad_input


# # create a class wrapper from PyTorch nn.Module, so
# # the function now can be easily used in models
# class Four(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, input):

class PNet(nn.Module):
    def __init__(self, Nin, Nout, N, K, B, H, x0):
        super(PNet, self).__init__()
        self.Nin = Nin
        self.Nout = Nout
        self.N = N
        self.K = K
        self.B = B
        self.x0 = x0
        self.H = H
        self.fc1 = nn.Linear(1, self.B*self.N, bias=False) # Fourier activation, cos/sin
        self.fc2 = nn.ModuleList([nn.Linear(self.B*self.N, 1) for i in range(self.H * self.Nin)]) # the second layer has F * Nin flows
        self.fc4 = nn.Linear(self.K, self.Nout) # the last layer for physical NN, l21 norm for sparsity,
        # gate neural network
        self.g1 = nn.ModuleList([nn.Linear(self.H * self.Nin, 10) for i in range(self.K)])
        self.g2 = nn.ModuleList([nn.Linear(10, 5) for i in range(self.K)])
        self.g3 = nn.ModuleList([nn.Linear(5, self.H * self.Nin) for i in range(self.K)])
        self.g4 = nn.ModuleList([nn.LayerNorm(self.H * self.Nin, elementwise_affine=False, eps=1e-3) for i in range(self.K)])
        # self.g5 = nn.ModuleList([nn.Linear(self.Nin, self.Nin) for i in range(self.K)])
    def forward(self, x):
        h2 = torch.empty(0)
        # Fourier activation-formulation layers
        for num in range(self.Nin):
            ele = x[:, num].view(x[:, num].size()[0], 1)
            if self.B == 3:
                h1 = four_poly(self.fc1(ele))
            elif self.B == 2:
                h1 = four(self.fc1(ele))
            for f in range(self.H):
                h2 = torch.cat((h2, self.fc2[num * self.H + f](h1)), 1)
        # gate design G_p
        h3 = torch.empty(0)
        Gp = torch.empty(0)
        for k in range(self.K):
            g = F.relu(self.g1[k](h2))
            g = F.relu(self.g2[k](g))
            # g = F.sigmoid(self.g3[k](g))
            g = self.g3[k](g)
            g = sig_log(6 * self.g4[k](g), self.x0) # 6 * Layer norm
            Gp = torch.cat((Gp, g), 1)


            # element-wise product for gate and h2, generate K output
            # ele = g[:, k*self.Nin:(k+1)*self.Nin] * h2 # hadamard product
            ele = g * h2 # hadamard product
            ele = ele + 1
            ele = torch.prod(ele, 1) # hidden neuron product
            ele = ele.view(ele.size()[0], 1)
            h3 = torch.cat((h3, ele), 1)
        h4 = self.fc4(h3)
        return h4, h3, h2, Gp

class VNet(nn.Module):
    def __init__(self, Nout, x1, M):
        super(VNet, self).__init__()
        self.Nout = Nout
        self.x1 = x1
        self.M = M
        self.fcM = nn.ModuleList([nn.Linear(self.Nout, self.Nout) for i in range(self.M)]) # Maximum M layers
        # gate neural network
        self.g1 = nn.Linear(self.Nout, 20)
        self.g2 = nn.Linear(20, 10)
        self.g3 = nn.Linear(10, self.M)
        # self.g5 = nn.ModuleList([nn.Linear(self.Nin, self.Nin) for i in range(self.K)])
    def forward(self, x):
        # form the gate
        g = F.relu(self.g1(x))
        g = F.relu(self.g2(g))
        Gv = logis(self.g3(g), self.x1)
        h = x
        output = x
        for m in range(self.M):
            h = F.relu(self.fcM[m](h))
            ele = Gv[:, m].view(Gv[:, m].size()[0], 1)
            h = ele * h
            output = output + h
        return output, Gv

# obtain data
Xtrain, ytrain, coeff, score = Data_gen_power()
# add noise
# missing

Xtrain2 = torch.tensor(Xtrain)
ytrain2 = torch.tensor(ytrain)
dataset = MyDataset(Xtrain, ytrain)
loader = DataLoader(
    dataset,
    batch_size=20,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# initial parameters
Nin, Nout, N, K, B, H, x0,  l1_weight, l1_threshold, lr0, Num_epoch = \
    Xtrain.shape[1], ytrain.shape[1], 6, 8, 3, 3, 4, 0.02, 0.02, 5e-4, 100  # x0 = 4
# input dim
# output dim
# number of frequency and polynomial degree
# number of multipliers
# number of base: 2 for sin/cos, 3 for sin/cos/poly
# number of Flows for each input element
# center point of logistic function, control the probability of being 0 for the gate
# l21 norm penalty
# number of epoches

x1, M, gamma = 0, 10, 5
# center for the logistic
# maxmum number of virtual layers
# the weight of maximized physical knowledge

# initial model
pnet = PNet(Nin, Nout, N, K, B, H, x0)
vnet = VNet(Nout,  x1, M)

# # test virtual DNN
# outputs, h3, h2, Gp = pnet(Xtrain2.float())
# outputs, Gv = vnet(outputs)

# training setting for the 1st layer, Fourier layer
# net.fc1.train(False) # Fourier layer is non-trainable
pnet.fc1.weight.requires_grad = False # Fourier layer is non-trainable
four_weight = np.ones((B * N, 1))
four_weight = torch.tensor(four_weight)
pnet.fc1.weight = torch.nn.Parameter(four_weight)


# set fc4 list to zeros
# fc4weight = np.zeros(pnet.fc4.weight.size())
# fc4weight = torch.tensor(fc4weight).float()
# pnet.fc4.weight = torch.nn.Parameter(fc4weight)


# # set fc2 list to zeros
# for k in range(K):
#     fc2weight = np.random.normal(-1, 1, pnet.fc2[k].weight.size())
#     fc2weight = torch.tensor(fc2weight).float()
#     pnet.fc2[k].weight = torch.nn.Parameter(fc2weight)
#     pnet.fc2[k].bias = torch.nn.Parameter(torch.tensor(0.0).float())


# set opt
criterion = nn.MSELoss()
optimizer = optim.Adam([{'params': pnet.parameters()}, {'params': vnet.parameters()}], lr=lr0, weight_decay=0.05)  # lr = 5e-2, use weight decay to avoid weight boosting


for epoch in range(Num_epoch):  # loop over the dataset multiple times
    for batch_idx, (data, label) in enumerate(loader):
        running_loss = 0.0

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputp, h3, h2, Gp = pnet(data.double())
        outputs, Gv = vnet(outputp)
        loss = criterion(outputs.float(), label.float()) + gamma * criterion(outputp.float(), label.float())
        # formalize l21 loss for W4 (Nout, K), ||W4||^2_{2,1} = (sum ||w4||_2)^2, w4 is the column of W4

        l21loss = torch.sum(torch.abs(pnet.fc4.weight))  # l1 norm

        # l21 norm
        # l21loss = 0
        # for num in range(K):
        #     #pnet.fc4.weight.grad check grad
        #     # l21loss = l21loss + torch.sqrt(sum(pnet.fc4.weight[:, num] ** 2))

        l21loss = l1_weight * l21loss
        loss = loss + l21loss
        pnet.fc1.weight.requires_grad = False  # Fourier layer is non-trainable
        loss.backward()
        optimizer.step()
        # set zeros to ensure sparsity for W4
        ind_zero = torch.where(torch.abs(pnet.fc4.weight) <= l1_threshold)
        pnet.fc4.weight.data[ind_zero] = 0

        # print statistics
        running_loss += loss.item() - l21loss.item()
        print('Epoch ', epoch, 'Batch index ', batch_idx, 'Training error ', running_loss)
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0
print('Finished Training')
print(pnet.fc4.weight)
for k in range(Nin):
    print('Flow ', k, pnet.fc2[k].weight)
# evaluate the physical exactness of training datasets
outputs, h3, h2, Gp = pnet(Xtrain2.double())
h2 = h2.cpu().detach().numpy()
h3 = h3.cpu().detach().numpy()
Gp = Gp.cpu().detach().numpy()
outputs = outputs.cpu().detach().numpy()
ytrain2 = ytrain2.cpu().detach().numpy()
Xtrain2 = Xtrain2.cpu().detach().numpy()
W4 = pnet.fc4.weight
W4 = W4.cpu().detach().numpy()
# obtain the extracted base set and evaluate the Jaccard index
# PB = {'poly': [1, 2], 'sinu': [1]}
PB = {'poly': [1, 2]}

# evaluate the learned physical bases and calculate the Jaccard index
J, Pred_PB_neuron = Base_eval(Xtrain2, h2, h3, Gp, W4, PB)
a = 1

plot1 = plt.figure(1)
for k in range(Xtrain2.shape[1]):
    for h in range(H):
        plt.scatter(Xtrain2[:, k], h2[:, H * k + h], label='h2' + str(h) + '-' + str(k))
    # plt.scatter(Xtrain2[:, k], ytrain2[:, k], label='y-' + str(k))
plt.legend()

# plot1 = plt.figure(2)
# for k in range(h3.shape[1]):
#     plt.scatter(Xtrain2[:, 0], h3[:, k], label='h3-' + str(k))
# plt.legend()

plot1 = plt.figure(2)
for k in range(ytrain2.shape[1]):
    if k < Xtrain2.shape[1]:
        plt.scatter(Xtrain2[:, k], outputs[:, k], label='o-' + str(k))
        plt.scatter(Xtrain2[:, k], ytrain2[:, k], label='y-' + str(k))
    else:
        plt.scatter(Xtrain2[:, 0], outputs[:, k], label='o-' + str(k))
        plt.scatter(Xtrain2[:, 0], ytrain2[:, k], label='y-' + str(k))

plt.legend()
plt.show()
a = 1

