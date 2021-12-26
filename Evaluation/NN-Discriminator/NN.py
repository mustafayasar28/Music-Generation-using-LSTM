# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 10:54:35 2021

@author: muhittin can
"""
import pickle
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim



open_file = open("Train_Label_Set", "rb")
Train_Label_Set = pickle.load(open_file)
open_file.close()
    
open_file = open("Train_Set", "rb")
Train_Set = pickle.load(open_file)
open_file.close()

open_file = open("Test_Label_Set", "rb")
Test_Label_Set = pickle.load(open_file)
open_file.close()

open_file = open("Test_Set", "rb")
Test_Set = pickle.load(open_file)
open_file.close()





class Net(nn.Module):

    def __init__(self,D_in, A,B, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, A).cuda()
        self.linear2 = nn.Linear(A, B).cuda()
        self.linear3 = nn.Linear(B, D_out).cuda()

    def forward(self, x):
        x = self.linear1(x).cuda()
        x = self.linear2(x).clamp(min = 0).cuda()
        y_predicted = self.linear3(x).cuda()
        return y_predicted
    
    
N, D_in, A,B, D_out = 1681, 442, 100, 50, 2



Train_Set = torch.tensor(Train_Set).cuda()
Train_Label_Set = torch.tensor(Train_Label_Set).cuda()
Test_Set = torch.tensor(Test_Set).cuda()
Test_Label_Set = torch.tensor(Test_Label_Set).cuda()
x = Variable(Train_Set,requires_grad = True)
y = Variable(Train_Label_Set,requires_grad = False)

model = Net(D_in, A,B,D_out)


criterion = nn.MSELoss(size_average = False)
model.zero_grad()
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-7)



for t in range(10000):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

optimizer = torch.optim.SGD(model.parameters(),lr = 1e-8)


for t in range(10000):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
def classify(Test):
    result = model(Test)
    Real = (result[0]-1)**2 + (result[1]-0)**2
    Fake = (result[0]-0)**2 + (result[1]-1)**2
    
    if Real<=Fake:
        return "Real"
    else:
        return "Fake"
        
True_Real = 0
True_Fake = 0
for i in range(len(Test_Set)):
    
    if classify(Test_Set[i]) == "Real":
        if Test_Label_Set[i][0] == 1.0 and Test_Label_Set[i][1] == 0.0 :
            True_Real  += 1
            
    if classify(Test_Set[i]) == "Fake":
        if Test_Label_Set[i][0] == 0.0 and Test_Label_Set[i][1] == 1.0 :
            True_Fake += 1
            
print("True_Real:",True_Real)
print("True_Fake:",True_Fake)
    