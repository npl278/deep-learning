import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim
import random
import math
import matplotlib.pyplot as plt 
from torchmetrics import Accuracy

epochs=100
learning_rate=1e-3

random.seed(0)

df=pd.read_csv(r'D:/xzq/course/programming/IGCA/handmade_nn/bank-full.csv',header=0,sep=';')
mapping={"yes":1,"no":0}
df["y"]=df["y"].map(mapping)
df=pd.concat([pd.get_dummies(df.iloc[:,:-1]),df.iloc[:,-1]],axis=1)
for col in df.columns[:-1]:
    mean=df[col].mean()
    var=np.sum(np.square(df[col]-mean))/len(df[col])
    df[col]=(df[col]-mean)/var**0.5
print(df.head())

def train_test_split(df1,ystart,rate):
    random_idx=[i for i in range(len(df1.index))]
    random.shuffle(random_idx)
    train_idx=random_idx[0:int(rate*len(random_idx))]
    test_idx=random_idx[int(rate*len(random_idx)):]
    xtrain=df1.iloc[train_idx,:ystart]
    ytrain=df1.iloc[train_idx,ystart:]
    xtest=df1.iloc[test_idx,:ystart]
    ytest=df1.iloc[test_idx,ystart:]
    return xtrain,ytrain,xtest,ytest


def get_batchs(xtrain,ytrain,batch_size):
    random_idx=[i for i in range(len(xtrain.index))]
    random.shuffle(random_idx)
    xtrain1=xtrain.iloc[random_idx]
    ytrain1=ytrain.iloc[random_idx]
    batchs=[xtrain1[batch_size*i:min(len(random_idx),batch_size*(i+1))] for i in range(math.floor(len(random_idx)/batch_size))]
    ybatchs=[ytrain1[batch_size*i:min(len(random_idx),batch_size*(i+1))] for i in range(math.floor(len(random_idx)/batch_size))]
    return batchs,ybatchs


class Net(nn.Module):
    def __init__(self,nx) -> None:
        super().__init__()
        self.model=nn.Sequential(nn.Linear(nx[0],nx[1]),
                                 nn.ReLU(),
                                 nn.Linear(nx[1],nx[2]),
                                 nn.ReLU(),
                                 nn.Linear(nx[2],nx[3]),
                                 nn.Sigmoid()
                                 )
    def forward(self,x):
        return self.model(x)
    

def getpred(output):
    pred=np.ones(len(output))
    pred.dtype='int64'
    for i in range(len(output)):
        if output[i]>0.5:
            pred[i]=1
        else:
            pred[i]=0
    return torch.from_numpy(pred)    

def train(model,Optimizer,xbatchs,ybatchs):
    model.train()
    lossfunc=nn.BCELoss()
    accuracy=Accuracy(2)
    for i in range(len(xbatchs)):
        x=xbatchs[i,:,:]
        y=ybatchs[i,:,:]
        y=y.view(-1).clone().type(torch.float)
        output=model(x)
        output=output.view(-1)
        pred=getpred(output)
        accuracy.update(pred,y.clone().type(torch.int64))
        loss=lossfunc(output,y)
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
    return accuracy.compute().item()


def validate(model,test_x,test_y):
    lossfunc=nn.BCELoss()
    y_hat=model(test_x)
    loss=lossfunc(y_hat,test_y)
    acc=Accuracy(2)
    pred=getpred(y_hat)
    acc.update(pred,test_y.clone().type(torch.int64).view(-1))
    return loss,acc.compute().item()


xtrain,ytrain,xtest,ytest=train_test_split(df,-1,0.8)
nx=[len(xtrain.columns),20,10,1]
xtest=torch.Tensor(np.array(xtest))
ytest=torch.Tensor(np.array(ytest))
model=Net(nx)
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)


for epoch in range(epochs):
    x_train_batchs,y_train_batchs=get_batchs(xtrain,ytrain,128)
    x_train_batchs=torch.Tensor(np.array(x_train_batchs))
    y_train_batchs=torch.Tensor(np.array(y_train_batchs))
    train_acc=train(model,optimizer,x_train_batchs,y_train_batchs)
    print("epoch {} train_acc:".format(epoch),end="")
    print(train_acc)
    if epoch%10==9:
        print(10*"-"+"after {} epochs:".format(epoch)+10*"-")
        loss,acc=validate(model,xtest,ytest)
        print("vatidate loss:",end="")
        print(loss)
        print("validate accuracy:",end="")
        print(acc)
        print('\n')