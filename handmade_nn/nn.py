import numpy as np

epochs=200
n_layers=6
batch_size=64
sample_num=-1
l=[]
nx=-1

class layer:
    def __init__(self,num,actfunc) -> None:
        self.a=np.zeros(self.cell_num,batch_size)
        self.z=np.zeros(self.cell_num,batch_size)
        self.i=num
        self.cell_num=l[self.i]
        self.actfun=actfunc
        self.W=np.random.random((self.cell_num,sample_num))
        self.b=np.zeros(self.cell_num)
        self.z=np.zeros(self.cell_num)
        self.a=np.zeros(self.cell_num)
        self.dW=np.zeros((self.cell_num,sample_num))
        self.db=np.zeros(self.cell_num)
    def actfunc(self,z):
        if self.actfunc=="sigmoid":
            return 1/(1+np.exp(z)) 
        elif self.actfunc=="tanh":
            return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        elif self.actfunc=="relu":
            return np.maximum(0,z)
    def derivative_actfunc(self,z):
        if self.actfunc=="sigmoid":
            return np.exp(-z)/(1+np.exp(-z))**2 
        elif self.actfunc=="tanh":
            return 1-((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))**2
        elif self.actfunc=="relu":
            if z>0:
                return 1
            else:
                return 0
    def forwardprop(self,lst):
        self.z=np.matmul(self.W,lst.a)+self.b
        self.a=self.actfunc(self.z)
    def backwardprop(self,lst):
        self.dz=np.multiply(self.da,self.derivative_actfunc(self.z))
        self.dW=np.matmul(self.dz,lst.a)
        self.db=self.dz
        lst.da=np.matmul(self.W.T,self.dz)
    def update(self,alpha):
        self.W=self.W-alpha*self.dW
        self.b-=alpha*self.db
    
inpt=layer(nx,)

for i in range(n_layers):
    



for epoch in range(epochs):
    pass