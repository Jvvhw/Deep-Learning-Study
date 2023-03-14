from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

class SingleLayer: 
    def __init__(self, learning_rate=0.1,l1=0,l2=0): 
        self.w = None
        self.b = None
        self.losses = []
        self.val_losses = []
        self.w_history = []
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2
    
    def update_val_loss(self,x_val,y_val):
        z = self.forpass(x_val)
        a = self.activation(z)
        a = np.clip(a,1e-10,1-1e-10)
        val_loss = np.sum(-(y_val * np.log(a) + (1-y_val) * np.log(1-a))) 
        self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))

    def reg_loss(self):
        return self.l1 * np.sum(np.abs(self.w)) + self.l2 / 2 * np.sum(self.w**2)


    def forpass(self, x):
        z = np.dot(x , self.w) + self.b
        return z
    
    def backprop(self, x, err):
        m = len(x)
        w_grad = np.dot(x.T, err) / m
        b_grad = np.sum(err) / m
        return w_grad, b_grad
    
    def activation(self,z):
        z = np.clip(z, -100, None)
        a = 1/(1 + np.exp(-z))         
        return a
    
    def fit(self,x,y,epochs=100,x_val=None,y_val=None):
        y = y.reshape(-1,1)
        y_val = y_val.reshape(-1,1)
        m = len(x)
        self.w = np.ones((x.shape[1],1))  
        self.b = 0
        self.w_history.append(self.w.copy())
        for i in range(epochs):
          z = self.forpass(x)
          a = self.activation(z)
          err = -(y - a)
          w_grad, b_grad = self.backprop(x, err)
          w_grad += (self.l1 * np.sign(self.w) + self.l2 * self.w) / m
          self.w -= self.lr * w_grad
          self.b -= self.lr * b_grad
          self.w_history.append(self.w.copy())
          a = np.clip(a,1e-10,1-1e-10)
          loss = np.sum(-(y*np.log(a) + (1-y) * np.log(1-a)))
          self.losses.append((loss + self.reg_loss()) / m)
          self.update_val_loss(x_val,y_val)

    def predict(self, x):
        z = self.forpass(x)
        return z > 0  
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y.reshape(-1,1))
    
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

x_train_all, x_test, y_train_all, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,stratify=y_train_all,test_size=0.2,random_state=42) 

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)

single_layer = SingleLayer(l2=0.01)
single_layer.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val,epochs=10000)
print(single_layer.score(x_val_scaled,y_val))

plt.ylim(0, 0.3)
plt.plot(single_layer.losses)
plt.plot(single_layer.val_losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'])
plt.show()

w2 = []
w3 = []
for w in single_layer.w_history:
    w2.append(w[2])
    w3.append(w[3])

plt.plot(w2,w3)
plt.xlabel('[w2]')
plt.ylabel('[w3]')
plt.show()    