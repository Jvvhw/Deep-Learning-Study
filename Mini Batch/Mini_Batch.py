from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from network_models import RandomInitNetwork
import numpy as np
import matplotlib.pyplot as plt

class MinibatchNetwork(RandomInitNetwork):
    def __init__(self, units=10, batch_size=32, learning_rate=0.1, l1=0, l2=0):
        super().__init__(units,learning_rate,l1,l2)
        self.batch_size = batch_size # 배치 크기

    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        y_val = y_val.reshape(-1,1)
        self.init_weights(x.shape[1])
        np.random.seed(42)
        for i in range(epochs):
            loss = 0
            for x_batch, y_batch in self.gen_batch(x,y): # 에포크를 도는 도중 미니배치를 순회하는 for문이 추가
                y_batch = y_batch.reshape(-1,1)
                m = len(x_batch)
                a = self.training(x_batch, y_batch, m)
                a = np.clip(a,1e-10,1-1e-10)
                loss += np.sum(-(y_batch * np.log(a) + (1-y_batch) * np.log(1-a)))
            self.losses.append((loss + self.reg_loss()) / len(x))
            self.update_val_loss(x_val, y_val)

    def gen_batch(self, x, y):
        length = len(x)
        bins = length // self.batch_size # 전체길이를 배치사이즈로 나누어서 배치갯수를 구함
        if length % self.batch_size:     # 나누어떨어지지 않으면 +1
            bins += 1 

        indexes = np.random.permutation(np.arange(len(x)))
        x = x[indexes]
        y = y[indexes]
        for i in range(bins): #구간의 갯수만큼 데이터를 자르기 위해서 인덱스를 계산
            start = self.batch_size * i  
            end = self.batch_size * (i+1)
            yield x[start:end], y[start:end] # for문 안에서 값을 반환해야하므로 yield를 사용

cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

x_train_all, x_test, y_train_all, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,stratify=y_train_all,test_size=0.2,random_state=42) 

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)

minibatch_net = MinibatchNetwork(l2=0.01, batch_size=512)
minibatch_net.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val, epochs = 500)
print(minibatch_net.score(x_val_scaled, y_val))

plt.plot(minibatch_net.losses)
plt.plot(minibatch_net.val_losses)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(['train_loss','val_loss'])
plt.show()
