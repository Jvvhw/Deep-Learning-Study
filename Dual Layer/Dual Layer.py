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
    

class DualLayer(SingleLayer):

    def __init__(self, units=10, learning_rate=0.1, l1=0, l2=0):
        self.units = units        # 은닉층의 뉴런 갯수  
        self.w1 = None            # 은닉층 가중치  
        self.b1 = None            # 은닉층 절편  
        self.w2 = None            # 출력층 가중치  
        self.b2 = None            # 출력층 절편  
        self.a1 = None            # 은닉층의 활성화 함수 통과한 출력  
        self.losses = []          # 훈련 손실  
        self.val_losses = []      # 검증 손실      
        self.lr = learning_rate   # 학습률
        self.l1 = l1              # l1 규제 하이퍼파라미터  
        self.l2 = l2              # l2 규제 하이퍼파라미터  

    def forpass(self, x):
        z1 = np.dot(x,self.w1) + self.b1         # 첫번째 층 선형 계산
        self.a1 = self.activation(z1)            # 활성화 함수 적용, 결과 = a1   
        z2 = np.dot(self.a1, self.w2) + self.b2  # 두번째 층 선형 계산 (출력층)
        return z2                                # 정방향 계산 첫번째 층 -> 활성화 함수 통과 -> 두번째 층 -> 출력                        

    def backprop(self, x, err):                  # 역전파 과정 잘 볼것
        m = len(x)                               # 샘플 갯수 m
        w2_grad = np.dot(self.a1.T, err) / m
        b2_grad = np.sum(err) / m                # 두 번째 층 (출력층) 가중치와 절편의 그레디언트 계산
        err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1-self.a1)  # 첫번째 층 (은닉층) 그레디언트 계산에 쓸 연쇄법칙 전개식
        w1_grad = np.dot(x.T, err_to_hidden) / m 
        b1_grad = np.sum(err_to_hidden, axis=0) / m                     # 은닉층 가중치와 절편 그레디언트 계산
        return w1_grad, b1_grad, w2_grad, b2_grad    
    
    def init_weights(self,n_features):              # 가중치 초기화
        self.w1 = np.ones((n_features, self.units)) # (특성 갯수, 은닉층 크기)
        self.b1 = np.zeros(self.units)              # 은닉층 크기
        self.w2 = np.ones((self.units, 1))          # (은닉층 크기, 1)
        self.b2 = 0

    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        y = y.reshape(-1,1)            # 타깃을 열 벡터로  
        y_val = y_val.reshape(-1,1)     
        m = len(x)                                                  
        self.init_weights(x.shape[1])  # 가중치 초기화
        for i in range(epochs):
            a = self.training(x, y, m)
            a = np.clip(a, 1e-10, 1-1e-10)
            loss = np.sum(-(y*np.log(a) + (1-y) * np.log(1-a)))
            self.losses.append((loss + self.reg_loss()) / m)
            self.update_val_loss(x_val, y_val)
    
    def training(self, x, y, m):
        z = self.forpass(x)    # 정방향 계산
        a = self.activation(z) # 활성화 함수 적용
        err = -(y-a)           # 오차 계산
        w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)       # 오차 역전파, 그레디언트 계산
        w1_grad += (self.l1 * np.sign(self.w1) + self.l2 * self.w1) / m  # 규제 적용   
        w2_grad += (self.l1 * np.sign(self.w2) + self.l2 * self.w2) / m
        #아래부터 차례대로 은닉층, 출력층 가중치, 절편 계산
        self.w1 -= self.lr * w1_grad
        self.b1 -= self.lr * b1_grad
        self.w2 -= self.lr * w2_grad
        self.b2 -= self.lr * b2_grad
        return a
    
    def reg_loss(self):
        return self.l1 * (np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + \
               self.l2 / 2 * (np.sum(self.w1**2) + np.sum(self.w2**2))
    

class RandomInitNetwork(DualLayer):  
# 손실 함수가 불안정하고 느리게 감소하여 Xavier/Glorot 방법으로 가중치를 초기화하는 클래스를 하나 더 생성 (by. chat GPT)
    def init_weights(self, n_features):
        n_output = 1  # 출력 뉴런의 개수
        limit = np.sqrt(2 / (n_features + n_output))
        self.w1 = np.random.uniform(-limit, limit, (n_features, self.units))
        self.b1 = np.zeros(self.units)
        self.w2 = np.random.uniform(-limit, limit, (self.units, n_output))
        self.b2 = 0

cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

x_train_all, x_test, y_train_all, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,stratify=y_train_all,test_size=0.2,random_state=42) 

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)


'''
dual_layer = DualLayer(l2 = 0.01)
dual_layer.fit(x_train_scaled, y_train, x_val = x_val_scaled, y_val = y_val, epochs =20000)
print(dual_layer.score(x_val_scaled, y_val))
'''

random_init_net = RandomInitNetwork(l2 = 0.01)
random_init_net.fit(x_train_scaled, y_train, x_val = x_val_scaled, y_val = y_val, epochs =20000)
print(random_init_net.score(x_val_scaled, y_val))


plt.ylim(0, 0.3)
plt.plot(random_init_net.losses)
plt.plot(random_init_net.val_losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss' , 'val_loss'])
plt.show()