from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt

class SingleLayer: 
# 가중치 변화를 보기 위해 이전 장에서 복사
# 실제로는 딥러닝 패키지들에서 모든기능을 포함한 클래스를 제공함
    def __init__(self, learning_rate=0.1,l1=0,l2=0): # l1과 l2로 규제를 적용할 수 있다, 초기값 0 = 규제없음
        self.w = None
        self.b = None
        self.losses = []
        self.val_losses = []
        self.w_history = []
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2
    
    def update_val_loss(self,x_val,y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i])
            a = self.activation(z)
            a = np.clip(a,1e-10,1-1e-10)
            val_loss += -(y_val[i]*np.log(a)+(1-y_val[i])*np.log(1-a)) 
        self.val_losses.append(val_loss/len(y_val) + self.reg_loss())

    def reg_loss(self):
        return self.l1 * np.sum(np.abs(self.w)) + self.l2 / 2 * np.sum(self.w**2)


    def forpass(self, x):
        z = np.sum(x * self.w) + self.b
        return z
    
    def backprop(self, x, err):
        w_grad = x* err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self,z):
        z = np.clip(z, -100, None)
        a = 1/(1 + np.exp(-z))         
        return a
    
    def fit(self,x,y,epochs=100,x_val=None,y_val=None):
        self.w = np.ones(x.shape[1])  
        self.b = 0
        self.w_history.append(self.w.copy())
        np.random.seed(42)
        for i in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x))) 
            for i in indexes:                                  
                z = self.forpass(x[i])  
                a = self.activation(z)  
                err = -(y[i] - a)          
                w_grad, b_grad = self.backprop(x[i], err)       
                w_grad += self.l1 * np.sign(self.w) + self.l2 * self.w # l1,l2 동시적용도 가능
                self.w -= self.lr * w_grad #학습률 적용        
                self.b -= b_grad        
                self.w_history.append(self.w.copy())
                a = np.clip(a,1e-10,1-1e-10) 
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a)) 
            self.losses.append(loss/len(y) + self.reg_loss())
            self.update_val_loss(x_val,y_val)

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x] 
        return (np.array(z) > 0)  
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    

cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

x_train_all, x_test, y_train_all, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)
# 훈련세트와 테스트 세트 분할 0.2
x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,stratify=y_train_all,test_size=0.2,random_state=42) 
# 훈련세트에서 검증 세트 분할 0.2

'''
sgd = SGDClassifier(loss='log',random_state=42) 
sgd.fit(x_train,y_train)
print(sgd.score(x_val, y_val))
'''
# loss함수를 log로 했을때와 hinge로 했을때 성능 비교
# 튜닝으로 모델의 성능을 향상시킬 수 있음

'''
print(cancer.feature_names[[2,3]])
plt.boxplot(x_train[:,2:4])
plt.xlabel('feature')
plt.ylabel('value')
plt.show()
'''
# 특성들간의 스케일 차이가 큰 것을 확인

'''
layer1 = SingleLayer()
layer1.fit(x_train,y_train)
print(layer1.score(x_val,y_val))

w2 = []
w3 = []
for w in layer1.w_history:
    w2.append(w[2])
    w3.append(w[3])

plt.plot(w2,w3)
plt.plot(w2[-1],w3[-1],'ro')
plt.xlabel('w[2]')
plt.ylabel('w[3]')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.annotate("w[3] 특성값의 스케일이 상대적으로 커서\n가중치가 학습과정에서 크게 요동침",(500,0.5))
plt.show()
'''


train_mean = np.mean(x_train,axis=0) # 평균
train_std = np.std(x_train,axis=0)   # 표준편차
x_train_scaled = (x_train - train_mean) / train_std # 표준화 공식 
x_val_scaled = (x_val - train_mean) / train_std     # 검증세트 표준화도 훈련세트의 표준편차와 평균으로 진행


'''
layer2 = SingleLayer()
layer2.fit(x_train_scaled,y_train)
print(layer2.score(x_val,y_val))      # 검증세트도 스케일링을 하지 않으면 성능이 안나옴
print(layer2.score(x_val_scaled,y_val))


w2 = []
w3 = []
for w in layer2.w_history:
    w2.append(w[2])
    w3.append(w[3])

plt.plot(w2,w3)
plt.plot(w2[-1],w3[-1],'ro')
plt.xlabel('w[2]')
plt.ylabel('w[3]')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.annotate("가중치가 이동되었고, 최적값에 빠르게 다가감",(-0.75,0.5))
plt.show()
'''

'''
layer3 = SingleLayer()
layer3.fit(x_train_scaled,y_train, x_val = x_val_scaled, y_val = y_val)
# 적절한 편향 분산 트레이드오프 선택
plt.ylim(0,0.3)
plt.plot(layer3.losses)
plt.plot(layer3.val_losses)
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend(['train_loss','val_loss'])
plt.show() 
'''

'''
layer4 = SingleLayer()
layer4.fit(x_train_scaled,y_train,epochs=20) # 앞선 손실값 그래프에서 약 20번의 에포크 이후로는 의미가 없다는 것을 알았음
print(layer4.score(x_val_scaled,y_val))
'''

'''
# l1 규제 적용
l1_list = [0.0001,0.001,0.01]
# 규제가 커질수록 손실은 높아지고 가중치는 0에 가까워짐
for l1 in l1_list:
    lyr = SingleLayer(l1=l1)
    lyr.fit(x_train_scaled,y_train,x_val=x_val_scaled,y_val=y_val)

    plt.plot(lyr.losses)
    plt.plot(lyr.val_losses)
    plt.title('Learning Curve (l1={})'.format(l1))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss','val_loss'])
    plt.ylim(0, 0.3)
    plt.show()

    plt.plot(lyr.w,'bo')
    plt.title('Weight (l1={})'.format(l1))
    plt.ylabel('value')
    plt.xlabel('weight')
    plt.ylim(-4,4)
    plt.show()
'''

'''
# l2 규제 적용
l2_list = [0.0001,0.001,0.01]
# l1과 양상은 같지만 바뀌는 정도가 상대적으로 덜함
for l2 in l2_list:
    lyr = SingleLayer(l2=l2)
    lyr.fit(x_train_scaled,y_train,x_val=x_val_scaled,y_val=y_val)

    plt.plot(lyr.losses)
    plt.plot(lyr.val_losses)
    plt.title('Learning Curve (l2={})'.format(l2))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss','val_loss'])
    plt.ylim(0, 0.3)
    plt.show()

    plt.plot(lyr.w,'bo')
    plt.title('Weight (l2={})'.format(l2))
    plt.ylabel('value')
    plt.xlabel('weight')
    plt.ylim(-4,4)
    plt.show()
'''

# 패키지에서 제공하는 규제 사용해보기
# 결과 score가 같다 -> 작동 방식이 동일함
sgd = SGDClassifier(loss='log_loss',penalty='l2',alpha=0.001, random_state=42)
sgd.fit(x_train_scaled,y_train)
sgd.score(x_val_scaled,y_val)


