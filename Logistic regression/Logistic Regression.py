from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

# 박스플롯으로 보면 4,14,24 번째 특성만 상대적으로 분포 범위가 넓음
'''
plt.boxplot(cancer.data)
plt.xlabel('feature')
plt.ylabel('value')
plt.show()
'''

#타겟 데이터를 살펴보면 0과 1로만 구성되어있음

#print(np.unique(cancer.target,return_counts=True))

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)
'''
stratify 클래스 비율이 불균형할 경우 y로 지정

test_size 훈련과 테스트군의 비율 조정, 현재 8:2 설정

랜덤 시드, 결과 비교를 위해 교재와 동일하게 설정

'''

#분할 확인
#print(x_train.shape, x_test.shape)

class LogisticNeuron:

    def __init__(self):
        self.w = None
        self.b = None

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b
        return z
    
    def backprop(self, x, err):
        w_grad = x* err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self,z):
        a = 1/(1 + np.exp(-z))         # 시그모이드 함수 자체
        return a
    
    def fit(self,x,y,epochs=100):
        self.w = np.ones(x.shape[1])   # ones,zeros MATLAB과 동일한 기능
        self.b = 0
        for i in range(epochs):
            for x_i, y_i in zip(x,y):
                z = self.forpass(x_i)  # 선형함수의 결과 z
                a = self.activation(z) # 활성화 함수를 통과
                err = -(y_i - a)       # 학습에 사용할 err   
                w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산 
                self.w -= w_grad       # 가중치 업데이트
                self.b -= b_grad 

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x] 
        a = self.activation(np.array(z))
        return (a > 0.5)     # 임계 함수

neuron = LogisticNeuron()
neuron.fit(x_train,y_train)  # 훈련 돌입

print(np.mean(neuron.predict(x_test) == y_test))

    
    
    

        