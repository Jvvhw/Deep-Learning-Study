from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

class SingleLayer:

    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b
        return z
    
    def backprop(self, x, err):
        w_grad = x* err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self,z):
        z = np.clip(z, -100, None)
        a = 1/(1 + np.exp(-z))         # 시그모이드 함수 자체
        return a
    
    def fit(self,x,y,epochs=100):
        self.w = np.ones(x.shape[1])   # ones,zeros MATLAB과 동일한 기능
        self.b = 0
        for i in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x))) # 인덱스 셔플 -> 결과값은 랜덤 번호표와 같음
            for i in indexes:                                  # 위의 랜덤한 순서대로 
                z = self.forpass(x[i])  # 선형함수의 결과 z
                a = self.activation(z)  # 활성화 함수를 통과
                err = -(y[i] - a)       # 학습에 사용할 err   
                w_grad, b_grad = self.backprop(x[i], err)      # 역방향 계산 
                self.w -= w_grad        # 가중치 업데이트
                self.b -= b_grad        # 절편 업데이트
                a = np.clip(a,1e-10,1-1e-10) # 로그 계산시 슈팅 방지

                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a)) # 평균 손실 저장
            self.losses.append(loss/len(y))

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x] 
        return (np.array(z) > 0)  
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    

layer = SingleLayer()
layer.fit(x_train, y_train)        # 훈련 돌입
print(layer.score(x_test,y_test))  # 테스트 군으로 검증

'''
매 실행마다 Score 값이 다르게 나오고 평균적으로 Logistic Regression의 코드보다 정확도가 높아진걸 확인할 수 있다.
에포크마다 훈련세트를 섞어 손실함수의 값을 줄였기 때문
'''


