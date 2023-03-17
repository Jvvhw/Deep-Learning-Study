from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd', alpha=0.01, batch_size=32, learning_rate_init=0.1, max_iter=500)

'''
hidden_layer_sizes=(10,) : 은닉층의 수와 뉴런 갯수를 지정 직관적임. (100,50,10) -> 층 3개 각각 뉴런 100개 50개 10개

activation='logistic' : 활성화 함수 선택, logistic은 시그모이드, ReLU나 atan도 선택 가능함

solver='sgd' : 경사하강법 알고리즘 sgd는 확률적경사하강법

alpha=0.01 규제 상수, 기본적으로 sklearn 에서는 l2규제만을 지원

batch_size=32 : 배치 크기

learning_rate_init=0.1 : 학습률 초기값

max_iter=500 : 에포크횟수
'''
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

x_train_all, x_test, y_train_all, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,stratify=y_train_all,test_size=0.2,random_state=42) 

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)

mlp.fit(x_train_scaled, y_train)
print(mlp.score(x_val_scaled, y_val))
