import csv
import matplotlib.pyplot as plt
import numpy as np

'''
-------------------------------------------------------------------------------------------------------------------------
'''
class Neuron:
    def __init__(self):
        self.w = 1
        self.b = 1

    def forpass(self,x):
        y_hat = x*self.w + self.b
        return y_hat
    
    def backprop(self,x,err):
        w_grad = x*err
        b_grad = 1*err
        return w_grad, b_grad

#csv파일을 열어서 리스트로 리턴하는 함수
def fileopen(filename):
    data = list()
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)
    return data



'''
-------------------------------------------------------------------------------------------------------------------------
'''

Vdc = 200
datas = list()
datas2 = list()

for n in range (0,11,1): 
    fname = 'Idata' + str(n) + '.csv'
    csv_list = fileopen(fname)
    Idata = []
    for n in range(0,21,1): 
        if n%2 == 1:
            Idata.append(round(float(csv_list[n][1]),2))
    datas.append(Idata)

for n in range (0,11,1): 
    fname = 'resistance' + str(n) + '.csv'
    csv_list = fileopen(fname)
    randata = []
    for n in range(0,10,1): 
        randata.append(round(float(csv_list[n][1]),2))
    datas2.append(randata)

list1 = sum(datas,[])
list2 = sum(datas2, [])

neuron = Neuron()


for i in range(1000):
    for x_i, y_i in zip(list2,list1):
        y_hat = neuron.forpass(x_i)
        err = y_hat - y_i
        w_grad, b_grad = neuron.backprop(x_i,err)
        neuron.w = neuron.w - (0.003*w_grad)
        neuron.b = neuron.b - (0.003*b_grad)

pt1 = (-0.1,-0.1 * neuron.w + neuron.b)
pt2 = (40, 40*neuron.w + neuron.b)

plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],'r')
plt.scatter(list2,list1)
plt.xlabel('resistance')
plt.ylabel('I')
plt.show()

