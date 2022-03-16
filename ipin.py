import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler      #数据预处理
import glob,os
from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor


base_path = '../'
traindata = np.array(pd.read_csv('./train2.csv'))
testdata = np.array(pd.read_csv('./test2.csv'))

#9296个参考标签 100个rss
rss = traindata[:,0:520]#9296*100
rss1 = rss[1955:3610,:]
pos = traindata[:,520:522]#9296*2
floor = traindata[:,522:523]

rssTest = testdata[:,0:520]
rssTest_pos = testdata[:,520:522]
print(rss)
print(rssTest)
def elm():
    #ELM隐藏层
    class HiddenLayer:
        #num为隐藏层节点数
        def __init__(self,x,num):
            row = x.shape[0]
            columns = x.shape[1]
            rnd = np.random.RandomState(4444)
            self.w = rnd.uniform(-1,1,(columns,num))
            self.b = np.zeros([row,num],dtype=float)
            for i in range(num):
                rand_b = rnd.uniform(-0.4,0.4)
                for j in range(row):
                    self.b[j,i] = rand_b
            h = self.sigmoid(np.dot(x,self.w)+self.b)
            self.H_ = np.linalg.pinv(h)
            #print(self.H_.shape) (10,7436)
        def sigmoid(self,x):
            return 1.0 / (1 + np.exp(-x))
    
        def regressor_train(self,T):
            #T = T.reshape(-1,1)
            self.beta = np.dot(self.H_,T)
            return self.beta
        def classifisor_train(self,T):
            en_one = OneHotEncoder()
            T = en_one.fit_transform(T.reshape(-1,1)).toarray() #独热编码之后一定要用toarray()转换成正常的数组
            # T = np.asarray(T)
            print(self.H_.shape)
            print(T.shape)
            self.beta = np.dot(self.H_,T)
            print(self.beta.shape)
            return self.beta
        def regressor_test(self,test_x):
            b_row = test_x.shape[0]
            h = self.sigmoid(np.dot(test_x,self.w)+self.b[:b_row,:])
            result = np.dot(h,self.beta)
            return result
        def classifisor_test(self,test_x):
            b_row = test_x.shape[0]
            h = self.sigmoid(np.dot(test_x,self.w)+self.b[:b_row,:])
            result = np.dot(h,self.beta)
            result = [item.tolist().index(max(item.tolist())) for item in result]
            return result

    stdsc = StandardScaler()
    x, y = stdsc.fit_transform(rss), pos
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
    a = HiddenLayer(x_train,5)
    a.regressor_train(y_train)
    result = a.regressor_test(rssTest[1])
 
    print("*************************")
    print("elm方法")
    print("预测目标所在位置为：", result[0])
    print("预测误差为：", result[0]-rssTest_pos[1])
    #print("目标所在楼层为：",floor[n])
    print("")


def bp():
    model = MLPRegressor(hidden_layer_sizes=(10,), random_state=10,learning_rate_init=0.1)  # BP神经网络回归模型
    model.fit(rss,pos)  # 训练模型
    pre = model.predict([rssTest[1]])  # 模型预测
    #np.abs(data_te.iloc[:,2]-pre).mean()  # 模型评价
    print("*************************")
    print("bp方法")
    print("预测目标所在位置为：",pre)
    print("预测误差为：", pre-rssTest_pos[1])
    #print("目标所在楼层为：",floor[n])
    print("")

def nn():
    
    n = np.argmin(np.linalg.norm(rss-rssTest[1],axis = 1))
    
    print("*************************")
    print("nn方法")
    print(n)
    print("预测目标所在位置为：",pos[n])
    print("预测误差为：", pos[n]-rssTest_pos[1])
    print("目标所在楼层为：",floor[n])
    print("")

def knn(k):
    arr = np.linalg.norm(rss-rssTest[1],axis = 1)
    id = np.argpartition(arr, k)
    targetpos = 0
    for i in range(k):
        targetpos += pos[id[i]]
    
    #return(id[0], id[1], id[2])
    print("*************************")
    print(pos[id[0]],pos[id[1]],pos[id[2]])
    print("knn方法")
    print("预测目标所在位置为：",targetpos/k)
    print("预测误差为：", targetpos/k - rssTest_pos[1])
    print("目标所在楼层为：",floor[id[k]])
    print("")

def kmeans(k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(rss1)
    print(kmeans.labels_)
    print(kmeans.predict([rssTest[1]]))
    

def landmark(k):
    arr = np.linalg.norm(rss-rssTest[1],axis = 1)
    id = np.argpartition(arr, k)#将欧氏距离最小前k个标签id打印
    sum = 0#就是求每个参考标签权重的分母
    for i in range(k):
        sum += 1/(arr[id[i]]**2)
    targetpos = 0
    w = np.zeros((k,1))
    #求参考抱歉权重
    for i in range(k):
        w[i] = (1/(arr[id[i]]**2))/sum
    #预测
    for i in range(k):
        targetpos += w[i] * pos[id[i]]

    print("*************************")
    print("landmark方法")
    print("预测目标所在位置为：",targetpos)
    print("预测误差为：", targetpos - rssTest_pos[1])
    print("目标所在楼层为：",floor[id[k]])
    print("")

nn()
knn(10)
landmark(10)
#kmeans(10)
elm()
#bp()