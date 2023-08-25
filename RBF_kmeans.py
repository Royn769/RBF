import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.cluster import KMeans
 
class RBF:
    #初始化学习率、学习步数
    def __init__(self,learning_rate=0.002,step_num=10001,hidden_size=10):
        self.learning_rate=learning_rate
        self.step_num=step_num
        self.hidden_size=hidden_size
    
    #使用 k-means 获取聚类中心、标准差
    def getC_S(self,x,class_num):
        estimator=KMeans(n_clusters=class_num,max_iter=10000) #构造聚类器
        estimator.fit(x) #聚类
        c=estimator.cluster_centers_
        n=len(c)
        s=0;
        for i in range(n):
            j=i+1
            while j<n:
                t=np.sum((c[i]-c[j])**2)
                s=max(s,t)
                j=j+1
        s=np.sqrt(s)/np.sqrt(2*n)
        return c,s
    
    #高斯核函数(c为中心，s为标准差)
    def kernel(self,x,c,s):
        x1=tf.tile(x,[1,self.hidden_size]) #将x水平复制 hidden次
        x2=tf.reshape(x1,[-1,self.hidden_size,self.feature])
        dist=tf.reduce_sum((x2-c)**2,2)
        return tf.exp(-dist/(2*s**2))
    
    #训练RBF神经网络
    def train(self,x,y):
        self.feature=np.shape(x)[1] #输入值的特征数
        self.c,self.s=self.getC_S(x,self.hidden_size) #获取聚类中心、标准差
        
        x_=tf.placeholder(tf.float32,[None,self.feature]) #定义placeholder
        y_=tf.placeholder(tf.float32,[None,np.shape(y)[1]]) #定义placeholder
        
        #定义径向基层
        z=self.kernel(x_,self.c,self.s)  
    
        #定义输出层
        w=tf.Variable(tf.random_normal([self.hidden_size,np.shape(y)[1]]))
        b=tf.Variable(tf.zeros([np.shape(y)[1]]))
        yf=tf.matmul(z,w)+b
        
        loss=tf.reduce_mean(tf.square(y_-yf))#二次代价函数
        optimizer=tf.train.AdamOptimizer(self.learning_rate) #Adam优化器     
        train=optimizer.minimize(loss) #最小化代价函数
        init=tf.global_variables_initializer() #变量初始化
    
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.step_num):
                sess.run(train,feed_dict={x_:x,y_:y})
                if epoch>0 and epoch%500==0:
                    mse=sess.run(loss,feed_dict={x_:x,y_:y})
                    # print(epoch,mse)
            self.w,self.b=sess.run([w,b],feed_dict={x_:x,y_:y})
        
    def kernel2(self,x,c,s): #预测时使用
        x1=np.tile(x,[1,self.hidden_size]) #将x水平复制 hidden次
        x2=np.reshape(x1,[-1,self.hidden_size,self.feature])
        dist=np.sum((x2-c)**2,2)
        return np.exp(-dist/(2*s**2))
    
    def predict(self,x):
        z=self.kernel2(x,self.c,self.s)
        pre=np.matmul(z,self.w)+self.b
        return pre