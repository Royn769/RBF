import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
 
class RBF:
    # initial
    def __init__(self,learning_rate=0.002,step_num=10001,hidden_size=10):
        self.learning_rate=learning_rate
        self.step_num=step_num
        self.hidden_size=hidden_size
    
    # gausse function c=center s=stardared deviation
    def kernel(self,x,c,s):
        x1=tf.tile(x,[1,self.hidden_size]) #tiling x for [hidden_size] times
        x2=tf.reshape(x1,[-1,self.hidden_size,self.feature])
        dist=tf.reduce_sum((x2-c)**2,2)
        return tf.exp(-dist/(2*s**2))
    
    # train RBF network
    def train(self,x,y):
        self.feature=np.shape(x)[1] # input dim
        x_=tf.placeholder(tf.float32,[None,self.feature]) #placeholder
        y_=tf.placeholder(tf.float32,[None,np.shape(y)[1]]) #placeholder
        
        #radial basis layer
        c=tf.Variable(tf.random_normal([self.hidden_size,self.feature]))
        s=tf.Variable(tf.random_normal([self.hidden_size]))
        z=self.kernel(x_,c,s)
    
        #output layer
        w=tf.Variable(tf.random_normal([self.hidden_size,np.shape(y)[1]]))
        b=tf.Variable(tf.zeros([np.shape(y)[1]]))
        yf=tf.matmul(z,w)+b
        
        loss=tf.reduce_mean(tf.square(y_-yf))#cost function
        optimizer=tf.train.AdamOptimizer(self.learning_rate) #Adam optimizer
        train=optimizer.minimize(loss) #minimize loss
        init=tf.global_variables_initializer() #initial variables
    
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.step_num):
                sess.run(train,feed_dict={x_:x,y_:y})
                if epoch>0 and epoch%500==0:
                    mse=sess.run(loss,feed_dict={x_:x,y_:y})
                    # print(epoch,mse)
            self.c,self.s,self.w,self.b=sess.run([c,s,w,b],feed_dict={x_:x,y_:y})
        
    def kernel2(self,x,c,s):
        x1=np.tile(x,[1,self.hidden_size])
        x2=np.reshape(x1,[-1,self.hidden_size,self.feature])
        dist=np.sum((x2-c)**2,2)
        return np.exp(-dist/(2*s**2))
    
    # predict
    def predict(self,x):
        z=self.kernel2(x,self.c,self.s)
        pre=np.matmul(z,self.w)+self.b
        # print(pre)
        return pre
    
    