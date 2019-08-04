import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
seed=23455
BATCH_SIZE=32
rng=np.random.RandomState(seed)
X=rng.rand(32,2)
Y=[[int(x0+x1>1)] for (x0,x1) in X]
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
loss=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=3000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%32
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%500==0:
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print("%d,%g"%(i,total_loss),"\n")
            print(sess.run(w1),sess.run(w2))
