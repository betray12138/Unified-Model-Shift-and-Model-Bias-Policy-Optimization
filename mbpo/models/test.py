import numpy as np
import tensorflow as tf
batch_size = 1024
num_nets = 7
batch_inds = np.arange(0, batch_size,dtype=np.int32).reshape((batch_size, 1))
model_inds = np.array(np.random.choice(num_nets, size=batch_size).reshape((batch_size, 1)),dtype=np.int32)
a = tf.random.normal((7,1024,18))
# index_3_0 = np.zeros((batch_size, 1),dtype=np.int32)
# print(index_3_0.shape)
# idx = np.hstack((model_inds,batch_inds))
# print(idx.shape)
# print(tf.gather_nd(a, idx).shape)

print(type(len(a)))