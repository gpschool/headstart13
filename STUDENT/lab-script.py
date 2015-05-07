# import libraries
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import gzip
import cPickle
# load the data
f = gzip.open('mnist.pkl.gz','rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#########################################
gray() # use black and white plotting
plt.figure()
imshow(train_set[0][0,:].reshape(28, 28))
plt.figure()
imshow(train_set[0][1,:].reshape(28, 28))
plt.figure()
imshow(train_set[0][2,:].reshape(28, 28))

#########################################
print train_set[1][0], train_set[1][1], train_set[1][2]

#########################################
dg1 = 8
dg2 = 1

x_all, t_all = train_set
x = x_all[logical_or(t_all==dg1, t_all==dg2)]
t = t_all[logical_or(t_all==dg1, t_all==dg2)]

t[t==dg1] = -1
t[t==dg2] = 1

#########################################
def train(x, t, epochs, status_func = None):
    n, d = x.shape
    w = np.zeros(d)
    w0 = 0.
    for epoch in range(epochs):
        data = np.column_stack([x, t])
        np.random.shuffle(data)
        xe = data[:,:-1]
        te = data[:,-1]
        for i in range(n):
            yi = sign(np.dot(xe[i], w)+w0)
            if yi != te[i]:
                w += te[i] * xe[i]
                w0 += te[i]
    return w,w0

#########################################
w,w0 = train(x, t,100)
imshow(w.reshape(28, 28))

#########################################
x_new, t_new = test_set
x_test = x_new[logical_or(t_new==dg1, t_new==dg2)]
t_test = t_new[logical_or(t_new==dg1, t_new==dg2)]
t_test[t_test==dg1] = -1
t_test[t_test==dg2] = 1

#########################################
def predict(x, w,w0):
    return sign(np.dot(x, w)+w0)

pred_test = predict(x_test, w,w0)

#########################################
ind_correct = np.where(t_test == pred_test)[0]
ind_error1 = np.where(logical_and(t_test == 1 , pred_test == -1))[0]
ind_error2 = logical_and(t_test == -1 , pred_test == 1)
