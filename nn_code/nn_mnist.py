import gzip
import cPickle

import tensorflow as tf
import numpy as np
import matplotlib as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y,10)
valid_y = one_hot(valid_y,10)
test_y = one_hot(test_y, 10)


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(test_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print test_y[57]


# TODO: the neural net!!


x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])


W1 = tf.Variable(np.float32(np.random.rand(784, 50)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(50)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(50, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1) #matmul funcion que multiplica el valor de la muestra por el peso y se le suma el umbral
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y)) # se suma el cuadrado de las restas de los valores de salidad y las etiquetas

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables() #tensorflow reserva espacio para todas las variables dentro de la gpu

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
parar = True
epoch = 0
error = 10000
contador = 0
errores = []

while parar:
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size] #almacene 20 muestras
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size] #almacene 20 etiquetas

        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys}) #lanzo el objeto trainin y lo alimento con LAS MUESTRAS Y LAS ETIQUETAS

    epoch += 1
    print "Validacion"
    print "Epoch #:", epoch, "Error validacion: ", sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    print "----------------------------------------------------------------------------------"
    errorActual = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    errores.append(errorActual)

    if (errorActual >= error):
        contador = 1

    if (contador == 1):
        parar = False

    error = errorActual

print "test"
result = sess.run(y, feed_dict={x: test_x})
aciertos = 0
for b, r in zip(test_y, result):
    if (np.argmax(b) == np.argmax(r)):
        aciertos += 1
print "Error: ", sess.run(loss, feed_dict={x: test_x, y_: test_y})
print "El numero de aciertos es: ", aciertos
print "----------------------------------------------------------------------------------"


plt.plot(errores)
plt.show()

