import tensorflow as tf
import numpy as np


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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:105, 0:4].astype('f4')  # the samples are the four first rows of data
x_validation = data[105:127, 0:4].astype('f4')  # the samples are the four first rows of data
x_test = data[127:, 0:4].astype('f4')  # the samples are the four first rows of data

y_data = one_hot(data[:105, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
y_validation = one_hot(data[105:127, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
y_test = one_hot(data[127:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code


print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print
#se vuelcan las pruebas para el entrenamiento 
x = tf.placeholder("float", [None, 4])  # samples matriz de nolose por el numero en este caso 4 pruebas
y_ = tf.placeholder("float", [None, 3])  # labels matriz de nolose por el numero en este caso 3 etiquetas


W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1) #np.random.rand matriz randon de [4,5]
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

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

while parar:
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size] #almacene 20 muestras
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size] #almacene 20 etiquetas

        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys}) #lanzo el objeto trainin y lo alimento con LAS MUESTRAS Y LAS ETIQUETAS

    epoch += 1
    print "Epoch #:", epoch, "Error validacion: ", sess.run(loss, feed_dict={x: x_validation, y_: y_validation})
    print "----------------------------------------------------------------------------------"
    errorActual = sess.run(loss, feed_dict={x: x_validation, y_: y_validation})

    if (errorActual >= error):
        contador = 1

    if (contador == 1):
        parar = False

    error = errorActual


print "test"
result = sess.run(y, feed_dict={x: x_test})
aciertos = 0
for b, r in zip(y_test, result):
    if (np.argmax(b) == np.argmax(r)):
        aciertos += 1
print "Error: ", sess.run(loss, feed_dict={x: x_test, y_: y_test})
print "El numero de aciertos es: ", aciertos
print "----------------------------------------------------------------------------------"
