from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
mx.random.seed(1)

data_ctx = mx.cpu()
model_ctx = mx.cpu()

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)


num_inputs = 784
num_outputs = 10
num_examples = 60000
batch_size = 64
train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)

test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)


W = nd.random_normal(shape=(num_inputs, num_outputs),ctx=model_ctx)
b = nd.random_normal(shape=num_outputs,ctx=model_ctx)

params = [W, b]

for param in params:
    param.attach_grad()

def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
    norms = nd.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

sample_y_linear = nd.random_normal(shape=(2,10))
sample_yhat = softmax(sample_y_linear)
#print(sample_yhat)

#print(nd.sum(sample_yhat, axis=1))

def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat

def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat+1e-6))

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

evaluate_accuracy(test_data, net)

epochs = 1
learning_rate = .01
print("Model is learning now")
for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

# Define the function to do prediction
def model_predict(net,data):
    output = net(data)
    return nd.argmax(output, axis=1)

# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mnist_test, 10000 , shuffle= False)
pred = np.zeros(10000, int)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
predict = pred.asnumpy()
testing_confusion_matrix = np.zeros((10, 10), int)
for i in range(10000):
    image, label = mnist_test[i]
    label1=int(label)
    y = int(predict[i])
    testing_confusion_matrix[label1, y] += 1

print(testing_confusion_matrix)

TP = np.zeros(10, int)
FP = np.zeros(10, int)
FN = np.zeros(10, int)

# Calculating True Positives
for i in range(10):
    for j in range(10):
        if (i == j):
            TP[i] += testing_confusion_matrix[i,j]
# Calculating False Positives
for i in range(10):
    FP[i] = sum(testing_confusion_matrix[:,i])-testing_confusion_matrix[i,i]


# Calculating False Negatives
for i in range(10):
    FN[i] = sum(testing_confusion_matrix[i, :], 2)-testing_confusion_matrix[i,i]

#Calculating Precision

prec = np.zeros(10, float)
for i in range(10):
    prec[i] = float(TP[i])/(float(TP[i]) + float(FP[i]))

print("Precision:")
pf = 0.0
for i in range(10):
    pf += prec[i]
print((pf/10.0)*100, "%")

#Caluclating Recall
recall = np.zeros(10,float)
for i in range(10):
    recall[i] = float(TP[i])/(float(TP[i]) + float(FN[i]))

print("Recall:")
re = 0.0
for i in range(10):
    re += recall[i]
print((re/10.0)*100, "%")

# Calculating Accuracy
def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements

print("accuracy:", (accuracy(testing_confusion_matrix)*100) ,"%")