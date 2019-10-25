#fully connected network with dropout
#after 50 epochs 87% test accuracy with dropout 
import math
import sys
import numpy as np  
#from download_mnist import load
import operator  
import time
from collections import Counter
import matplotlib.pyplot as plt
np.set_printoptions(threshold = sys.maxsize)
X, y, x_test, y_test = load()



#layer1 784->200
W1 = 0.01 * np.random.randn(784,300)
b1 = np.zeros((1,300))

#layer2 200->50
W2 = .001 * np.random.rand(300,50)
b2 = np.zeros((1,50))

#layer3 50->10
W3 = 0.01 * np.random.randn(50,10)
b3 = np.zeros((1,10))

step_size = .01 #learning rate
reg = .01 # regularization 
dropout_rate = .5

num_examples = X.shape[0]

#training
#200 iterations
start = time.time()
for i in range(50):
      
      #forward implement dropout here
      X = np.where(X > 0 , X, .01 * X)
      hidden_layer1 = np.dot(X,W1) +b1
      hidden_layer1 =  np.where(hidden_layer1 > 0, hidden_layer1, .01* hidden_layer1)
      drop1 = np.random.rand(*hidden_layer1.shape) < dropout_rate /dropout_rate #scale for test
      hidden_layer1 *= drop1
      hidden_layer2 = np.dot(hidden_layer1, W2) +b2
      drop2 = np.random.rand(*hidden_layer2.shape) < dropout_rate / dropout_rate
      hidden_layer2 *= drop2
      scores = np.dot(hidden_layer2, W3) + b3
     
      exp_scores = np.exp(scores)
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
       

      predicted = np.argmax(scores,axis=1)
      print("train accuracy: ", np.mean(predicted==y))

      # compute the loss: average cross-entropy loss and regularization
      correct_logprobs = -np.log(probs[range(num_examples),y])
      data_loss = np.sum(correct_logprobs)/num_examples
      reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2) + .5*reg*np.sum(W3*W3)
      loss = data_loss + reg_loss
      if i % 1 == 0:
        print ("iteration: " ,i, " loss ",  loss)
      
      # compute the gradient on scores
      dscores = probs
      dscores[range(num_examples),y] -= 1
      dscores /= num_examples
      
       #backprop
      dW3 = np.dot(hidden_layer2.T,dscores)
      db3 = np.sum(dscores, axis=0,keepdims=True)

      dhidden2 = np.dot(dscores,W3.T)
      dhidden2[hidden_layer2 <=0 ] = .01 * dhidden2[hidden_layer2 <= 0]

      dW2 = np.dot(hidden_layer1.T, dhidden2)
      db2 = np.sum(dhidden2, axis=0, keepdims=True)
      
      dhidden1 = np.dot(dhidden2, W2.T)
      dhidden1[hidden_layer1 <= 0] = .01 * dhidden1[hidden_layer1 <= 0]
      
      
      dW1 = np.dot(X.T, dhidden1)
      db1 = np.sum(dhidden1, axis=0, keepdims=True)
      
     #include regularization
      dW3 += reg * W3
      dW2 += reg * W2
      dW1 += reg * W1
     
     #update weights
      W1 += -step_size * dW1
      b1 += -step_size * db1
      W2 += -step_size * dW2
      b2 += -step_size * db2
      W3 += -step_size * dW3
      b3 += -step_size * db3

#end training
end = time.time()
print("training time: ", end-start, " seconds")
#test

#feed test data
x_test = np.where(x_test > 0 , x_test, .01 * x_test)
hidden_layer1 = np.dot(x_test,W1) +b1
hidden_layer1 =  np.where(hidden_layer1 > 0, hidden_layer1, .01* hidden_layer1) 
hidden_layer2 = np.dot(hidden_layer1, W2) +b2
scores = np.dot(hidden_layer2, W3) + b3

predicted = np.argmax(scores,axis=1)
print("test accuracy: ", np.mean(predicted==y_test))

