import numpy as np
import pickle
from mlxtend.data import loadlocal_mnist


X, y = loadlocal_mnist(
        images_path="/home/andrey/datasets/mnist/t10k-images-idx3-ubyte", 
        labels_path="/home/andrey/datasets/mnist/t10k-labels-idx1-ubyte")


with open('b1.pickle', 'rb') as f: 
    b1 = pickle.load(f)

with open('b2.pickle', 'rb') as f:
    b2 = pickle.load(f)

with open('W1.pickle', 'rb') as f:
    W1 = pickle.load(f)

with open('W2.pickle', 'rb') as f:
    W2 = pickle.load(f)


def sigmoid(x):             
    return 1.0 / (1.0 + np.exp(-x))

X = X * (1.0 / 255)

b1 = np.array(b1)
b2 = np.array(b2)
W1 = np.array(W1)
W2 = np.array(W2)

W1 = W1.reshape(-1, 16)
W2 = W2.reshape(-1, 10) 

W1T = W1.T

w1n1 = W1T[0]
w1n2 = W1T[1]
w1n3 = W1T[2]
w1n4 = W1T[3]
w1n5 = W1T[4]
w1n6 = W1T[5]
w1n7 = W1T[6]
w1n8 = W1T[7]
w1n9 = W1T[8]
w1n10 = W1T[9]
w1n11 = W1T[10]
w1n12 = W1T[11]
w1n13 = W1T[12]
w1n14 = W1T[13]
w1n15 = W1T[14]
w1n16 = W1T[15]

W2T = W2.T

w2n1 = W2T[0]
w2n2 = W2T[1]
w2n3 = W2T[2]
w2n4 = W2T[3]
w2n5 = W2T[4]
w2n6 = W2T[5]
w2n7 = W2T[6]
w2n8 = W2T[7]
w2n9 = W2T[8]
w2n10 = W2T[9]

def forward(x0):
    h0 = np.array([
        sigmoid(np.sum(x0 * w1n1)),
        sigmoid(np.sum(x0 * w1n2)),
        sigmoid(np.sum(x0 * w1n3)),
        sigmoid(np.sum(x0 * w1n4)),
        sigmoid(np.sum(x0 * w1n5)),
        sigmoid(np.sum(x0 * w1n6)),
        sigmoid(np.sum(x0 * w1n7)),
        sigmoid(np.sum(x0 * w1n8)),
        sigmoid(np.sum(x0 * w1n9)),
        sigmoid(np.sum(x0 * w1n10)),
        sigmoid(np.sum(x0 * w1n11)),
        sigmoid(np.sum(x0 * w1n12)),
        sigmoid(np.sum(x0 * w1n13)),
        sigmoid(np.sum(x0 * w1n14)),
        sigmoid(np.sum(x0 * w1n15)),
        sigmoid(np.sum(x0 * w1n16))
    ])
    
    z0 = np.array([
        sigmoid(np.sum(h0 * w2n1)),
        sigmoid(np.sum(h0 * w2n2)),
        sigmoid(np.sum(h0 * w2n3)),
        sigmoid(np.sum(h0 * w2n4)),
        sigmoid(np.sum(h0 * w2n5)),
        sigmoid(np.sum(h0 * w2n6)),
        sigmoid(np.sum(h0 * w2n7)),
        sigmoid(np.sum(h0 * w2n8)),
        sigmoid(np.sum(h0 * w2n9)),
        sigmoid(np.sum(h0 * w2n10))
    ])

    return z0



np.set_printoptions(precision=3, suppress=True)

total = 0
stats = np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.int32)

for idx in range(0, 10000):
    h = forward(X[idx])
    yp = h.argmax()

    if yp != y[idx]:
        total += 1
        stats[yp] += 1

print("                    0     1     2     3     4     5     6     7     8     9")
print("stats per number: ", stats / total)
print("accuracy ", 100 - total / 100, "%")
print("ERR T P  0     1     2     3     4     5     6     7     8     9")
for idx in range(0, 16):
    h = forward(X[idx])
    yp = h.argmax()

    print("   " if y[idx] == yp else " * ", y[idx], yp, h)


## Matrix approach


h = sigmoid(X.dot(W1) + b1)
p = sigmoid(h.dot(W2) + b2)

print((np.sum(y == np.argmax(p, axis=1)) / len(y)))

