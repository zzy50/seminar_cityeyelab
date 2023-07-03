import numpy as np
from numpy.typing import NDArray

def sparse_categorical_crossentropy(y_pred: NDArray, y_true: NDArray):
    n_classes = y_pred.shape[-1]
    y_true_onehot = np.eye(n_classes)[y_true]
    ce = - np.sum(y_true_onehot * np.log(y_pred), axis=-1)
    return ce


def relu(x: NDArray):
	return np.maximum(0.0, x)


def relu_derivative(x: NDArray):
    return (x > 0).astype(float)


def softmax(x: NDArray):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def forward(x, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3


def backward(x, y_true, z1, a1, z2, a2, z3, a3, W2, W3):
    n_classes = a3.shape[-1]
    y_true_onehot = np.eye(n_classes)[y_true]
    dz3 = a3 - y_true_onehot
    # dz3 = sparse_categorical_crossentropy(a3, y_true)
    dW3 = np.dot(a2.T, dz3)
    db3 = np.sum(dz3, axis=0)
    
    da2 = np.dot(dz3, W3.T)
    dz2 = da2 * relu_derivative(z2)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0)
    
    return dW1, db1, dW2, db2, dW3, db3




def sgd_with_weight_decay(params, grads, learning_rate, weight_decay):
    for param, grad in zip(params, grads):
        param -= learning_rate * (grad + weight_decay * param)


def train(x, y_true, W1, b1, W2, b2, W3, b3, learning_rate, weight_decay):
    z1, a1, z2, a2, z3, a3 = forward(x, W1, b1, W2, b2, W3, b3)
    loss = sparse_categorical_crossentropy(a3, y_true)
    dW1, db1, dW2, db2, dW3, db3 = backward(x, y_true, z1, a1, z2, a2, z3, a3, W2, W3)

    params = [W1, b1, W2, b2, W3, b3]
    grads = [dW1, db1, dW2, db2, dW3, db3]
    sgd_with_weight_decay(params=params,
                          grads=grads,
                          learning_rate=learning_rate,
                          weight_decay=weight_decay)
    
    return loss


def fit(x, y_true, W1, b1, W2, b2, W3, b3, learning_rate, weight_decay, epochs):
    losses = []
    for epoch in range(epochs):
        loss = train(x=x,
                     y_true=y_true,
                     W1=W1,
                     b1=b1,
                     W2=W2,
                     b2=b2,
                     W3=W3,
                     b3=b3,
                     learning_rate=learning_rate,
                     weight_decay=weight_decay)
        losses.append(loss)
        print(f'Epoch {epoch + 1}/{epochs}: loss = {loss:.4f}')

        # 학습률 조정
        if epoch > 0 and loss > losses[-2]:
            learning_rate *= 0.5
            print(f'Reduce learning rate to {learning_rate:.4f}')

    return losses


# def main():
#     x = np.array([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9]])
#     x = np.reshape(x, -1)
#     y_true = np.array([1])

#     W1 = np.random.randn(9, 4)
#     b1 = np.zeros(4)
#     W2 = np.random.randn(4, 5)
#     b2 = np.zeros(5)
#     W3 = np.random.randn(5, 3)
#     b3 = np.zeros(3)

#     z1, a1, z2, a2, z3, a3 = forward(x, W1, b1, W2, b2, W3, b3)
#     dW1, db1, dW2, db2, dW3, db3 = backward(x, y_true, z1, a1, z2, a2, z3, a3, W2, W3)

#     print(dW1)
#     print(db1)
#     print(dW2)
#     print(db2)
#     print(dW3)
#     print(db3)

    # n_classes = a3.shape[-1]
    # y_true_onehot = np.eye(n_classes)[y_true]
    # dz3 = a3 - y_true_onehot