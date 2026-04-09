from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1] # C
    num_train = X.shape[0] # N
    for i in range(num_train):
        scores = X[i].dot(W) #(D,) * (D, C) = (C,)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        p[y[i]] -= 1  # gradient of the loss with respect to scores
        dW += np.outer(X[i], p)  # contribution to the gradient from all classes

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W  # gradient of the regularization term
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    S = X.dot(W) # (N, D) * (D, C) = (N, C)
    S -= np.max(S,axis=1,keepdims=True) # numeric stability
    p = np.exp(S)
    p /= np.sum(p, axis=1, keepdims=True) # (N, C)
    logp = np.log(p)
    loss = -np.sum(logp[np.arange(X.shape[0]), y]) / X.shape[0] + reg * np.sum(W * W) #用到了高级索引[np.arange(X.shape[0]), y]，返回(N,)的数组，包含每个样本对应正确类别的log概率
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    p[np.arange(X.shape[0]), y] -= 1 # (N, C)，每个样本对应正确类别的概率减1
    dW = X.T.dot(p) / X.shape[0] + 2 * reg * W # (D, N) * (N, C) = (D, C)

    return loss, dW
