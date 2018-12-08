import numpy as np
def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    u = x.mean(1)

    return x-u[:,None]



def gcn(x, scale=1., bias=0.001):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    var = x.var(1)
    return scale*x/np.sqrt(bias+var)[:,None]


def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    f_u  = x.mean(0)
    return (x-f_u,xtest-f_u)

def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    U, S, V = np.linalg.svd(x.T.dot(x/x.shape[0])+(np.identity(x.shape[1])*bias))
    pca = U.dot(np.diag(1.0/np.sqrt(S))).dot(U.T)
    return (x.dot(pca),xtest.dot(pca))



def clean(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    xtrain, xtest = feature_zero_mean(gcn(sample_zero_mean(x)), gcn(sample_zero_mean(xtest)))
    #xtrain, xtest = zca(xtrain,xtest)
    xtrain        = np.reshape(xtrain, (x.shape[0],3, image_size,image_size))
    xtest         = np.reshape(xtest, (xtest.shape[0],3, image_size,image_size))
    return (xtrain,xtest)