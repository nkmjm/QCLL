
import numpy as np
import math
import numpy.matlib


#####
##### Make data
#####

def makeInDT_regression(funcType, nCopy, nSample, with_noise = False, noizeScale = 0.05, equal_interval = False):
    # funcType: 'x2', 'e_x', 'sin_x', 'abs_x'
    
    if equal_interval is False:
        x = np.random.uniform(-1, 1, nSample)
    elif equal_interval is True:
        x = (np.array(range(nSample)) - int(nSample/2)) /int(nSample/2)
    
    if with_noise is False:
        if funcType is 'x2':
            y = x**2
        elif funcType is 'e_x':
            y = math.e**x
        elif funcType is 'sin_x':
            y = np.array([math.sin(math.pi* xi) for xi in x])
        elif funcType is 'abs_x':
            y = np.abs(x)
        elif funcType is 'x_plot':
            x = (np.array(range(100))-50)/50
            y = np.nan

    elif with_noise is True:
        ex = noizeScale* np.random.randn(nSample)
        if funcType is 'x2':
            y = x**2 + ex
        elif funcType is 'e_x':
            y = math.e**x + ex
        elif funcType is 'sin_x':
            y = np.array([math.sin(math.pi* xi) for xi in x]) + ex
        elif funcType is 'abs_x':
            y = np.abs(x) + ex
            
    # copy x
    X = numpy.matlib.repmat(x, nCopy, 1)
    X = X.transpose()

    return X, y, nSample

        
def makeInDT_classification(nSample):

    x0 = []
    x1 = []
    while len(x0) + len(x1) < 2*nSample:
        tx1, tx2 = np.random.uniform(-1, 1, 2)
        r = np.sqrt(tx1**2 + tx2**2)
        if r <= 0.4:
            if len(x0) < nSample:
                x0.append([tx1, tx2])
        elif (r >= 0.6) & (r <= 1):
            if len(x1) < nSample:
                x1.append([tx1, tx2])

    return np.array(x0), np.array(x1)

def copyInDT_classification(x0, x1, nCopy):

    X0 = numpy.matlib.repmat(x0, 1, nCopy)
    X1 = numpy.matlib.repmat(x1, 1, nCopy)
    X  = np.concatenate([X0, X1], axis=0)
    nSample = len(x0)
    class0 = [1,0]
    class1 = [0,1]
    class0s = numpy.matlib.repmat(class0, nSample, 1)
    class1s = numpy.matlib.repmat(class1, nSample, 1)
    y = np.concatenate([class0s, class1s], axis=0)

    return X, y

def makeInDT_grid():

    a = np.meshgrid(range(-12,13),range(-12, 13))
    a = np.array(a, dtype='float')
    a = a*0.08
    x = np.reshape(a[0], [625, 1])
    y = np.reshape(a[1], [625, 1])
    xy = np.concatenate([x, y], axis=1)

    return xy



