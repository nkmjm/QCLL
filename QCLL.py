
import numpy as np
import random
import math
import numpy.matlib
import scipy.optimize as opt


#####
##### Count sketch (for qi)
#####

def makeC(d, n):
    # C: [k x n mat] for countSketch computation

    C = np.zeros([d, n]) # init 
    s = random.choices([-1, 1], k=n)
    h = random.choices(range(d), k=n)
    for i in range(n):
        C[h[i], i] = s[i]
        
    return C
    
def qiCountSketch(Xs, Cs):
    # Xs: [nQbits][2 x 1 vec]
    # Cs: [nQbits][d x n mat] made by func 'makeC'

    n = Xs[0].shape[0]
    for i in range(len(Xs)):
        C = Cs[i]
        t_prd = np.dot(C, Xs[i])
        # fft
        fftprd = np.fft.fft(t_prd, axis=0)
        if i > 0:
            # element-wise product between previous prd and current prd
            fftprd = prev_fftprd* fftprd
        prev_fftprd = fftprd
    
    # inverse fft
    prd = np.fft.ifft(fftprd, axis=0)
    
    # Get only real
    prd_real = prd.real
    
    return prd_real

###
### Other func

def outerProduct(Xs):

    tX = Xs[0]
    for i in range(1,len(Xs)):
        tX = np.kron(Xs[i], tX)

    return tX

def normalize(X):

    cX = X - np.mean(X)
    maxX = np.max(cX)
    minX  = np.min(cX)
    d = maxX - np.abs(minX)
    if d >= 0:
        cX = cX/maxX
    else:
        cX = cX/np.abs(minX)

    return cX

    
def makeV(inDT):
    # inDT: [nQbits][nSample vec]
    # Vs: [nQbits][2 x 1 vec]

    Vs = []
    for i in range(len(inDT)):
        tDT = inDT[i]
        tV = np.array([[tDT], [np.sqrt(1 - (tDT**2))]])
        Vs.append(tV)

    return Vs 

def makeW(thetas):
    # thetas: [nParams vec] (radian)
    # Ws: [nParams][2 x 1 vec]

    Ws = []
    for i in range(len(thetas)):
        theta = thetas[i]
        tw = np.array([[math.cos(theta)],[math.sin(theta)]])
        Ws.append(tw)

    return Ws

def makeW_for_grad(thetas, i):

    Ws = []
    for j in range(len(thetas)):
        theta = thetas[j]
        if j == i:
            tw = np.array([[-math.sin(theta)],[math.cos(theta)]])
        else:
            tw = np.array([[math.cos(theta)],[math.sin(theta)]])
        Ws.append(tw)

    return Ws

def predictY(init_thetas_and_a_and_b, inVs, Cs_V, Cs_Ws):

    thetas = init_thetas_and_a_and_b[:-2]
    a = init_thetas_and_a_and_b[len(init_thetas_and_a_and_b)-2]
    b = init_thetas_and_a_and_b[len(init_thetas_and_a_and_b)-1]
    nSamples = len(inVs)
    nQbits = len(inVs[0])
    Ws = makeW(thetas)
    
    ys = []
    us = []
    for i in range(nSamples):
        tVs = inVs[i]
        # Perform qiCountSketch (to compute inner-product)
        csV = qiCountSketch(tVs, Cs_V)
        sub_u = []
        for k in range(len(Cs_Ws)):
            csW = qiCountSketch(Ws, Cs_Ws[k])
            u = np.dot(csW.transpose(), csV)
            sub_u.append(u[0][0])
        # Compute Predict (y)
        y = a* np.sum(np.abs(sub_u)**2) + b
        # Make list y
        ys.append(y)
        us.append(sub_u)

    return np.array(ys), np.array(us)

def predictY_kclass(thetas_and_a_and_b, inVs, Cs_V, Cs_Ws):
    
    nClasses = len(Cs_Ws)
    theta_ind = 2*nClasses
    us = []
    for k in range(nClasses):
        a_ind = 2*nClasses - 2*k
        b_ind = 2*nClasses - (2*k + 1)
        a = thetas_and_a_and_b[len(thetas_and_a_and_b) - a_ind]
        b = thetas_and_a_and_b[len(thetas_and_a_and_b) - b_ind]
        # Set new_init_thetas_and_a_and_b
        new_thetas_and_a_and_b = list(thetas_and_a_and_b[:-theta_ind])
        new_thetas_and_a_and_b.append(a)
        new_thetas_and_a_and_b.append(b)
        # Predict ys
        prd_y, u = predictY(new_thetas_and_a_and_b, inVs, Cs_V, Cs_Ws[k])
        prd_y = np.reshape(prd_y, [len(prd_y), 1])
        us.append(u)
        # concat prd_ys
        if k == 0:
            prd_ys = prd_y
        else:
            prd_ys = np.concatenate([prd_ys, prd_y], axis=1)
            
    return prd_ys, np.array(us)

def computeLoss(thetas_and_a_and_b, ys, inVs, Cs_V, Cs_Ws):

    thetas = thetas_and_a_and_b[:-2]
    a = thetas_and_a_and_b[len(thetas_and_a_and_b)-2]
    b = thetas_and_a_and_b[len(thetas_and_a_and_b)-1]
    prd_y, _ = predictY(thetas_and_a_and_b, inVs, Cs_V, Cs_Ws)
    res = ys - prd_y
    error = np.sum(res**2)
    print(error)

    return error

def computeGradient(thetas_and_a_and_b, ys, inVs, Cs_V, Cs_Ws):

    thetas = thetas_and_a_and_b[:-2]
    a = thetas_and_a_and_b[len(thetas_and_a_and_b)-2]
    b = thetas_and_a_and_b[len(thetas_and_a_and_b)-1]
    prd_ys, us = predictY(thetas_and_a_and_b, inVs, Cs_V, Cs_Ws)
    # Set csgradW
    csgradWs = []
    for k in range(len(Cs_Ws)): # nOut
        csgradW = []
        for i in range(len(thetas)): # nThetas
            gradWs  = makeW_for_grad(thetas, i)
            csgradW.append(qiCountSketch(gradWs, Cs_Ws[k]))
        csgradW = np.squeeze(csgradW) # param x d
        csgradWs.append(csgradW)
    # Set csV
    csV = []
    for j in range(len(inVs)):
        tVs = inVs[j]
        csV.append(qiCountSketch(tVs, Cs_V))
    csV = np.squeeze(csV) # sample x d
    ### grad: thetas
    elm = np.zeros([len(thetas), len(us)])
    for k in range(len(csgradWs)):
        us_grad = np.dot(csgradWs[k], csV.transpose()) # param x sample
        us_rep   = numpy.matlib.repmat(us[:,k], len(thetas), 1) # param x sample
        elm = elm + (us_rep* us_grad) # param x sample
    grad_thetas = -4* np.sum(elm* np.matlib.repmat((ys - prd_ys), len(thetas), 1), axis=1) # sample x param
    grad_thetas = list(grad_thetas)
    ### grad: a
    grad_a = np.sum(2*(prd_ys - ys)* np.squeeze(np.sum((us**2), axis=1)) )
    ### grad: b
    grad_b = np.sum(2*(prd_ys - ys))
    ### grad: concat
    grad = grad_thetas
    grad.append(grad_a)
    grad.append(grad_b)

    return np.array(grad)

def softmax(Q):
    
    max_q = np.max(Q)
    E = np.exp(Q - max_q)
    t_out = E/np.sum(E)
    return t_out

def computeLoss_kclass(thetas_and_a_and_b, ys, inVs, Cs_V, Cs_Ws):
    
    prd_ys, _ = predictY_kclass(thetas_and_a_and_b, inVs, Cs_V, Cs_Ws)
    nSample = prd_ys.shape[0]
    pE = []
    for i in range(nSample):
        pE.append(softmax(prd_ys[i,:])) 
    loss = (-1)* ys* np.log(pE) # loss: softmax cross entropy
    print(np.sum(loss))
    
    return np.sum(loss)

def computeGradient_kclass(thetas_and_a_and_b, ys, inVs, Cs_V, Cs_Ws):
    
    # params
    nClasses = len(Cs_Ws)
    nOut = len(Cs_Ws[0])
    nSamples = ys.shape[0]

    prd_ys, us = predictY_kclass(thetas_and_a_and_b, inVs, Cs_V, Cs_Ws)

    # probability
    pE = []
    for i in range(len(prd_ys)):
        pE.append(softmax(prd_ys[i,:]))
    gradBase = pE - ys # nSamples x nClasses

    ### grad: a and b
    
    usum = np.sum(us**2, axis=2)
    grad_a = np.sum(gradBase* usum.transpose(), axis=0) 
    grad_b = np.sum(gradBase, axis=0)
    
    ### grad: thetas
    
    thetas = thetas_and_a_and_b[:-nClasses*2] # nClasses x (2: a and b)
    nThetas = len(thetas)
    
    a_vec = []
    for k in range(nClasses):
        a_ind = 2*nClasses - 2*k
        a_vec.append(thetas_and_a_and_b[len(thetas_and_a_and_b) - a_ind])
    a_vec = np.reshape(np.squeeze(a_vec), nClasses, 1)
    a_mat = numpy.matlib.repmat(a_vec, nSamples, 1)
    
    # Set csV
    csV = []
    for j in range(len(inVs)):
        tVs = inVs[j]
        csV.append(qiCountSketch(tVs, Cs_V))
    csV = np.squeeze(csV) # sample x d
    
    # grad_theta
    grad_theta = []
    for i in range(len(thetas)):       
        gradWs = makeW_for_grad(thetas, i)
        # u_grad
        for oi in range(nOut):
            u_grad=[]
            for k in range(nClasses):
                csgradWs = np.squeeze(qiCountSketch(gradWs, Cs_Ws[k][oi]))
                u_grad.append(np.dot(csgradWs, csV.transpose()))
            u_grad = np.array(u_grad)
            if oi == 0:
                elm = np.squeeze(us[:, :, oi])* u_grad
            else:
                elm = elm + (np.squeeze(us[:, :, oi])* u_grad) # nClasses x nSamples
        # grad_theta      
        t_grad_theta = np.sum(gradBase* 2* a_mat* elm.transpose()) # gradBase: nSamples x nClasses
        grad_theta.append(t_grad_theta)
    
    # grad
    grad = grad_theta
    for i in range(nClasses):
        grad.append(grad_a[i])
        grad.append(grad_b[i])
    
    return grad

def computeProb_kclass(thetas_and_a_and_b, inVs, Cs_V, Cs_Ws):
    
    prd_ys, _ = predictY_kclass(thetas_and_a_and_b, inVs, Cs_V, Cs_Ws)

    # probability of each class
    prob = []
    for i in range(prd_ys.shape[0]):
        t_out = softmax(prd_ys[i,:])
        prob.append(t_out)     
    return prob



###
### Regression

class regression():
    
    # Input
    # --------------------------------------------------------
    # nQbits      : the number of qubits
    # nParams   : the number of parameter theta
    # d            : the number of dimensions for low dimensional side of count sketch matrix
    # nOut       : the number of the dimensions of the output vectors
    #
    # (.fit)
    # inDT        : input data [nQbits][nSamples vec]
    # ys           : training samples [nSamples vec]
    #
    # (.predict)
    # inDT        : input data [nQbits][nSamples vec]
    # paramType: parameter types. 'init' (initial) of 'opt' (optimized)
    #
    # (.loss)
    # inDT        : input data [nQbits][nSamples vec]
    # ys           : training samples [nSamples vec]
    # paramType: parameter types. 'init' (initial) of 'opt' (optimized)
    # --------------------------------------------------------
    
    # Output
    # --------------------------------------------------------
    # (.predict)
    # prd_ys: predicted training samples
    #
    # (.loss)
    # loss:  sum of squared error
    # --------------------------------------------------------
    
    
    def __init__(self, nQbits, nParams, d, nOut):

        self.nQbits = nQbits
        self.nParams = nParams
        self.d = d
        self. nOut = nOut

    def fit(self, inDT, ys):
        
        nQbits = self.nQbits
        nParams = self.nParams
        d = self.d
        nOut = self.nOut
        
        nSample = len(inDT)

        # for V
        inVs = [] # [nSample][nQbits][2 x 1 vec]
        for i in range(nSample):
            Vs = makeV(inDT[i])
            inVs.append(Vs)

        # Set Cs (count sketch mat.) for Vs & Ws: shared across samples
        Cs_V = []
        for i in range(nQbits):
            Cs_V.append(makeC(d, 2)) 
        Cs_Ws = []
        for k in range(nOut):
            Cs_W = []
            for i in range(nParams):
                Cs_W.append(makeC(d, 2))
            Cs_Ws.append(Cs_W)

        # Set init_thetas for Ws
        init_thetas_and_a_and_b = list(math.pi* np.random.rand(nParams))
        init_thetas_and_a_and_b.append(1) # append a: scale
        init_thetas_and_a_and_b.append(0) # append b: intercept

        # Optimization
        result = opt.minimize(computeLoss, init_thetas_and_a_and_b, args=(ys, inVs, Cs_V, Cs_Ws), jac=computeGradient, method='SLSQP', tol=10**(-5), options={'maxiter':100})

        # Out items 
        # count sketch
        self.Cs_V = Cs_V
        self.Cs_Ws = Cs_Ws
        # result
        self.init_thetas_and_a_and_b = init_thetas_and_a_and_b
        self.opt_thetas_and_a_and_b = result.x
    
        return self
    
    def predict(self, inDT, paramType='opt'):
        
        # count sketch
        Cs_V = self.Cs_V
        Cs_Ws = self.Cs_Ws
        # result
        init_thetas_and_a_and_b = self.init_thetas_and_a_and_b
        opt_thetas_and_a_and_b = self.opt_thetas_and_a_and_b
        
        nSample = len(inDT)
        
        # for V
        inVs = [] # [nSample][nQbits][2 x 1 vec]
        for i in range(nSample):
            Vs = makeV(inDT[i])
            inVs.append(Vs)

        if paramType is 'init':
            prd_ys, _ = predictY(init_thetas_and_a_and_b, inVs, Cs_V, Cs_Ws)
        elif paramType is 'opt':
            prd_ys, _ = predictY(opt_thetas_and_a_and_b, inVs, Cs_V, Cs_Ws)
            
        return prd_ys
        
    def loss(self, inDT, ys, paramType='opt'):
        
        # count sketch
        Cs_V = self.Cs_V
        Cs_Ws = self.Cs_Ws
        # result
        init_thetas_and_a_and_b = self.init_thetas_and_a_and_b
        opt_thetas_and_a_and_b = self.opt_thetas_and_a_and_b
        
        nSample = len(inDT)
        
        # for V
        inVs = [] # [nSample][nQbits][2 x 1 vec]
        for i in range(nSample):
            Vs = makeV(inDT[i])
            inVs.append(Vs)
        
        if paramType is 'init':
            loss = computeLoss(init_thetas_and_a_and_b, ys, inVs, Cs_V, Cs_Ws)
        elif paramType is 'opt':
            loss = computeLoss(opt_thetas_and_a_and_b, ys, inVs, Cs_V, Cs_Ws)
        
        return loss   


###
### Classification

class classification():
    
    # Input
    # --------------------------------------------------------
    # nQbits      : the number of qubits
    # nParams   : the number of parameter theta
    # d            : the number of dimensions for low dimensional side of count sketch matrix
    # nOut       : the number of the dimensions of the output vectors
    #
    # (.fit)
    # inDT        : input data [nQbits][nSamples vec]
    # ys           : training samples [nSamples vec]
    #
    # (.probability)
    # inDT        : input data [nQbits][nSamples vec]
    # paramType: parameter types. 'init' (initial) of 'opt' (optimized)
    #
    # (.loss)
    # inDT        : input data [nQbits][nSamples vec]
    # ys           : training samples [nSamples vec]
    # paramType: parameter types. 'init' (initial) of 'opt' (optimized)
    # --------------------------------------------------------
    
    # Output
    # --------------------------------------------------------
    # (.probability)
    # prob: probability for classes
    #
    # (.loss)
    # loss:  softmax cross entropy
    # --------------------------------------------------------
    
    def __init__(self, nQbits, nParams, d, nOut):

        self.nQbits = nQbits
        self.nParams = nParams
        self.d = d
        self.nOut = nOut

    def fit(self, inDT, ys):

        nQbits = self.nQbits
        nParams = self.nParams
        d = self.d
        nOut = self.nOut

        nClasses = int(inDT.shape[1]/nQbits) # 2 for negoro data

        # Reset nSample
        nSample = len(inDT)

        # for V
        inVs = [] # [nSample][nQbits][2 x 1 vec]
        for i in range(nSample):
            Vs = makeV(inDT[i])
            inVs.append(Vs)

        # Set Cs (count sketch mat.) for Vs & Ws: shared across samples
        Cs_V = []
        for i in range(nQbits*nClasses):
            Cs_V.append(makeC(d, 2))

        Cs_Wss = []
        for k in range(nClasses):
            Cs_Ws = []
            for j in range(nOut):
                Cs_W = []
                for i in range(nParams):
                    Cs_W.append(makeC(d, 2))
                Cs_Ws.append(Cs_W)
            Cs_Wss.append(Cs_Ws)

        # Set init_thetas for Ws
        init_thetas_and_a_and_b = list(math.pi* np.random.rand(nParams))
        for k in range(nClasses):
            init_thetas_and_a_and_b.append(1) # append a
            init_thetas_and_a_and_b.append(0) # append b
            
        # Optimization 
        result = opt.minimize(computeLoss_kclass, init_thetas_and_a_and_b, args=(ys, inVs, Cs_V, Cs_Wss), jac=computeGradient_kclass, method='SLSQP', tol=10**(-5), options={'maxiter':100})
          
        # Out items 
        # count sketch
        self.Cs_V = Cs_V
        self.Cs_Wss = Cs_Wss
        # result
        self.init_thetas_and_a_and_b = init_thetas_and_a_and_b
        self.opt_thetas_and_a_and_b = result.x

        return self
    
    def probability(self, inDT, paramType='opt'):
        
        # count sketch
        Cs_V = self.Cs_V
        Cs_Wss = self.Cs_Wss
        # result
        init_thetas_and_a_and_b = self.init_thetas_and_a_and_b
        opt_thetas_and_a_and_b = self.opt_thetas_and_a_and_b
        
        nSample = len(inDT)
        
        # inVs for grids
        inVs = [] # [nSample][nQbits][2 x 1 vec]
        for i in range(nSample):
            Vs = makeV(inDT[i])
            inVs.append(Vs)

        if paramType is 'init':
            prob = computeProb_kclass(init_thetas_and_a_and_b, inVs, Cs_V, Cs_Wss)
        elif paramType is 'opt':
            prob = computeProb_kclass(opt_thetas_and_a_and_b, inVs, Cs_V, Cs_Wss)
        
        return prob
    
    def loss(self, inDT, ys, paramType='opt'):
        
        # count sketch
        Cs_V = self.Cs_V
        Cs_Wss = self.Cs_Wss
        # result
        init_thetas_and_a_and_b = self.init_thetas_and_a_and_b
        opt_thetas_and_a_and_b = self.opt_thetas_and_a_and_b
        
        nSample = len(inDT)
        
        # inVs for grids
        inVs = [] # [nSample][nQbits][2 x 1 vec]
        for i in range(nSample):
            Vs = makeV(inDT[i])
            inVs.append(Vs)

        if paramType is 'init':
            loss = computeLoss_kclass(init_thetas_and_a_and_b, ys, inVs, Cs_V, Cs_Wss)
        elif paramType is 'opt':
            loss = computeLoss_kclass(opt_thetas_and_a_and_b, ys, inVs, Cs_V, Cs_Wss)

        return loss
