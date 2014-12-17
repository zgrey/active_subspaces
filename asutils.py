import numpy as np
import matplotlib.pyplot as plt
import os

def normalize_uniform(X,xl,xu):
    M,m = X.shape
    xl = xl.reshape(1,m,order='F')
    xu = xu.reshape(1,m,order='F')
    XX = 2*(X - np.tile(xl,(M,1)))/(np.tile(xu,(M,1))-np.tile(xl,(M,1)))-1
    return XX.copy()

def normalize_gaussian(X,mu,C):
    M,m = X.shape
    mu = mu.reshape(1,m,order='F')
    L = np.linalg.cholesky(C)
    X0 = X - np.tile(mu,(M,1))
    XX = np.linalg.solve(L,X0.T)
    return XX.T.copy()

def full_index_set(n,d):
    if d == 1:
        I = np.array([[n]])
    else:
        II = full_index_set(n,d-1)
        m = II.shape[0]
        I = np.hstack((np.zeros((m,1)),II))
        for i in range(1,n+1):
            II = full_index_set(n-i,d-1)
            m = II.shape[0]
            T = np.hstack((i*np.ones((m,1)),II))
            I = np.vstack((I,T))
    return I
    
def index_set(n,d):
    I = np.zeros((1,d))
    for i in range(1,n+1):
        II = full_index_set(i,d)
        I = np.vstack((I,II))
    return I[:,::-1]

def lingrad(X,f):
    M,m = X.shape
    A = np.hstack((np.ones((M,1)), X))
    u = np.linalg.lstsq(A,f)[0]
    w = u[1:]/np.linalg.norm(u[1:])
    return w
    
def quadreg(Xsq,indices,f):
    M,m = indices.shape
    u = np.linalg.lstsq(Xsq,f)[0]
    b = u[1:m+1]
    A = np.zeros((m,m))
    for i in range(m+1,M):
        ind = indices[i,:]
        loc = np.nonzero(ind!=0)[0]
        if loc.size==1:
            A[loc,loc] = 2.0*u[i]
        elif loc.size==2:
            A[loc[0],loc[1]] = u[i]
            A[loc[1],loc[0]] = u[i]
        else:
            print 'error!'
    return b,A
    
def get_eigenpairs(b,A,gamma):
    m = b.shape[0]
    lam,W = np.linalg.eig(np.outer(b,b.T) + np.dot(A,np.dot(np.diagflat(gamma),A)))
    ind = np.argsort(lam)[::-1]
    lam = lam[ind]
    W = W[:,ind]
    W = W*np.tile(np.sign(W[0,:]),(m,1))
    return lam,W
    
def local_linear_gradients(X,f,XX):
    M,m = X.shape
    MM = XX.shape[0]
    G = np.zeros((MM,m))
    nloc = np.minimum(2*m,M)
    for i in range(MM):
        x = XX[i,:]
        A = np.sum((X - np.tile(x,(M,1)))**2,axis=1)
        ind = np.argsort(A)
        A = np.hstack((np.ones((nloc,1)), X[ind[:nloc],:]))
        u = np.linalg.lstsq(A,f[ind[:nloc]])[0]
        G[i,:] = u[1:].copy()
    return G
    
def quadratic_model_check(X,f,gamma,k,n_boot=1000):
    M,m = X.shape
    gamma = gamma.reshape((1,m))
    k_sub = np.minimum(k,m-1)
    
    indices = index_set(2,m)
    mI = indices.shape[0]
    Xsq = np.zeros((M,mI))
    for i in range(mI):
        ind = indices[i,:]
        Xsq[:,i] = np.prod(np.power(X,np.tile(ind,(M,1))),axis=1)
    
    # get regression coefficients
    b,A = quadreg(Xsq,indices,f)
    
    # compute eigenpairs
    lam,W = get_eigenpairs(b,A,gamma)
    
    # bootstrap
    sub_dist = np.zeros((m-1,n_boot))
    lam_boot = np.zeros((m,n_boot))
    ind = np.random.randint(M,size=(M,n_boot))
    for i in range(n_boot):
        b0,A0 = quadreg(Xsq[ind[:,i],:],indices,f[ind[:,i]])
        lam0,W0 = get_eigenpairs(b0,A0,gamma)
        lam_boot[:,i] = lam0
        for j in range(m-1):
            sub_dist[j,i] = np.linalg.norm(np.dot(W[:,:j+1].T,W0[:,j+1:]),ord=2)

    lam_br = np.zeros((k,2))
    sub_br = np.zeros((k_sub,3))
    for i in range(k):
        lam_sort = np.sort(lam_boot[i,:])
        lam_br[i,0] = lam_sort[np.floor(0.025*n_boot)]
        lam_br[i,1] = lam_sort[np.ceil(0.925*n_boot)]
    for i in range(k_sub):
        sub_sort = np.sort(sub_dist[i,:])
        sub_br[i,0] = sub_sort[np.floor(0.025*n_boot)]        
        sub_br[i,1] = np.mean(sub_sort)
        sub_br[i,2] = sub_sort[np.ceil(0.925*n_boot)]
    return lam[:k],W,lam_br,sub_br
    
    
def linear_model_check(X,f,n_boot=1000):
    M,m = X.shape
    w = lingrad(X,f)
    
    # bootstrap
    ind = np.random.randint(M,size=(M,n_boot))
    w_boot = np.zeros((m,n_boot))
    for i in range(n_boot): 
        w_boot[:,i] = lingrad(X[ind[:,i],:],f[ind[:,i]])
    return w,w_boot

def get_active_subspace(G,k,n_boot=1000):
    
    # set integers
    M,m = G.shape
    k_sub = np.minimum(k,m-1)
    
    # compute active subspace
    U,sig,W = np.linalg.svd(G,full_matrices=False)
    lam = (1.0/M)*(sig[:k]**2)
    W = W.T
    W = W*np.tile(np.sign(W[0,:]),(m,1))
    
    # bootstrap
    lam_boot = np.zeros((k,n_boot))
    sub_dist = np.zeros((k_sub,n_boot))
    ind = np.random.randint(M,size=(M,n_boot))
    for i in range(n_boot):
        U0,sig0,W0 = np.linalg.svd(G[ind[:,i],:],full_matrices=False)
        W0 = W0.T
        W0 = W0*np.tile(np.sign(W0[0,:]),(m,1))
        lam_boot[:,i] = (1.0/M)*(sig0[:k]**2)
        for j in range(k_sub):
            sub_dist[j,i] = np.linalg.norm(np.dot(W[:,:j+1].T,W0[:,j+1:]),ord=2)

    lam_br = np.zeros((k,2))
    sub_br = np.zeros((k_sub,3))
    for i in range(k):
        lam_sort = np.sort(lam_boot[i,:])
        lam_br[i,0] = lam_sort[np.floor(0.025*n_boot)]
        lam_br[i,1] = lam_sort[np.ceil(0.925*n_boot)]
    for i in range(k_sub):
        sub_sort = np.sort(sub_dist[i,:])
        sub_br[i,0] = sub_sort[np.floor(0.025*n_boot)]
        sub_br[i,1] = np.mean(sub_sort)
        sub_br[i,2] = sub_sort[np.ceil(0.925*n_boot)]
    return lam,W,lam_br,sub_br

def quadtest(X):
    M,m = X.shape
    B = np.random.normal(size=(m,m))
    Q = np.linalg.qr(B)[0]
    e = np.array([10**(-i) for i in range(1,m+1)])
    A = np.dot(Q,np.dot(np.diagflat(e),Q.T))
    f = np.zeros((M,1))
    for i in range(M):
        z = X[i,:]
        f[i] = 0.5*np.dot(z,np.dot(A,z.T))
        
    return f,e,Q,A

def sufficient_summary_plot(y,f,w,w_boot=None,in_labels=None,out_label=None):
    
    # make figs directory
    if not os.path.isdir('figs'):
        os.mkdir('figs')
    
    # set plot fonts
    myfont = {'family' : 'lucinda',
            'weight' : 'normal',
            'size'   : 14}
    plt.rc('font', **myfont)
    
    # check sizes of y and w
    m = w.shape[0]
    n = len(w.shape)    
    if n == 1:
        y1 = y
        w1 = w
    else:
        y1 = y[:,0]
        y2 = y[:,1]
        w1 = w[:,0]
        w2 = w[:,1]
    
    # set labels for plots
    if in_labels is None:
        in_labels = [str(i) for i in range(1,m+1)]
    if out_label is None:
        out_label = 'Output'
    
    plt.figure()
    if w_boot is not None:
        plt.plot(range(1,m+1),w_boot,color='0.7')
    plt.plot(range(1,m+1),w1,'ko-',markersize=12)
    plt.xlabel('Variable')
    plt.ylabel('Weights')
    plt.grid(True)
    if m<=10:
        plt.xticks(range(1,m+1),in_labels,rotation='vertical')
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.15)
    plt.axis([1,m,-1,1])
    figname = 'figs/ssp1_weights_' + out_label + '.eps'
    plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    
    plt.figure()
    plt.plot(y1,f,'bo',markersize=12)
    plt.xlabel('Active variable')
    plt.ylabel(out_label)
    plt.grid(True)
    figname = 'figs/ssp1_' + out_label + '.eps'
    plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
        
    if n==2:
        
        plt.figure()
        plt.plot(range(1,m+1),w1,'bo-',markersize=12,label='1')
        plt.plot(range(1,m+1),w2,'ro-',markersize=12,label='2')
        plt.xlabel('Variable')
        plt.ylabel('Weights')
        plt.grid(True)
        if m<=10:
            plt.xticks(range(1,m+1),in_labels,rotation='vertical')
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
        plt.axis([1,m,-1,1])
        plt.legend(loc='best')
        figname = 'figs/ssp2_weights_' + out_label + '.eps'
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
        
        plt.figure()
        plt.scatter(y1,y2,c=f,s=150.0,vmin=np.min(f),vmax=np.max(f))
        plt.xlabel('Active variable 1')
        plt.ylabel('Active variable 2')
        plt.title(out_label)
        figname = 'figs/ssp2_' + out_label + '.eps'
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    
    plt.show()
    
def plot_active_subspace(lam,W,lam_br=None,sub_br=None,in_labels=None,out_label=None):
    
    # make figs directory
    if not os.path.isdir('figs'):
        os.mkdir('figs')
    
    # set plot fonts
    myfont = {'family' : 'lucinda',
            'weight' : 'normal',
            'size'   : 14}
    plt.rc('font', **myfont)
    
    # number of variables
    k = lam.shape[0]
    m = W.shape[0]
    
    # set labels for plots
    if in_labels is None:
        in_labels = [str(i) for i in range(1,m+1)]
    if out_label is None:
        out_label = 'Output'
    
    plt.figure()
    plt.semilogy(range(1,k+1),lam,'ko-')
    if lam_br is not None:
        plt.fill_between(range(1,k+1),lam_br[:,0],lam_br[:,1],
            facecolor='0.7', interpolate=True)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalues')
    plt.title(out_label)
    plt.grid(True)
    figname = 'figs/evals_' + out_label + '.eps'
    plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    
    plt.figure()
    plt.plot(range(1,m+1),W[:,0],'bo-',markersize=12,label='1')
    plt.plot(range(1,m+1),W[:,1],'ro-',markersize=12,label='2')
    plt.plot(range(1,m+1),W[:,2],'go-',markersize=12,label='3')
    plt.xlabel('Variable')
    plt.ylabel('Eigenvectors')
    plt.grid(True)
    if m<=10:
        plt.xticks(range(1,m+1),in_labels,rotation='vertical')
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.15)
    plt.axis([1,m,-1,1])
    plt.legend(loc='best')
    figname = 'figs/evecs_' + out_label + '.eps'
    plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    
    if sub_br is not None:
        kk = sub_br.shape[0]
        plt.figure()
        plt.semilogy(range(1,kk+1),sub_br[:,1],'ko-',markersize=12)
        plt.fill_between(range(1,kk+1),sub_br[:,0],sub_br[:,2],
            facecolor='0.7', interpolate=True)
        plt.xlabel('Subspace dimension')
        plt.ylabel('Subspace distance')
        plt.grid(True)
        plt.xticks(range(1,k+1))
        plt.axis([1,kk,0.001,1])
        figname = 'figs/subspace_' + out_label + '.eps'
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    
    plt.show()
    
    
    
