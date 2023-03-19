import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla


seed=2397
rng=np.random.default_rng(seed)
maxit=10


#Order of GMRES polynomial
k=4

#First fit polynomial (without constant term) to the constant 1
eigs=np.array([-0.5,-0.2,0.2,0.3,0.4,0.5,1.0])


e=np.ones(len(eigs))
for it in range(0,maxit):
    #eigs=np.array(list(rng.uniform(-1,-0.1,4)) + list(rng.uniform(0.1,1.0,4)))
    V=np.zeros((len(eigs),k))
    for i in range(1,k+1):
        V[:,i-1]=eigs**i



    #Solve for the coefficients
    #using least-squares
    c=np.linalg.lstsq(np.diag(e)@V,e,rcond=None)[0]



    #Now sample polynomial for visualization
    nsamples=1000
    xs=np.linspace(-1.0,1.0,nsamples)
    ys=np.zeros(nsamples)
    conv=np.zeros(len(eigs))
    for i in range(1,k+1):
        ys+=c[i-1]*xs**i
        #Also calculate the convergence rate
        conv+=c[i-1]*eigs**i





    plt.plot(xs,1.0-ys)
    #Put markers on the eigenvalues
    plt.plot(eigs,np.zeros(len(eigs)),'o')
    plt.ylim(-2.0,2.0)
    plt.title(f"GMRES({k}) polynomial at eigenvalues of A and iteration {it}")

    #Draw whiskers extending +-1 from each eigenvalue to demonstrate the region of convergence
    #Do this in a single plot so that we can label them in the legend
    for eig,ei in zip(eigs,e):
        plt.plot([eig,eig],[-abs(ei),abs(ei)],color='black',linestyle='--')
    e=e-conv*e

    plt.legend(['Polynomial','Eigenvalues','Convergence region'])
    plt.savefig(f"ex03/{str(it).zfill(3)}.png")
    plt.close()


    #Add above plot to animation





