import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla




#Order of GMRES polynomial
k=4

#First fit polynomial (without constant term) to the constant 1
eigs=np.array([-0.5,-0.3,0.2,0.3,0.4,0.5,1.0])
V=np.zeros((len(eigs),k))
for i in range(1,k+1):
    V[:,i-1]=eigs**i


e=np.ones(len(eigs))

#Solve for the coefficients
#using least-squares
c=np.linalg.lstsq(V,e,rcond=None)[0]



#Now sample polynomial for visualization
nsamples=1000
xs=np.linspace(-1.0,1.0,nsamples)
ys=np.zeros(nsamples)
conv=np.zeros(len(eigs))
for i in range(1,k+1):
    ys+=c[i-1]*xs**i
    #Also calculate the convergence rate
    conv+=c[i-1]*eigs**i


convrate=max(abs(1.0-conv))





plt.subplot(2,1,1)
plt.plot(xs,1.0-ys)
#Put markers on the eigenvalues
plt.plot(eigs,np.zeros(len(eigs)),'o')
plt.ylim(-2.0,2.0)
plt.title(f"GMRES{k} polynomial at eigenvalues of A")

#Draw whiskers extending +-1 from each eigenvalue to demonstrate the region of convergence
#Do this in a single plot so that we can label them in the legend
for eig in eigs:
    plt.plot([eig,eig],[-1.0,1.0],color='black',linestyle='--')

plt.legend(['Polynomial','Eigenvalues','Convergence region'])



#Draw a box around the eigenvalues with radius 1 and label this box "region of convergence"
#plt.gca().add_patch(matplotlib.patches.Rectangle((min(eigs),-1),2.0,2.0,fill=False))
#label the box
#plt.text(0.0,1.0,"Region of convergence",ha='center',va='bottom')

#Place computed convergence rate in top of plot as text for reference
plt.text(0.0,1.5,f"Convergence rate bound: {convrate:.2e}",ha='center',va='bottom')
#plt.text(0.0,-1.5,f"Convergence rate bound: {convrate:.2f}",ha='left',va='bottom')





seed=2397
rng=np.random.default_rng(seed)
D=np.diag(eigs)
Q,_=np.linalg.qr(rng.random((len(eigs),len(eigs))))
A=Q @ D @ Q.T
b=Q @ np.ones(len(eigs))

resl=[]
def callback(xk):
    resl.append(np.linalg.norm(A @ xk - b))


spla.gmres(A,b,callback=callback,restart=k,callback_type='x')

#Put GMRES convergence into a subplot
plt.subplot(2,1,2)

predicted=[]
for i in range(len(resl)):
    if convrate<1.0:
        predicted.append(np.linalg.norm(b)*(convrate**i))
    else:
        predicted.append(np.linalg.norm(b))

#Make room for plot title of bottom subplot
plt.subplots_adjust(hspace=0.5)
plt.title("GMRES Convergence")
plt.semilogy(resl)
plt.semilogy(predicted)
plt.legend(['Actual GMRES residual','Residual bound'])
plt.xlabel("Iteration")

plt.savefig('ex02.svg')
