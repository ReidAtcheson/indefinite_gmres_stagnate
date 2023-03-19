import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


k=1

#First fit polynomial (without constant term) to the constant 1
eigs=np.array([0.2,0.3,0.4,0.5,1.0])
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


convrate=np.linalg.norm(1.0-conv)
print(1.0-conv)
print(convrate)



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

#Place computed convergence rate in bottom of plot as text for reference
plt.text(0.0,-1.5,f"Convergence rate: {convrate:.2f}",ha='center',va='bottom')


plt.savefig('ex02.svg')
