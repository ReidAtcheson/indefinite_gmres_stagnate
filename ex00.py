import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


seed=23084
rng=np.random.default_rng(seed)
m=128
d=rng.choice([1,3],size=m)
D=np.diag(d)
Q,_=la.qr(rng.uniform(-1,1,(m,m)))
A= Q @ D @ Q.T
b=rng.uniform(-1,1,m)


it=0
resl=[]
def callback(xk):
    global it
    res=np.linalg.norm(b-A@xk)
    resl.append(res)
    print(f"it={it} res={res}")
    it=it+1
spla.gmres(A,b,callback=callback,callback_type='x',restart=1,maxiter=100)


plt.semilogy(resl)
plt.title("GMRES(1) Convergence")
plt.ylabel("Residual")
plt.xlabel("Iteration")

seed=23084
rng=np.random.default_rng(seed)
m=128
d=rng.choice([-1,1],size=m)
D=np.diag(d)
Q,_=la.qr(rng.uniform(-1,1,(m,m)))
A= Q @ D @ Q.T
b=rng.uniform(-1,1,m)


it=0
resl=[]
def callback(xk):
    global it
    res=np.linalg.norm(b-A@xk)
    resl.append(res)
    print(f"it={it} res={res}")
    it=it+1
spla.gmres(A,b,callback=callback,callback_type='x',restart=1,maxiter=100)


plt.semilogy(resl)
plt.legend(["Eigenvalue clusters=[1,3] cond(A)=3","Eigenvalue clusters=[-1,1] cond(A)=1"])

plt.savefig("ex00.svg")

