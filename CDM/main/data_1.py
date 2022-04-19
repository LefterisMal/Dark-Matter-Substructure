from subhalos_latest2 import *

M0=1e12
N_ma=400
zmax=7
N_r=100

logmamin=-6
mmodel=np.array([1,2,3,4,5])
rmin=0.0001
logr=True
redshift=0
dz=0.1

def data_saving(M0=M0,N_ma=N_ma,zmax=zmax,N_r=N_r,logmamin=logmamin,mmodel=mmodel, \
           redshift=redshift,dz=dz,rmin=rmin,logr=logr):
    
    for i in mmodel:
        if i==1:
            N_r_c=1
        else:
            N_r_c=N_r
        
        ma200, rs_a, rhos_a, m0, rs0, rhos0, ct0, weight, survive,r,zacc  = rs_rhos_calc_new(M0=M0, \
        logmamin=logmamin,N_ma=N_ma, zmax=zmax,N_r=N_r_c,mmodel=i, rmin=rmin,logr=logr,dz=dz)

        ma200  *= Msolar
        m0     *= Msolar
        rs_a   *= kpc
        rs0    *= kpc
        rhos_a *= Msolar/pc**3
        rhos0  *= Msolar/pc**3

        np.savetxt('data/01/ma200_'+str(i)+'.txt',ma200)
        np.savetxt('data/01/m0_'+str(i)+'.txt',m0)
        np.savetxt('data/01/rs_a_'+str(i)+'.txt',rs_a)
        np.savetxt('data/01/rhos_a_'+str(i)+'.txt',rhos_a)
        np.savetxt('data/01/rs0_'+str(i)+'.txt',rs0)
        np.savetxt('data/01/rhos0_'+str(i)+'.txt',rhos0)
        np.savetxt('data/01/ct0_'+str(i)+'.txt',ct0)
        np.savetxt('data/01/weight_'+str(i)+'.txt',weight)
        np.savetxt('data/01/survive_'+str(i)+'.txt',survive)
        np.savetxt('data/01/r_'+str(i)+'.txt',r)
        np.savetxt('data/01/zacc_'+str(i)+'.txt',zacc)

data_saving()
