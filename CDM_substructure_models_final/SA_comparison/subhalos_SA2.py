import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from scipy import integrate
from numpy.random import *
from scipy import interpolate
from scipy import optimize
from scipy import stats
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from numpy.polynomial.hermite import hermgauss
from scipy import special


c=2.99792e+10
G=6.6742e-8
Mpc=3.086e+24
kpc = Mpc/1000.
pc = kpc/1000.
cm       = 1.
km       = 1.e5*cm
s        = 1.
gram     = 1.
c        = 2.99792e+10*cm/s
OmegaL=0.692
OmegaC=0.25793
OmegaB=0.049150
Omegam=0.1415
Omegar=0.0
Omega0=1.0
pOmega=[OmegaC+OmegaB,Omegar,OmegaL]
H0=67.81
h=H0/100.0
OmegaM=Omegam*pow(h,-2)
H0=H0*(1.0e+5)/Mpc
rhocrit0=3*pow(H0,2)*pow(8.0*np.pi*G,-1)
Msolar=1.988435e+33

def fc(x):
    return np.log(1+x)-x*pow(1+x,-1)

def conc200(M200,z):
    alpha_cMz_1=1.7543-0.2766*(1+z)+0.02039*pow(1+z,2)
    beta_cMz_1=0.2753+0.00351*(1+z)-0.3038*pow(1+z,0.0269)
    gamma_cMz_1=-0.01537+0.02102*pow(1+z,-0.1475)
    c_Mz_1=pow(10,alpha_cMz_1+beta_cMz_1*np.log10(M200/Msolar) \
               *(1+gamma_cMz_1*pow(np.log10(M200/Msolar),2)))
    alpha_cMz_2=1.3081-0.1078*(1+z)+0.00398*pow(1+z,2)
    beta_cMz_2=0.0223-0.0944*pow(1+z,-0.3907)
    c_Mz_2=pow(10,alpha_cMz_2+beta_cMz_2*np.log10(M200/Msolar))
    return np.where(z<=4,c_Mz_1,c_Mz_2)

def g(z):
    return (OmegaB+OmegaC)*(1.+z)**3+Omegar*(1+z)**4+OmegaL

def rhocrit(z):
    return 3.0*pow(Hz(z),2)*pow(np.pi*8.0*G,-1)

def Mvir_from_M200(M200, z):
    gz=g(z)
    c200 = conc200(M200,z)
    r200 = (3.0*M200/(4*np.pi*200*rhocrit0*gz))**(1./3.)
    rs = r200/c200
    fc200=fc(c200)
    rhos = M200/(4*np.pi*rs**3*fc200)
    pOmega = [OmegaC+OmegaB,Omegar,OmegaL]
    Dc = Delc(Omegaz(pOmega,z)-1.)
    rvir = optimize.fsolve(lambda r: 3.*(rs/r)**3*fc(r/rs)*rhos-Dc*rhocrit0*gz,r200)
    Mvir = 4*np.pi*rs**3*rhos*fc(rvir/rs)
    return Mvir

def Mvir_from_M200_fit(M200, z):
    a1 = 0.5116
    a2 = -0.4283
    a3 = -3.13e-3
    a4 = -3.52e-5
    pOmega=[OmegaC+OmegaB,Omegar,OmegaL]
    Oz=Omegaz(pOmega,z)
    def ffunc(x):
        return np.power(x,3.0)*(np.log(1.0+1.0/x)-1.0/(1.0+x))
    def xfunc(f):
        p = a2 + a3*np.log(f) + a4*np.power(np.log(f),2.0)
        return np.power(a1*np.power(f,2.0*p)+(3.0/4.0)**2,-0.5)+2.0*f
    return Delc(Oz-1)/200.0*M200 \
        *np.power(conc200(M200,z)*xfunc(Delc(Oz-1)/200.0*ffunc(1.0/conc200(M200,z))),-3.0)

def growthD(z):
    Omega_Lz=OmegaL*pow(OmegaL+OmegaM*pow(1+z,3),-1)
    Omega_Mz=1-Omega_Lz
    phiz=pow(Omega_Mz,4.0/7.0)-Omega_Lz+(1+Omega_Mz/2.0)*(1+Omega_Lz/70.0)
    phi0=pow(OmegaM,4.0/7.0)-OmegaL+(1+OmegaM/2.0)*(1+OmegaL/70.0)
    return (Omega_Mz/OmegaM)*(phi0/phiz)*pow(1+z,-1)

def xi(M):
    return pow(M*pow((1e+10)*pow(h,-1),-1),-1)

def sigmaMz(M,z):
    return growthD(z)*22.26*pow(xi(M),0.292)*pow(1+1.53*pow(xi(M),0.275)+3.36*pow(xi(M),0.198),-1)

def dOdz(z):
    return -OmegaL*3*Omegam*pow(h,-2)*pow(1+z,2)*pow(OmegaL+Omegam*pow(h,-2)*pow(1+z,3),-2)

def dDdz(z):
    Omega_Lz=OmegaL*pow(OmegaL+Omegam*pow(h,-2)*pow(1+z,3),-1)
    Omega_Mz=1-Omega_Lz
    phiz=pow(Omega_Mz,4.0/7.0)-Omega_Lz+(1+Omega_Mz/2.0)*(1+Omega_Lz/70.0)
    phi0=pow(Omegam*pow(h,-2),4.0/7.0)-OmegaL+(1+Omegam*pow(h,-2)/2.0)*(1+OmegaL/70.0)

    dphidz=dOdz(z)*((-4.0/7.0)*pow(Omega_Mz,-3.0/7.0)+(Omega_Mz-Omega_Lz)/140.0+(1.0/70.0)-(3.0/2.0))
    return (phi0/OmegaM)*(-dOdz(z)*pow(phiz*(1+z),-1)-Omega_Mz*(dphidz*(1+z)+phiz)*pow(phiz,-2)*pow(1+z,-2))

def Mzi(M0,z):
    a=1.686*np.sqrt(2.0/np.pi)*dDdz(0)+1.0
    zf=-0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
    q=4.137*pow(zf,-0.9476)
    fM0=pow(pow(sigmaMz(M0/q,0),2)-pow(sigmaMz(M0,0),2),-0.5)
    return M0*pow(1+z,a*fM0)*np.exp(-fM0*z)

def Mzzi(M0,z,zi):
    Mzi0=Mzi(M0,zi)
    zf=-0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
    q=4.137*pow(zf,-0.9476)
    fMzi=pow(pow(sigmaMz(Mzi0/q,zi),2)-pow(sigmaMz(Mzi0,zi),2),-0.5)
    alpha=fMzi*(1.686*np.sqrt(2.0/np.pi)*pow(growthD(zi),-2)*dDdz(zi)+1)
    beta=-fMzi
    return Mzi0*pow(1+z-zi,alpha)*np.exp(beta*(z-zi))

def Hz(z):
    return H0*np.sqrt(OmegaL+OmegaM*pow(1+z,3))

def Omegaz(p,x):
    E=p[0]*pow(1+x,3)+p[1]*pow(1+x,2)+p[2]
    return p[0]*pow(1+x,3)*pow(E,-1)

def Delc(x):
    return 18*pow(np.pi,2)+(82*x)-39*pow(x,2)

# Replaced Mvir_from_M200 with Mvir_from_M200_fit
def dMdz(M0,z,zi):
    Mzi0=Mzi(M0,zi)
    zf=-0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
    q=4.137*pow(zf,-0.9476)
    fMzi=pow(pow(sigmaMz(Mzi0/q,zi),2)-pow(sigmaMz(Mzi0,zi),2),-0.5)
    alpha=fMzi*(1.686*np.sqrt(2.0/np.pi)*pow(growthD(zi),-2)*dDdz(zi)+1)
    beta=-fMzi
    Mzzidef=Mzi0*pow(1+z-zi,alpha)*np.exp(beta*(z-zi))
    Mzzivir=Mvir_from_M200_fit(Mzzidef*Msolar,z)
    return (beta+alpha*pow(1+z-zi,-1))*Mzzivir/Msolar

def dsdm(M,z):
    s=pow(sigmaMz(M,z),2)
    dsdsigma=2*sigmaMz(M,z)
    dxidm=(-1.0)*pow(10,10)*pow(h,-1)*pow(M,-2)
    dsigmadxi=sigmaMz(M,z)*(0.292*pow(xi(M),-1)-(0.275*1.53*pow(xi(M),-0.725)+0.198*3.36*pow(xi(M),-0.802))*pow(1+1.53*pow(xi(M),0.275)+3.36*pow(xi(M),0.198),-1))
    return dsdsigma*dsigmadxi*dxidm

def delc_Y11(z):
    return 1.686*pow(growthD(z),-1)

def s_Y11(M):
    return pow(sigmaMz(M,0),2)

def Ffunc(dela,sig1,sig2):
    return pow(2*np.pi,-0.5)*dela*pow(sig2-sig1,-1.5)

def Gfunc(dela,sig1,sig2):
    #parameter namne is taken in a consistent way with that of Ffunc
    G0=0.57
    gamma1=0.38
    gamma2=-0.01
    s1=np.sqrt(sig1)
    s2=np.sqrt(sig2)
    return G0*pow(s2/s1,gamma1)*pow(dela/s1,gamma2)

#return eq.14 in Yang+11
def Ffunc_Yang(delc1,delc2,sig1,sig2):
    return pow(2*np.pi,-0.5)*(delc2-delc1)*pow(sig2-sig1,-1.5)*np.exp(-pow(delc2-delc1,2)*pow(2*(sig2-sig1),-1))

# Main function to calculate subhalo accretion history
#Mhost should be in Msolar unit
def Na_calc(ma, zacc, Mhost, z0=0, N_herm=200, Nrand=1000,sigmafac=1,Namodel=3):
    """ Returns Na, Eq. (3) of Yang et al. (2011) """
    #parameter "model" added, 1,2,3 for model 1,2,3 in Yang et al.2011
    zacc_2d=zacc.reshape(np.alen(zacc),1)
    M200_0 = Mzzi(Mhost,zacc_2d,z0)
    logM200_0 = np.log10(M200_0)
    if N_herm==1:
        sigmalogM200=0.12-0.15*np.log10(M200_0/Mhost)
        logM200=logM200_0+sigmafac*sigmalogM200
        M200=10**logM200
    else:
        xxi,wwi = hermgauss(N_herm)
        xxi = xxi.reshape(np.alen(xxi),1,1)
        wwi = wwi.reshape(np.alen(wwi),1,1)
        #eq.21 in Yang+2011
        sigmalogM200 = 0.12-0.15*np.log10(M200_0/Mhost)
        logM200 = np.sqrt(2)*sigmalogM200*xxi+logM200_0
        M200 = 10**logM200
    mmax=np.minimum(M200,Mhost/2.0)
    Mmax=np.minimum(M200_0+mmax,Mhost)
    if Namodel==3:
        zlist=zacc_2d*np.linspace(1,0,Nrand)
        iMmax=np.argmin(np.abs(Mzzi(Mhost,zlist,z0)-Mmax),axis=-1)
        z_Max=zlist[np.arange(np.alen(zlist)),iMmax]
        z_Max_3d=z_Max.reshape(N_herm,np.alen(zlist),1)
        delcM=delc_Y11(z_Max_3d)
        delca=delc_Y11(zacc_2d)
        sM = s_Y11(Mmax)
        sa = s_Y11(ma)
        xmax = pow((delca-delcM),2)*pow((2*(s_Y11(mmax)-sM)),-1)
        normB = special.gamma(0.5)*special.gammainc(0.5,xmax)/np.sqrt(np.pi)
        #those reside in the exponential part of eq.14
        Phi = Ffunc_Yang(delcM,delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
    elif Namodel==1:
        delca=delc_Y11(zacc_2d)
        sM=s_Y11(M200)
        sa = s_Y11(ma)
        xmin=s_Y11(mmax)-s_Y11(M200)
        normB=pow(2*np.pi,-0.5)*delca*(2.)*pow(xmin,-0.5)*special.hyp2f1(0.5,0.,1.5,-sM/xmin)
        Phi = Ffunc(delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
    else:
        delca=delc_Y11(zacc_2d)
        sM=s_Y11(M200)
        sa = s_Y11(ma)
        xmin=s_Y11(mmax)-s_Y11(M200)
        normB=pow(2*np.pi,-0.5)*delca*0.57*pow(delca/np.sqrt(sM),-0.01)*(2./(1.-0.38))*pow(sM,-0.38/2.)*pow(xmin,0.5*(0.38-1.))*special.hyp2f1(0.5*(1-0.38),-0.38/2.,0.5*(3.-0.38),-sM/xmin)
        Phi = Ffunc(delca,sM,sa)*Gfunc(delca,sM,sa)/normB*np.heaviside(mmax-ma,0)

    if N_herm==1:
        F2t = np.nan_to_num(Phi)
        F2=F2t.reshape((len(zacc_2d),len(ma)))

    else:
        F2 = np.sum(np.nan_to_num(Phi)*wwi/np.sqrt(np.pi),axis=0)
    Na = F2*dsdm(ma,0)*dMdz(Mhost,zacc_2d,z0)*(1+zacc_2d)
    return Na


#################################################################################################

##My modifications:


#I added these two functions for the c calculation from a given Mvir (for the dm/dt(r) toy model relation)


def c_from_Mvir(Mvir,z):
    alpha_cMz_1=1.62774-0.2458*(1+z)+0.01716*pow(1+z,2)
    beta_cMz_1=1.66079+0.00359*(1+z)-1.6901*pow(1+z,0.00417)
    gamma_cMz_1= -0.02049+0.0253*pow(1+z,-0.1044)
    c_Mz_1=pow(10,alpha_cMz_1+beta_cMz_1*np.log10(Mvir/Msolar) \
               *(1+gamma_cMz_1*pow(np.log10(Mvir/Msolar),2)))
    alpha_cMz_2=1.226-0.1009*(1+z)+0.00378*pow(1+z,2)
    beta_cMz_2=0.008634-0.08814*pow(1+z,-0.58816)
    c_Mz_2=pow(10,alpha_cMz_2+beta_cMz_2*np.log10(Mvir/Msolar))
    return np.where(z<=4,c_Mz_1,c_Mz_2)

#Correa model from:https://arxiv.org/pdf/1502.00391.pdf ->relations (19),(20)

##################################################################################################################



#I added this function for the calculation of dm/dt average for all cases.

def msolve(m,z,r,M0,model):
    
    """The mean dm/dt formula is the same as before (without spatial information) with an extra term for 
    the dm/dt-distance dependence.
    
    dm/dt average at different distance from host's center with possible cases:
    
    1)No spatial information for model=1,the r variable is not contributing to the result.
    2)First spatial fitting for spatial mdot function for model=2.
    3)New spatial fitting for spatial mdot function for model=3.
    4)Combination of toy model at distance/Rvir<0.2 and new spatial fitting (model 3) for higher distance/Rvir values.  
    5)Toy spatial model for model=5.
    """
    
    def Mzvir(z):
        Mz200=Mzzi(M0,z,0)
        Mz200solar=Mz200*Msolar
        Mvirsolar=Mvir_from_M200(Mz200solar,z)
        return Mvirsolar/Msolar

    
    def AMz(z):
        log10a=(-0.0003*np.log10(Mzvir(z))+0.02)*z+(0.011*np.log10(Mzvir(z))-0.354)
        return pow(10,log10a)

    def zetaMz(z):
        return (0.00012*np.log10(Mzvir(z))-0.0033)*z+(-0.0011*np.log10(Mzvir(z))+0.026)

    def tdynz(z):
        Oz_z=Omegaz(pOmega,z)
        return 1.628*pow(h,-1)*pow(Delc(Oz_z-1)/178.0,-0.5)*pow(Hz(z)/H0,-1)*(86400*365*(1e+9))
    
    def mdot_r(r,z,M0,model):
        if model==2:
            return 10**(0.2-1.75*0.3*r)
        elif model==3:
            a=1.34
            b=1.39
            c=2.87
            eq=a-b*(1+(1-r)**c)**(-1/c)
            return 10**eq
        elif model==4:
            M0=1e12
            if r>0.2:
                a=1.34
                b=1.39
                c=2.87
                eq=a-b*(1+(1-r)**c)**(-1/c)
                return 10**eq
            else:
                c=c_from_Mvir(M0*Msolar,z)
                return np.log((np.log(1+c)-c/(1+c))/(np.log(1+r*c)-(c*r)/(1+c*r)))     
        elif model==5:
            c=c_from_Mvir(M0*Msolar,z)
            return np.log((np.log(1+c)-c/(1+c))/(np.log(1+r*c)-(c*r)/(1+c*r)))
        else:
            return 1

    if model==1:
        return AMz(z)*(m/tdynz(z))*pow(m/Mzvir(z),zetaMz(z))*pow(Hz(z)*(1+z),-1)
    else:
        if z<0:
            z=0
        return mdot_r(r,z,M0,model)*AMz(z)*(m/tdynz(z))*pow(m/Mzvir(z),zetaMz(z))*pow(Hz(z)*(1+z),-1)
    

    
#Spatial distribution weights (dN/dr):

def radial_weights(r,M,z):
    
    """Probability of distance/Rvir value for a subhalo based on the NFW density profile
    at an early stage of a host halo (for example z=7) before the tidal forces change the
    spatial distribution of subhalos.
    """
    #c=1   #This an extreme case of minimum concentration of a host halo.
    c=c_from_Mvir(Mvir=M*Msolar,z=z)
    M_all=np.log(1+c)-c/(1+c)
    Prob=(r*c**2)/((1+c*r)**2)
    return Prob/M_all
    
####################################################################################################################


def rs_rhos_calc_new(M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=100, sigmalogc=0.128, N_herm=5, logmamin=-6, logmamax=None,sigmafac=0,N_hermNa=200,Namodel=3,N_r=5,rmin=0.01,rmax=0.99,mmodel=3,NFW=True,logr=False):

    """New modifications at rs_rhos_calc() function:
    1)Defining the range of possible positions of subhalos by r_min,r_max variables in units of distance/Rvir(host). 
    2)Defining the number of position's values in the given range by N_r variable.
    3)There is the option of choosing distance/Rvir values in logspace by the argument logr=True.
    4)Defining the mass loss model according to the mass loss function given above by mmodel variable.
    5)This function returns an additional array named rshh,which contains the positions of the subhalos in units of 
    distance/Rvir(host).
    
    """
    
    
    zdist = np.arange(redshift+dz,zmax+dz,dz)
    if logmamax==None:
        logmamax = np.log10(0.1*M0)
    ma200 = np.logspace(logmamin,logmamax,N_ma)
    rs_acc = np.zeros((N_r,len(zdist),N_herm,len(ma200)))
    rhos_acc = np.zeros((N_r,len(zdist),N_herm,len(ma200)))
    rs_z0 = np.zeros((N_r,len(zdist),N_herm,len(ma200)))
    rhos_z0 = np.zeros((N_r,len(zdist),N_herm,len(ma200)))
    ct_z0 = np.zeros((N_r,len(zdist),N_herm,len(ma200)))
    survive=np.zeros((N_r,len(zdist),N_herm,len(ma200)))
    m0_matrix = np.zeros((N_r,len(zdist),N_herm,len(ma200)))
    rshh = np.zeros((N_r,len(zdist),N_herm,len(ma200)))
    Oz_0=Omegaz(pOmega,redshift)


    if logr==False:
        r_edges=np.linspace(rmin,rmax,N_r+1)
    else:
        r_edges=np.logspace(np.log10(rmin),np.log10(rmax),N_r+1)
    r=(r_edges[1:]+r_edges[:-1])/2.
    dr=(r_edges[1:]-r_edges[:-1])
    m=np.zeros((N_r,N_ma))
    for iz in range(len(zdist)):
        ma = Mvir_from_M200(ma200*Msolar,zdist[iz])/Msolar
        Oz = Omegaz(pOmega,zdist[iz])
        zcalc = np.linspace(zdist[iz],redshift,100)
        

        for i in range(0,N_r):
            rshh[i,iz] = r[i]*np.ones((N_herm,1)) 
            sol = odeint(msolve,ma,zcalc,args=(r[i],M0,mmodel))
            m0 = sol[-1]
            m0 = np.where(m0<10**(logmamin-1),1e-8,m0)
            m[i,:]=m0
            
            
            c200sub = conc200(ma200*Msolar,zdist[iz])
            rvirsub = pow(3*ma*Msolar*pow(rhocrit0*g(zdist[iz])*Delc(Oz-1)*4*np.pi,-1),1.0/3.0)
            r200sub = pow(3*ma200*Msolar*pow(rhocrit0*g(zdist[iz])*200*4*np.pi,-1),1.0/3.0)
            c_mz = c200sub*rvirsub/r200sub
            x1,w1 = hermgauss(N_herm)
            x1 = x1.reshape(np.alen(x1),1)
            w1 = w1.reshape(np.alen(w1),1)
            log10c_sub = np.log10(c_mz) +np.sqrt(2)*sigmalogc*x1
            c_sub = pow(10.0,log10c_sub)
            rs_acc[i,iz] = rvirsub/c_sub
            rhos_acc[i,iz] = ma*Msolar/(4*np.pi*rs_acc[i,iz]**3*fc(c_sub))
            rmax_acc = rs_acc[i,iz]*2.163
            Vmax_acc = np.sqrt(rhos_acc[i,iz]*4*np.pi*G/4.625)*rs_acc[i,iz]
            Vmax_z0 = Vmax_acc*(pow(2,0.4)*pow(m0/ma,0.3)*pow(1+m0/ma,-0.4))
            rmax_z0 = rmax_acc*(pow(2,-0.3)*pow(m0/ma,0.4)*pow(1+m0/ma,0.3))
            rs_z0[i,iz] = rmax_z0/2.163
            rhos_z0[i,iz] = (4.625/(4*np.pi*G))*pow(Vmax_z0/rs_z0[i,iz],2)
            ctemp = np.linspace(0,100,1000)
            ftemp = interp1d(fc(ctemp),ctemp,fill_value='extrapolate')
            ct_z0[i,iz] = ftemp(m0*Msolar/(4*np.pi*rhos_z0[i,iz]*rs_z0[i,iz]**3))
            survive[i,iz] = np.where(ct_z0[i,iz]>0.77,1,0)
            m0_matrix[i,iz] = m0*np.ones((N_herm,1))

    Na = Na_calc(ma,zdist,M0,z0=redshift,N_herm=N_hermNa,Nrand=1000,sigmafac=sigmafac,Namodel=Namodel)
    Na_total = integrate.simps(integrate.simps(Na,x=np.log(ma)),x=np.log(1+zdist))
    weight = Na/(1.0+zdist.reshape(np.alen(zdist),1))
    weight = weight/np.sum(weight)*Na_total
    weight = (weight.reshape((len(zdist),1,len(ma))))*w1/np.sqrt(np.pi)
    weight = np.array(N_r*[weight])
    ma200_matrix = ma200*np.ones((N_r,len(zdist),N_herm,1))

    
    if mmodel!=1 and NFW==True:
        
        #The radial weights are radial_weights(radius,M=M0)*dr in order to be unitless 
        #probability (the function radial_weights() is dN/dr):    
        w_r = radial_weights(r,M=M0,z=redshift) #*dr
        weight=(weight.T*w_r).T
    #else:
    #    weight=(weight.T*dr).T
    ma200_matrix = ma200_matrix.reshape(len(zdist)*N_herm*len(ma200)*N_r)
    m0_matrix = m0_matrix.reshape(len(zdist)*N_herm*len(ma200)*N_r)
    rs_acc = rs_acc.reshape(len(zdist)*N_herm*len(ma200)*N_r)
    rhos_acc = rhos_acc.reshape(len(zdist)*N_herm*len(ma200)*N_r)
    rs_z0 = rs_z0.reshape(len(zdist)*N_herm*len(ma200)*N_r)
    rhos_z0 = rhos_z0.reshape(len(zdist)*N_herm*len(ma200)*N_r)
    ct_z0 = ct_z0.reshape(len(zdist)*N_herm*len(ma200)*N_r)
    weight = weight.reshape(len(zdist)*N_herm*len(ma200)*N_r)
    survive = (survive==1).reshape(len(zdist)*N_herm*len(ma200)*N_r)
    rshh = rshh.reshape(len(zdist)*N_herm*len(ma200)*N_r)
    z_acc = (zdist.reshape(len(zdist),1,1))*np.ones((1,N_herm,N_ma))
    z_acc = np.array(N_r*[z_acc])
    z_acc = z_acc.reshape(len(zdist)*N_herm*N_ma*N_r)

    return ma200_matrix, rs_acc/kpc, rhos_acc/(Msolar/pc**3), \
        m0_matrix, rs_z0/kpc, rhos_z0/(Msolar/pc**3), ct_z0, \
        weight, survive ,rshh ,z_acc



####################################################################################################################

#Similar function to rs_rhos_calc_new() with the difference being that the substructure is calculated at
#distance/Rvir in [radius-dr/2,radius+dr/2] with radius and dr given by the user: 

def rs_rhos_calc_one_r(M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=100, sigmalogc=0.128, N_herm=5, logmamin=-6, logmamax=None,sigmafac=0,N_hermNa=200,Namodel=2,mmodel=3,radius=0.5,dr=0.01):
    
    """Main parameters of rs_rhos_calc_new() function:
    1)Defining the distance from the host's center in units of distance/Rvir(host) with the variable radius. 
    2)Defining the distance range around the radius with the variable dr in units of distance/Rvir(host). 
    3)Defining the mass loss model according to the mass loss function given above by mmodel variable.
    4)This function returns the substructure in [radius-dr/2,radius+dr/2] of a host halo with mass M0 at a given redshift. 
    
    """
    
    zdist = np.arange(redshift+dz,zmax+dz,dz)
    if logmamax==None:
        logmamax = np.log10(0.1*M0)
    ma200 = np.logspace(logmamin,logmamax,N_ma)
    rs_acc = np.zeros((len(zdist),N_herm,len(ma200)))
    rhos_acc = np.zeros((len(zdist),N_herm,len(ma200)))
    rs_z0 = np.zeros((len(zdist),N_herm,len(ma200)))
    rhos_z0 = np.zeros((len(zdist),N_herm,len(ma200)))
    ct_z0 = np.zeros((len(zdist),N_herm,len(ma200)))
    survive=np.zeros((len(zdist),N_herm,len(ma200)))
    m0_matrix = np.zeros((len(zdist),N_herm,len(ma200)))
    rshh = np.zeros((len(zdist),N_herm,len(ma200)))
    Oz_0=Omegaz(pOmega,redshift)


    
    for iz in range(len(zdist)):
        ma = Mvir_from_M200(ma200*Msolar,zdist[iz])/Msolar
        Oz = Omegaz(pOmega,zdist[iz])
        zcalc = np.linspace(zdist[iz],redshift,100)
        sol = odeint(msolve,ma,zcalc,args=(radius,M0,mmodel))
        m0 = np.maximum(sol[-1],0.)

            
        c200sub = conc200(ma200*Msolar,zdist[iz])
        rvirsub = pow(3*ma*Msolar*pow(rhocrit0*g(zdist[iz])*Delc(Oz-1)*4*np.pi,-1),1.0/3.0)
        r200sub = pow(3*ma200*Msolar*pow(rhocrit0*g(zdist[iz])*200*4*np.pi,-1),1.0/3.0)
        c_mz = c200sub*rvirsub/r200sub
        x1,w1 = hermgauss(N_herm)
        x1 = x1.reshape(np.alen(x1),1)
        w1 = w1.reshape(np.alen(w1),1)
        log10c_sub = np.log10(c_mz) +np.sqrt(2)*sigmalogc*x1
        c_sub = pow(10.0,log10c_sub)
        rs_acc[iz] = rvirsub/c_sub
        rhos_acc[iz] = ma*Msolar/(4*np.pi*rs_acc[iz]**3*fc(c_sub))
        rmax_acc = rs_acc[iz]*2.163
        Vmax_acc = np.sqrt(rhos_acc[iz]*4*np.pi*G/4.625)*rs_acc[iz]
        Vmax_z0 = Vmax_acc*(pow(2,0.4)*pow(m0/ma,0.3)*pow(1+m0/ma,-0.4))
        rmax_z0 = rmax_acc*(pow(2,-0.3)*pow(m0/ma,0.4)*pow(1+m0/ma,0.3))
        rs_z0[iz] = rmax_z0/2.163
        rhos_z0[iz] = (4.625/(4*np.pi*G))*pow(Vmax_z0/rs_z0[iz],2)
        ctemp = np.linspace(0,100,1000)
        ftemp = interp1d(fc(ctemp),ctemp,fill_value='extrapolate')
        ct_z0[iz] = ftemp(m0*Msolar/(4*np.pi*rhos_z0[iz]*rs_z0[iz]**3))
        survive[iz] = np.where(ct_z0[iz]>0.77,1,0)
        m0_matrix[iz] = m0*np.ones((N_herm,1))

    
    Na = Na_calc(ma,zdist,M0,z0=0,N_herm=N_hermNa,Nrand=1000,sigmafac=sigmafac,Namodel=Namodel)
    Na_total = integrate.simps(integrate.simps(Na,x=np.log(ma)),x=np.log(1+zdist))
    weight = Na/(1.0+zdist.reshape(np.alen(zdist),1))
    weight = weight/np.sum(weight)*Na_total
    weight = (weight.reshape((len(zdist),1,len(ma))))*w1/np.sqrt(np.pi)
    
    #The radial weight are radial_weights(radius,M=M0)*dr in order to be unitless 
    #probability (the function radial_weights() is dN/dr):
    
    #weight = radial_weights(radius,M=M0)*dr*weight
    weight = weight.reshape(len(zdist)*N_herm*len(ma200))

    
    ma200_matrix = ma200*np.ones((len(zdist),N_herm,1))
    ma200_matrix = ma200_matrix.reshape(len(zdist)*N_herm*len(ma200))
    m0_matrix = m0_matrix.reshape(len(zdist)*N_herm*len(ma200))
    rs_acc = rs_acc.reshape(len(zdist)*N_herm*len(ma200))
    rhos_acc = rhos_acc.reshape(len(zdist)*N_herm*len(ma200))
    rs_z0 = rs_z0.reshape(len(zdist)*N_herm*len(ma200))
    rhos_z0 = rhos_z0.reshape(len(zdist)*N_herm*len(ma200))
    ct_z0 = ct_z0.reshape(len(zdist)*N_herm*len(ma200))
    survive = (survive==1).reshape(len(zdist)*N_herm*len(ma200))
    rshh = rshh.reshape(len(zdist)*N_herm*len(ma200))

    return ma200_matrix, rs_acc/kpc, rhos_acc/(Msolar/pc**3), \
        m0_matrix, rs_z0/kpc, rhos_z0/(Msolar/pc**3), ct_z0, \
        weight, survive 

