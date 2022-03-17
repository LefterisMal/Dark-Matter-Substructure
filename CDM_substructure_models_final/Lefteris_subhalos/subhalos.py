import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy import integrate
from numpy.random import *
from scipy import interpolate
from scipy import optimize
from scipy import stats
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from numpy.polynomial.hermite import hermgauss
from scipy import special

#from numba import jit

cm       = 1.
km       = 1.e5*cm
s        = 1.
gram     = 1.
c        = 2.99792e+10*cm/s
G        = 6.6742e-8*cm**3/gram/s**2
Mpc      = 3.086e+24*cm
kpc      = Mpc/1000.
pc       = kpc/1000.
OmegaL   = 0.692
OmegaC   = 0.25793
OmegaB   = 0.049150
Omegam   = 0.1415
Omegar   = 0.0
Omega0   = 1.0
pOmega   = [OmegaC+OmegaB,Omegar,OmegaL]
H0       = 67.81*km/s/Mpc
h        = H0/(100.0*km/s/Mpc)
OmegaM   = Omegam*pow(h,-2)
rhocrit0 = 3*pow(H0,2)*pow(8.0*np.pi*G,-1)
Msolar   = 1.988435e+33*gram
GeV      = 1.7827e-24*gram
degree   = np.pi/180.


class subhalos:

    def __init__(self, filename_sigma=None):

        if filename_sigma==None:
            filename_sigma='tab_sigma_Sk_kcut1d1.dat'
        M_dummy,sigma_dummy,dlnsigmadlnM_dummy \
            = np.loadtxt('data/'+filename_sigma,unpack=True)
        self.sigma_int = interp1d(np.log(M_dummy),np.log(sigma_dummy))
        self.dsdM_int = interp1d(np.log(M_dummy),dlnsigmadlnM_dummy)

    def fc(self, x):
        return np.log(1+x)-x*pow(1+x,-1)

    def conc200(self, M200, z, model='Prada'):
        if(model=='Prada'):
            c200 = self.conc200_Prada(M200/Msolar,z)
        elif(model=='Correa'):
            c200 = self.conc200_Correa(M200,z)
        return c200

    def conc200_Correa(self, M200, z):
        alpha_cMz_1=1.7543-0.2766*(1+z)+0.02039*pow(1+z,2)
        beta_cMz_1=0.2753+0.00351*(1+z)-0.3038*pow(1+z,0.0269)
        gamma_cMz_1=-0.01537+0.02102*pow(1+z,-0.1475)
        c_Mz_1=pow(10,alpha_cMz_1+beta_cMz_1*np.log10(M200/Msolar) \
                   *(1+gamma_cMz_1*pow(np.log10(M200/Msolar),2)))
        alpha_cMz_2=1.3081-0.1078*(1+z)+0.00398*pow(1+z,2)
        beta_cMz_2=0.0223-0.0944*pow(1+z,-0.3907)
        c_Mz_2=pow(10,alpha_cMz_2+beta_cMz_2*np.log10(M200/Msolar))
        return np.where(z<=4,c_Mz_1,c_Mz_2)

    def conc200_Prada(self, M, z):
        def cmin(x):
            return 3.681+(5.033-3.681)*(np.arctan(6.948*(x-0.424))/np.pi+1./2.)
        def sigmininv(x):
            return 1.047+(1.646-1.047)*(np.arctan(7.386*(x-0.526))/np.pi+1./2.)
        def B00(x):
            return  cmin(x)/cmin((OmegaL/OmegaM)**(1./3.))
        def B01(x):
            return sigmininv(x)/sigmininv((OmegaL/OmegaM)**(1/3))
        def crlC(sigp):
            return 2.881*((sigp/1.257)**1.022+1)*np.exp(0.06/sigp**2)
        def tmpc(sigma,z):
            return B00((OmegaL/OmegaM)**(1./3.)/(1.+z))*crlC(sigma*B01((OmegaL/OmegaM)**(1./3.)/(1.+z)))
        def Dzfit(z):
            OmgLz=OmegaL/(OmegaL+OmegaM*(1.+z)**3)
            OmgMz=1.-OmgLz
            Psi=OmgMz**(4./7.)-OmgLz+(1.+OmgMz/2.)*(1.+OmgLz/70.)
            Psi0=OmegaM**(4./7.)-OmegaL+(1.+OmegaM/2.)*(1.+OmegaL/70.)
            return OmgMz/OmegaM*Psi0/Psi/(1.+z)
        return tmpc(np.exp(self.sigma_int(np.log(M)))*Dzfit(z),z)

    def g(self, z):
        return (OmegaB+OmegaC)*(1.+z)**3+Omegar*(1+z)**4+OmegaL

    def rhocrit(self, z):
        return 3.0*pow(self.Hz(z),2)*pow(np.pi*8.0*G,-1)

    def Mvir_from_M200(self, M200, z):
        gz = self.g(z)
        c200 = self.conc200(M200,z)
        r200 = (3.0*M200/(4*np.pi*200*rhocrit0*gz))**(1./3.)
        rs = r200/c200
        fc200 = self.fc(c200)
        rhos = M200/(4*np.pi*rs**3*fc200)
        Dc = self.Delc(self.Omegaz(pOmega,z)-1.)
        rvir = optimize.fsolve(lambda r: 3.*(rs/r)**3*self.fc(r/rs)*rhos-Dc*rhocrit0*gz,r200)
        Mvir = 4*np.pi*rs**3*rhos*self.fc(rvir/rs)
        return Mvir

    def Mvir_from_M200_fit(self, M200, z):
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        Oz = self.Omegaz(pOmega,z)
        def ffunc(x):
            return np.power(x,3.0)*(np.log(1.0+1.0/x)-1.0/(1.0+x))
        def xfunc(f):
            p = a2 + a3*np.log(f) + a4*np.power(np.log(f),2.0)
            return np.power(a1*np.power(f,2.0*p)+(3.0/4.0)**2,-0.5)+2.0*f
        return self.Delc(Oz-1)/200.0*M200 \
            *np.power(self.conc200(M200,z) \
            *xfunc(self.Delc(Oz-1)/200.0*ffunc(1.0/self.conc200(M200,z))),-3.0)

    def growthD(self, z):
        Omega_Lz = OmegaL*pow(OmegaL+OmegaM*pow(1+z,3),-1)
        Omega_Mz = 1-Omega_Lz
        phiz = pow(Omega_Mz,4.0/7.0)-Omega_Lz+(1+Omega_Mz/2.0)*(1+Omega_Lz/70.0)
        phi0 = pow(OmegaM,4.0/7.0)-OmegaL+(1+OmegaM/2.0)*(1+OmegaL/70.0)
        return (Omega_Mz/OmegaM)*(phi0/phiz)*pow(1+z,-1)

    def xi(self, M):
        return pow(M*pow((1e+10)*pow(h,-1),-1),-1)

    def sigmaMz(self, M, z):
        sigmaM0 = np.exp(self.sigma_int(np.log(M)))
        sigmaMz = sigmaM0*self.growthD(z)
        return sigmaMz

    def dOdz(self, z):
        return -OmegaL*3*Omegam*pow(h,-2)*pow(1+z,2)*pow(OmegaL+Omegam*pow(h,-2)*pow(1+z,3),-2)

    def dDdz(self, z):
        Omega_Lz = OmegaL*pow(OmegaL+Omegam*pow(h,-2)*pow(1+z,3),-1)
        Omega_Mz = 1-Omega_Lz
        phiz = pow(Omega_Mz,4.0/7.0)-Omega_Lz+(1+Omega_Mz/2.0)*(1+Omega_Lz/70.0)
        phi0 = pow(Omegam*pow(h,-2),4.0/7.0)-OmegaL+(1+Omegam*pow(h,-2)/2.0)*(1+OmegaL/70.0)
        dphidz = self.dOdz(z)*((-4.0/7.0)*pow(Omega_Mz,-3.0/7.0)+(Omega_Mz-Omega_Lz)/140.0+(1.0/70.0)-(3.0/2.0))
        return (phi0/OmegaM)*(-self.dOdz(z)*pow(phiz*(1+z),-1)-Omega_Mz*(dphidz*(1+z)+phiz)*pow(phiz,-2)*pow(1+z,-2))

    def Mzi(self, M0, z):
        a = 1.686*np.sqrt(2.0/np.pi)*self.dDdz(0)+1.0
        zf = -0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
        q = 4.137*pow(zf,-0.9476)
        fM0 = pow(pow(self.sigmaMz(M0/q,0),2)-pow(self.sigmaMz(M0,0),2),-0.5)
        return M0*pow(1+z,a*fM0)*np.exp(-fM0*z)

    def Mzzi(self, M0, z, zi):
        Mzi0 = self.Mzi(M0,zi)
        zf = -0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
        q = 4.137*pow(zf,-0.9476)
        fMzi = pow(pow(self.sigmaMz(Mzi0/q,zi),2)-pow(self.sigmaMz(Mzi0,zi),2),-0.5)
        alpha = fMzi*(1.686*np.sqrt(2.0/np.pi)*pow(self.growthD(zi),-2)*self.dDdz(zi)+1)
        beta = -fMzi
        return Mzi0*pow(1+z-zi,alpha)*np.exp(beta*(z-zi))

    def Hz(self, z):
        return H0*np.sqrt(OmegaL+OmegaM*pow(1+z,3))

    def Omegaz(self, p, x):
        E=p[0]*pow(1+x,3)+p[1]*pow(1+x,2)+p[2]
        return p[0]*pow(1+x,3)*pow(E,-1)

    def Delc(self, x):
        return 18*pow(np.pi,2)+(82*x)-39*pow(x,2)

    def dMdz(self, M0, z, zi, sigmafac=0):
        Mzi0 = self.Mzi(M0,zi)
        zf = -0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
        q = 4.137*pow(zf,-0.9476)
        fMzi = pow(pow(self.sigmaMz(Mzi0/q,zi),2)-pow(self.sigmaMz(Mzi0,zi),2),-0.5)
        alpha = fMzi*(1.686*np.sqrt(2.0/np.pi)*pow(self.growthD(zi),-2)*self.dDdz(zi)+1)
        beta = -fMzi
        Mzzidef = Mzi0*pow(1+z-zi,alpha)*np.exp(beta*(z-zi))
        Mzzivir = self.Mvir_from_M200_fit(Mzzidef*Msolar,z)
        return (beta+alpha*pow(1+z-zi,-1))*Mzzivir/Msolar

    def dsdm(self, M, z):
        sigma0 = np.exp(self.sigma_int(np.log(M)))
        dsdM0 = self.dsdM_int(np.log(M))*2.*sigma0**2/M
        dsdMz = dsdM0*self.growthD(z)
        return dsdMz

    def delc_Y11(self, z):
        return 1.686*pow(self.growthD(z),-1)

    def s_Y11(self, M):
        return pow(self.sigmaMz(M,0),2)

    def Ffunc(self, dela, sig1, sig2):
        return pow(2*np.pi,-0.5)*dela*pow(sig2-sig1,-1.5)

    def Gfunc(self, dela, sig1, sig2):
        G0=0.57
        gamma1=0.38
        gamma2=-0.01
        s1=np.sqrt(sig1)
        s2=np.sqrt(sig2)
        return G0*pow(s2/s1,gamma1)*pow(dela/s1,gamma2)

    def Ffunc_Yang(self, delc1, delc2, sig1, sig2):
        return pow(2*np.pi,-0.5)*(delc2-delc1)*pow(sig2-sig1,-1.5) \
            *np.exp(-pow(delc2-delc1,2)*pow(2*(sig2-sig1),-1))

    def Na_calc(self, ma, zacc, Mhost, z0=0, N_herm=200, Nrand=1000, sigmafac=0, Namodel=3):
        """ Returns Na, Eq. (3) of Yang et al. (2011) """
        zacc_2d = zacc.reshape(np.alen(zacc),1)
        M200_0 = self.Mzzi(Mhost,zacc_2d,z0)
        logM200_0 = np.log10(M200_0)
        if N_herm==1:
            sigmalogM200_0 = 0.12+0.15*np.log10(Mhost/M200_0)
            sigmalogM200_1 = sigmalogM200_0[zacc_2d>1.][0] \
                /np.log10(M200_0[zacc_2d>1.][0]/Mhost) \
                *np.log10(M200_0/Mhost)
            sigmalogM200 = np.where(zacc_2d>1.,sigmalogM200_0,sigmalogM200_1)
            logM200=logM200_0+sigmafac*sigmalogM200
            M200=10**logM200
            if(sigmafac>0.):
                M200 = np.where(M200<Mhost,M200,Mhost)
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
            zlist = zacc_2d*np.linspace(1,0,Nrand)
            iMmax = np.argmin(np.abs(self.Mzzi(Mhost,zlist,z0)-Mmax),axis=-1)
            z_Max = zlist[np.arange(np.alen(zlist)),iMmax]
            z_Max_3d = z_Max.reshape(N_herm,np.alen(zlist),1)
            delcM = self.delc_Y11(z_Max_3d)
            delca = self.delc_Y11(zacc_2d)
            sM = self.s_Y11(Mmax)
            sa = self.s_Y11(ma)
            xmax = pow((delca-delcM),2)*pow((2*(self.s_Y11(mmax)-sM)),-1)
            normB = special.gamma(0.5)*special.gammainc(0.5,xmax)/np.sqrt(np.pi)
            #those reside in the exponential part of eq.14
            Phi = self.Ffunc_Yang(delcM,delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
        elif Namodel==1:
            delca = self.delc_Y11(zacc_2d)
            sM = self.s_Y11(M200)
            sa = self.s_Y11(ma)
            xmin = self.s_Y11(mmax)-self.s_Y11(M200)
            normB = pow(2*np.pi,-0.5)*delca*(2.)*pow(xmin,-0.5)*special.hyp2f1(0.5,0.,1.5,-sM/xmin)
            Phi = self.Ffunc(delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
        else:
            delca = self.delc_Y11(zacc_2d)
            sM = self.s_Y11(M200)
            sa = self.s_Y11(ma)
            xmin = self.s_Y11(mmax)-self.s_Y11(M200)
            normB = pow(2*np.pi,-0.5)*delca*0.57 \
                *pow(delca/np.sqrt(sM),-0.01)*(2./(1.-0.38))*pow(sM,-0.38/2.) \
                *pow(xmin,0.5*(0.38-1.)) \
                *special.hyp2f1(0.5*(1-0.38),-0.38/2.,0.5*(3.-0.38),-sM/xmin)
            Phi = self.Ffunc(delca,sM,sa)*self.Gfunc(delca,sM,sa)/normB \
                *np.heaviside(mmax-ma,0)

        if N_herm==1:
            F2t = np.nan_to_num(Phi)
            F2=F2t.reshape((len(zacc_2d),len(ma)))
        else:
            F2 = np.sum(np.nan_to_num(Phi)*wwi/np.sqrt(np.pi),axis=0)
        Na = F2*self.dsdm(ma,0)*self.dMdz(Mhost,zacc_2d,z0,sigmafac)*(1+zacc_2d)
        return Na

    def rs_rhos_calc(self, M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=100, sigmalogc=0.128,
                     N_herm=5, logmamin=-6, logmamax=None, sigmafac=0,
                     N_hermNa=200, Namodel=3, profile_change=True):

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
        Oz_0 = self.Omegaz(pOmega,redshift)

        def Mzvir(z):
            Mz200 = self.Mzzi(M0,z,0)
            if N_hermNa==1:
                logM200_0 = np.log10(Mz200)
                ##added according to the modifications in Na_calc
                sigmalogM200_0 = 0.12-0.15*np.log10(Mz200/M0)
                Mz1 = self.Mzzi(M0,1.,0.)
                sigma1 = 0.12-0.15*np.log10(Mz1/M0)
                sigmalogM200_1 = sigma1/np.log10(Mz1/M0)*np.log10(Mz200/M0)
                sigmalogM200 = np.where(z>1.,sigmalogM200_0,sigmalogM200_1)
                logM200 = logM200_0+sigmafac*sigmalogM200
                M200 = 10**logM200
                if(sigmafac>0.):
                    M200 = np.where(M200<M0,M200,M0)
                Mz200solar = M200*Msolar
                Mvirsolar = self.Mvir_from_M200(Mz200solar,z)
            else:
                Mz200solar = Mz200*Msolar
                Mvirsolar = self.Mvir_from_M200(Mz200solar,z)
            return Mvirsolar/Msolar

        def AMz(z):
            log10a=(-0.0003*np.log10(Mzvir(z))+0.02)*z \
                +(0.011*np.log10(Mzvir(z))-0.354)
            return pow(10,log10a)

        def zetaMz(z):
            return (0.00012*np.log10(Mzvir(z))-0.0033)*z \
                +(-0.0011*np.log10(Mzvir(z))+0.026)

        def tdynz(z):
            Oz_z = self.Omegaz(pOmega,z)
            return 1.628*pow(h,-1)*pow(self.Delc(Oz_z-1)/178.0,-0.5)*pow(self.Hz(z)/H0,-1)*(86400*365*(1e+9))

        def msolve(m, z):
            return AMz(z)*(m/tdynz(z))*pow(m/Mzvir(z),zetaMz(z))*pow(self.Hz(z)*(1+z),-1)

        for iz in range(len(zdist)):
            ma = self.Mvir_from_M200(ma200*Msolar,zdist[iz])/Msolar
            Oz = self.Omegaz(pOmega,zdist[iz])
            zcalc = np.linspace(zdist[iz],redshift,100)
            sol = odeint(msolve,ma,zcalc)
            m0 = sol[-1]
            c200sub = self.conc200(ma200*Msolar,zdist[iz])
            rvirsub = pow(3*ma*Msolar*pow(rhocrit0*self.g(zdist[iz]) \
                *self.Delc(Oz-1)*4*np.pi,-1),1.0/3.0)
            r200sub = pow(3*ma200*Msolar*pow(rhocrit0*self.g(zdist[iz]) \
                *200*4*np.pi,-1),1.0/3.0)
            c_mz = c200sub*rvirsub/r200sub
            x1,w1 = hermgauss(N_herm)
            x1 = x1.reshape(np.alen(x1),1)
            w1 = w1.reshape(np.alen(w1),1)
            log10c_sub = np.sqrt(2)*sigmalogc*x1+np.log10(c_mz)
            c_sub = pow(10.0,log10c_sub)
            rs_acc[iz] = rvirsub/c_sub
            rhos_acc[iz] = ma*Msolar/(4*np.pi*rs_acc[iz]**3*self.fc(c_sub))
            if(profile_change==True):
                rmax_acc = rs_acc[iz]*2.163
                Vmax_acc = np.sqrt(rhos_acc[iz]*4*np.pi*G/4.625)*rs_acc[iz]
                Vmax_z0 = Vmax_acc*(pow(2,0.4)*pow(m0/ma,0.3)*pow(1+m0/ma,-0.4))
                rmax_z0 = rmax_acc*(pow(2,-0.3)*pow(m0/ma,0.4)*pow(1+m0/ma,0.3))
                rs_z0[iz] = rmax_z0/2.163
                rhos_z0[iz] = (4.625/(4*np.pi*G))*pow(Vmax_z0/rs_z0[iz],2)
            else:
                rs_z0[iz] = rs_acc[iz]
                rhos_z0[iz] = rhos_acc[iz]
            ctemp = np.linspace(0,100,1000)
            ftemp = interp1d(self.fc(ctemp),ctemp,fill_value='extrapolate')
            ct_z0[iz] = ftemp(m0*Msolar/(4*np.pi*rhos_z0[iz]*rs_z0[iz]**3))
            survive[iz] = np.where(ct_z0[iz]>0.77,1,0)
            m0_matrix[iz] = m0*np.ones((N_herm,1))

        Na = self.Na_calc(ma,zdist,M0,z0=0,N_herm=N_hermNa,Nrand=1000,
                          sigmafac=sigmafac,Namodel=Namodel)
        Na_total = integrate.simps(integrate.simps(Na,x=np.log(ma)),x=np.log(1+zdist))
        weight = Na/(1.0+zdist.reshape(np.alen(zdist),1))
        weight = weight/np.sum(weight)*Na_total
        weight = (weight.reshape((len(zdist),1,len(ma))))*w1/np.sqrt(np.pi)
        z_acc = (zdist.reshape(len(zdist),1,1))*np.ones((1,N_herm,N_ma))
        z_acc = z_acc.reshape(len(zdist)*N_herm*N_ma)
        ma200_matrix = ma200*np.ones((len(zdist),N_herm,1))
        ma200_matrix = ma200_matrix.reshape(len(zdist)*N_herm*len(ma200))
        m0_matrix = m0_matrix.reshape(len(zdist)*N_herm*len(ma200))
        rs_acc = rs_acc.reshape(len(zdist)*N_herm*len(ma200))
        rhos_acc = rhos_acc.reshape(len(zdist)*N_herm*len(ma200))
        rs_z0 = rs_z0.reshape(len(zdist)*N_herm*len(ma200))
        rhos_z0 = rhos_z0.reshape(len(zdist)*N_herm*len(ma200))
        ct_z0 = ct_z0.reshape(len(zdist)*N_herm*len(ma200))
        weight = weight.reshape(len(zdist)*N_herm*len(ma200))
        survive = (survive==1).reshape(len(zdist)*N_herm*len(ma200))

        return ma200_matrix, z_acc, rs_acc/kpc, rhos_acc/(Msolar/pc**3), \
            m0_matrix, rs_z0/kpc, rhos_z0/(Msolar/pc**3), ct_z0, \
            weight, survive


####################################################
####################################################
####################################################
####################################################
####################################################


"""

def shmf_calc(M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128, N_herm=5,
              logmamin=-6, logmamax=None, massbins=100, sigmafac=1, N_hermNa=200, Namodel=3):
    ma200,z_a,rs_a,rhos_a,m0,rs,rhos,ct,w,surv \
        = rs_rhos_calc(M0=M0,redshift=redshift,dz=dz,zmax=zmax,N_ma=N_ma,sigmalogc=sigmalogc,
                       N_herm=N_herm,logmamin=logmamin,logmamax=logmamax,sigmafac=sigmafac,
                       N_hermNa=N_hermNa,Namodel=Namodel)
    Delta_N,lnm_edges = np.histogram(np.log(m0),weights=w,
                                     range=(np.log(np.min(m0)),np.log(np.max(m0))),
                                     bins=massbins)
    lnm = (lnm_edges[1:]+lnm_edges[:-1])/2.0
    Delta_lnm = lnm_edges[1:]-lnm_edges[:-1]
    dNdlnm = Delta_N/Delta_lnm
    m = np.exp(lnm)
    dNdm = dNdlnm/m
    return m,dNdm




def boost_calc(M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128, N_herm=5):
    Lsh,m0,weights,weight_MF = subhalo_properties(M0=M0,redshift=redshift,dz=dz,zmax=zmax,
                                        N_ma=N_ma,sigmalogc=sigmalogc,N_herm=N_herm)
    boost = np.sum(Lsh*weights)
    return boost



def boost_calc_iter2(M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128, N_herm=5):

    zdist = np.arange(redshift+dz,zmax+dz,dz)
    ma200 = np.logspace(-6,np.log10(0.1*M0),N_ma)
    Lsh = np.zeros((len(zdist),len(ma200)))
    Oz_0=Omegaz(pOmega,redshift)

    z_data = np.arange(0.0,6.9,0.1)
    B_2d = np.zeros((np.alen(ma200),np.alen(z_data)))
    for i in np.arange(np.alen(z_data)):
        boost_data = np.loadtxt('data/Correa0313HG_BshMhost1.000000e-04-1.000000e+16_z'
                                +'%.3s00000.dat' % z_data[i])
        m200_interp = boost_data[:,2]
        boost_interp = boost_data[:,3]
        f_logB = interp1d(np.log(m200_interp),np.log(boost_interp),
                          bounds_error=False,fill_value=-100.0)
        B_2d[:,i] = np.exp(f_logB(np.log(ma200)))
    f_z = interp1d(z_data,B_2d,axis=-1,bounds_error=False,fill_value=0.0)
    Bssh0 = f_z(zdist).T

    Mvir = Mvir_from_M200(Mzi(M0,redshift)*Msolar,redshift)
    R200 = pow(3.0*Mzi(M0,redshift)*Msolar*pow(rhocrit0*g(redshift)*200*4*np.pi,-1),1.0/3.0)
    Rvir = pow(3*Mvir*pow(rhocrit0*g(redshift)*Delc(Oz_0-1)*4*np.pi,-1),1.0/3.0)
    c200_Mz = conc200(Mzi(M0,redshift)*Msolar,redshift)
    c_Mz = c200_Mz*Rvir/R200
    x2,w2 = hermgauss(N_herm)
    log10c_host = np.sqrt(2)*sigmalogc*x2+np.log10(c_Mz)
    c_host = 10.**log10c_host
    Rs = Rvir/c_host
    RHOs = Mvir/(4*np.pi*Rs**3*fc(c_host))
    Lhost = np.sum(RHOs**2*Rs**3*(1-(1+c_host)**-3)*w2/np.sqrt(np.pi))

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

    def msolve(m,z):
        return AMz(z)*(m/tdynz(z))*pow(m/Mzvir(z),zetaMz(z))*pow(Hz(z)*(1+z),-1)

    for iz in range(len(zdist)):
        ma = Mvir_from_M200(ma200*Msolar,zdist[iz])/Msolar
        Oz = Omegaz(pOmega,zdist[iz])
        zcalc = np.linspace(zdist[iz],redshift,100)
        sol = odeint(msolve,ma,zcalc)
        m0 = sol[-1]
        c200sub = conc200(ma200*Msolar,zdist[iz])
        rvirsub = pow(3*ma*Msolar*pow(rhocrit0*g(zdist[iz])*Delc(Oz-1)*4*np.pi,-1),1.0/3.0)
        r200sub = pow(3*ma200*Msolar*pow(rhocrit0*g(zdist[iz])*200*4*np.pi,-1),1.0/3.0)
        c_mz = c200sub*rvirsub/r200sub
        x1,w1 = hermgauss(N_herm)
        x1 = x1.reshape(np.alen(x1),1)
        w1 = w1.reshape(np.alen(w1),1)
        log10c_sub = np.sqrt(2)*sigmalogc*x1+np.log10(c_mz)
        c_sub = pow(10.0,log10c_sub)
        rs_acc = rvirsub/c_sub
        rhos_acc = ma*Msolar/(4*np.pi*rs_acc**3*fc(c_sub))
        rmax_acc = rs_acc*2.163
        Vmax_acc = np.sqrt(rhos_acc*4*np.pi*G/4.625)*rs_acc
        Vmax_z0 = Vmax_acc*(pow(2,0.4)*pow(m0/ma,0.3)*pow(1+m0/ma,-0.4))
        rmax_z0 = rmax_acc*(pow(2,-0.3)*pow(m0/ma,0.4)*pow(1+m0/ma,0.3))
        rs_z0 = rmax_z0/2.163
        rhos_z0 = (4.625/(4*np.pi*G))*pow(Vmax_z0/rs_z0,2)
        ctemp = np.linspace(0,100,1000)
        ftemp = interp1d(fc(ctemp),ctemp,fill_value='extrapolate')
        ct_z0 = ftemp(m0*Msolar/(4*np.pi*rhos_z0*rs_z0**3))
        Bssh = Bssh0[iz]
        Bssh = Bssh*(np.log(np.sqrt(1+ct_z0**2)+ct_z0)-ct_z0/np.sqrt(1+ct_z0**2))
        Bssh = Bssh/(np.log(np.sqrt(1+c_sub**2)+c_sub)-c_sub/np.sqrt(1+c_sub**2))
        Bssh = Bssh/((1-(1+ct_z0)**-3)/(1-(1+c_sub)**-3))
        Lsh[iz] = np.sum(rhos_z0**2*rs_z0**3*(1-(1+ct_z0)**-3)*(1+Bssh)*np.where(ct_z0>0.77,1,0)*w1/np.sqrt(np.pi),axis=0)

#    Na = Na_calc(ma,zdist,M0,z0=0)
    Na = Na_calc(ma,zdist,M0,z0=0,N_herm=N_hermNa,Nrand=1000,sigmafac=sigmafac,Namodel=2)
    Na_total = integrate.simps(integrate.simps(Na,x=np.log(ma)),x=np.log(1+zdist))
    weight = Na/(1.0+zdist.reshape(np.alen(zdist),1))
    weight = weight/np.sum(weight)*Na_total

    Lsh = Lsh.reshape(len(zdist)*len(ma200))
    weight = weight.reshape(len(zdist)*len(ma200))
    Lsh /= Lhost
    boost = np.sum(Lsh*weight)

    return boost




def boost_calc_iter3(M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128, N_herm=5):

    zdist = np.arange(redshift+dz,zmax+dz,dz)
    ma200 = np.logspace(-6,np.log10(0.1*M0),N_ma)
    Lsh = np.zeros((len(zdist),len(ma200)))
    Oz_0=Omegaz(pOmega,redshift)

    z_data = np.arange(0.0,6.9,0.1)
    B_2d = np.zeros((np.alen(ma200),np.alen(z_data)))
    for i in np.arange(np.alen(z_data)):
        boost_data = np.loadtxt('data/sub1Correa0313HG_BshMhost1.000000e-04-1.000000e+16_z'
                                +'%.3s00000.dat' % z_data[i])
        m200_interp = boost_data[:,2]
        boost_interp = boost_data[:,3]
        f_logB = interp1d(np.log(m200_interp),np.log(boost_interp),
                          bounds_error=False,fill_value=-100.0)
        B_2d[:,i] = np.exp(f_logB(np.log(ma200)))
    f_z = interp1d(z_data,B_2d,axis=-1,bounds_error=False,fill_value=0.0)
    Bssh0 = f_z(zdist).T

    Mvir = Mvir_from_M200(Mzi(M0,redshift)*Msolar,redshift)
    R200 = pow(3.0*Mzi(M0,redshift)*Msolar*pow(rhocrit0*g(redshift)*200*4*np.pi,-1),1.0/3.0)
    Rvir = pow(3*Mvir*pow(rhocrit0*g(redshift)*Delc(Oz_0-1)*4*np.pi,-1),1.0/3.0)
    c200_Mz = conc200(Mzi(M0,redshift)*Msolar,redshift)
    c_Mz = c200_Mz*Rvir/R200
    x2,w2 = hermgauss(N_herm)
    log10c_host = np.sqrt(2)*sigmalogc*x2+np.log10(c_Mz)
    c_host = 10.**log10c_host
    Rs = Rvir/c_host
    RHOs = Mvir/(4*np.pi*Rs**3*fc(c_host))
    Lhost = np.sum(RHOs**2*Rs**3*(1-(1+c_host)**-3)*w2/np.sqrt(np.pi))

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

    def msolve(m,z):
        return AMz(z)*(m/tdynz(z))*pow(m/Mzvir(z),zetaMz(z))*pow(Hz(z)*(1+z),-1)

    iz=0
    for iz in range(len(zdist)):
        ma = Mvir_from_M200(ma200*Msolar,zdist[iz])/Msolar
        Oz = Omegaz(pOmega,zdist[iz])
        zcalc = np.linspace(zdist[iz],redshift,100)
        sol = odeint(msolve,ma,zcalc)
        m0 = sol[-1]
        c200sub = conc200(ma200*Msolar,zdist[iz])
        rvirsub = pow(3*ma*Msolar*pow(rhocrit0*g(zdist[iz])*Delc(Oz-1)*4*np.pi,-1),1.0/3.0)
        r200sub = pow(3*ma200*Msolar*pow(rhocrit0*g(zdist[iz])*200*4*np.pi,-1),1.0/3.0)
        c_mz = c200sub*rvirsub/r200sub
        x1,w1 = hermgauss(N_herm)
        x1 = x1.reshape(np.alen(x1),1)
        w1 = w1.reshape(np.alen(w1),1)
        log10c_sub = np.sqrt(2)*sigmalogc*x1+np.log10(c_mz)
        c_sub = pow(10.0,log10c_sub)
        rs_acc = rvirsub/c_sub

        rhos_acc = ma*Msolar/(4*np.pi*rs_acc**3*fc(c_sub))
        rmax_acc = rs_acc*2.163
        Vmax_acc = np.sqrt(rhos_acc*4*np.pi*G/4.625)*rs_acc
        Vmax_z0 = Vmax_acc*(pow(2,0.4)*pow(m0/ma,0.3)*pow(1+m0/ma,-0.4))
        rmax_z0 = rmax_acc*(pow(2,-0.3)*pow(m0/ma,0.4)*pow(1+m0/ma,0.3))
        rs_z0 = rmax_z0/2.163
        rhos_z0 = (4.625/(4*np.pi*G))*pow(Vmax_z0/rs_z0,2)
        ctemp = np.linspace(0,100,1000)
        ftemp = interp1d(fc(ctemp),ctemp,fill_value='extrapolate')
        ct_z0 = ftemp(m0*Msolar/(4*np.pi*rhos_z0*rs_z0**3))
        Bssh = Bssh0[iz]
        Bssh = Bssh*(np.log(np.sqrt(1+ct_z0**2)+ct_z0)-ct_z0/np.sqrt(1+ct_z0**2))
        Bssh = Bssh/(np.log(np.sqrt(1+c_sub**2)+c_sub)-c_sub/np.sqrt(1+c_sub**2))
        Bssh = Bssh/((1-(1+ct_z0)**-3)/(1-(1+c_sub)**-3))
        Lsh[iz] = np.sum(rhos_z0**2*rs_z0**3*(1-(1+ct_z0)**-3)*(1+Bssh)*np.where(ct_z0>0.77,1,0)*w1/np.sqrt(np.pi),axis=0)

#    Na = Na_calc(ma,zdist,M0,z0=0)
    Na = Na_calc(ma,zdist,M0,z0=0,N_herm=N_hermNa,Nrand=1000,sigmafac=sigmafac,Namodel=2)
    Na_total = integrate.simps(integrate.simps(Na,x=np.log(ma)),x=np.log(1+zdist))
    weight = Na/(1.0+zdist.reshape(np.alen(zdist),1))
    weight = weight/np.sum(weight)*Na_total

    Lsh = Lsh.reshape(len(zdist)*len(ma200))
    weight = weight.reshape(len(zdist)*len(ma200))
    Lsh /= Lhost
    boost = np.sum(Lsh*weight)

    return boost





def boost_calc_iter4(M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128, N_herm=5):

    zdist = np.arange(redshift+dz,zmax+dz,dz)
    ma200 = np.logspace(-6,np.log10(0.1*M0),N_ma)
    Lsh = np.zeros((len(zdist),len(ma200)))
    Oz_0=Omegaz(pOmega,redshift)

    z_data = np.arange(0.0,6.9,0.1)
    B_2d = np.zeros((np.alen(ma200),np.alen(z_data)))
    for i in np.arange(np.alen(z_data)):
        boost_data = np.loadtxt('data/sub2Correa0313HG_BshMhost1.000000e-04-1.000000e+16_z'
                                +'%.3s00000.dat' % z_data[i])
        m200_interp = boost_data[:,2]
        boost_interp = boost_data[:,3]
        f_logB = interp1d(np.log(m200_interp),np.log(boost_interp),
                          bounds_error=False,fill_value=-100.0)
        B_2d[:,i] = np.exp(f_logB(np.log(ma200)))
    f_z = interp1d(z_data,B_2d,axis=-1,bounds_error=False,fill_value=0.0)
    Bssh0 = f_z(zdist).T

    Mvir = Mvir_from_M200(Mzi(M0,redshift)*Msolar,redshift)
    R200 = pow(3.0*Mzi(M0,redshift)*Msolar*pow(rhocrit0*g(redshift)*200*4*np.pi,-1),1.0/3.0)
    Rvir = pow(3*Mvir*pow(rhocrit0*g(redshift)*Delc(Oz_0-1)*4*np.pi,-1),1.0/3.0)
    c200_Mz = conc200(Mzi(M0,redshift)*Msolar,redshift)
    c_Mz = c200_Mz*Rvir/R200
    x2,w2 = hermgauss(N_herm)
    log10c_host = np.sqrt(2)*sigmalogc*x2+np.log10(c_Mz)
    c_host = 10.**log10c_host
    Rs = Rvir/c_host
    RHOs = Mvir/(4*np.pi*Rs**3*fc(c_host))
    Lhost = np.sum(RHOs**2*Rs**3*(1-(1+c_host)**-3)*w2/np.sqrt(np.pi))

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

    def msolve(m,z):
        return AMz(z)*(m/tdynz(z))*pow(m/Mzvir(z),zetaMz(z))*pow(Hz(z)*(1+z),-1)

    iz=0
    for iz in range(len(zdist)):
        ma = Mvir_from_M200(ma200*Msolar,zdist[iz])/Msolar
        Oz = Omegaz(pOmega,zdist[iz])
        zcalc = np.linspace(zdist[iz],redshift,100)
        sol = odeint(msolve,ma,zcalc)
        m0 = sol[-1]
        c200sub = conc200(ma200*Msolar,zdist[iz])
        rvirsub = pow(3*ma*Msolar*pow(rhocrit0*g(zdist[iz])*Delc(Oz-1)*4*np.pi,-1),1.0/3.0)
        r200sub = pow(3*ma200*Msolar*pow(rhocrit0*g(zdist[iz])*200*4*np.pi,-1),1.0/3.0)
        c_mz = c200sub*rvirsub/r200sub
        x1,w1 = hermgauss(N_herm)
        x1 = x1.reshape(np.alen(x1),1)
        w1 = w1.reshape(np.alen(w1),1)
        log10c_sub = np.sqrt(2)*sigmalogc*x1+np.log10(c_mz)
        c_sub = pow(10.0,log10c_sub)
        rs_acc = rvirsub/c_sub

        rhos_acc = ma*Msolar/(4*np.pi*rs_acc**3*fc(c_sub))
        rmax_acc = rs_acc*2.163
        Vmax_acc = np.sqrt(rhos_acc*4*np.pi*G/4.625)*rs_acc
        Vmax_z0 = Vmax_acc*(pow(2,0.4)*pow(m0/ma,0.3)*pow(1+m0/ma,-0.4))
        rmax_z0 = rmax_acc*(pow(2,-0.3)*pow(m0/ma,0.4)*pow(1+m0/ma,0.3))
        rs_z0 = rmax_z0/2.163
        rhos_z0 = (4.625/(4*np.pi*G))*pow(Vmax_z0/rs_z0,2)
        ctemp = np.linspace(0,100,1000)
        ftemp = interp1d(fc(ctemp),ctemp,fill_value='extrapolate')
        ct_z0 = ftemp(m0*Msolar/(4*np.pi*rhos_z0*rs_z0**3))
        Bssh = Bssh0[iz]
        Bssh = Bssh*(np.log(np.sqrt(1+ct_z0**2)+ct_z0)-ct_z0/np.sqrt(1+ct_z0**2))
        Bssh = Bssh/(np.log(np.sqrt(1+c_sub**2)+c_sub)-c_sub/np.sqrt(1+c_sub**2))
        Bssh = Bssh/((1-(1+ct_z0)**-3)/(1-(1+c_sub)**-3))
        Lsh[iz] = np.sum(rhos_z0**2*rs_z0**3*(1-(1+ct_z0)**-3)*(1+Bssh)*np.where(ct_z0>0.77,1,0)*w1/np.sqrt(np.pi),axis=0)

#    Na = Na_calc(ma,zdist,M0,z0=0)
    Na = Na_calc(ma,zdist,M0,z0=0,N_herm=N_hermNa,Nrand=1000,sigmafac=sigmafac,Namodel=2)
    Na_total = integrate.simps(integrate.simps(Na,x=np.log(ma)),x=np.log(1+zdist))
    weight = Na/(1.0+zdist.reshape(np.alen(zdist),1))
    weight = weight/np.sum(weight)*Na_total

    Lsh = Lsh.reshape(len(zdist)*len(ma200))
    weight = weight.reshape(len(zdist)*len(ma200))
    Lsh /= Lhost
    boost = np.sum(Lsh*weight)

    return boost


def Na_calc_Yangmodel3(ma, zacc, Mhost, z0=0, N_herm=200, Nrand=1000,sigmafac=1):
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
#    Mmax=np.minimum(M200+mmax,Mhost)
    Mmax=np.minimum(M200_0+mmax,Mhost)
#    print(mmax,Mmax)

    zlist=zacc_2d*np.linspace(1,0,Nrand)
    iMmax=np.argmin(np.abs(Mzzi(Mhost,zlist,z0)-Mmax),axis=-1)

    z_Max=zlist[np.arange(np.alen(zlist)),iMmax]
    z_Max_3d=z_Max.reshape(N_herm,np.alen(zlist),1)
    delcM=delc_Y11(z_Max_3d)
    delca=delc_Y11(zacc_2d)

    sM = s_Y11(Mmax)
    sa = s_Y11(mmax)*(np.linspace(1,10,500).reshape(500,1,1,1))
    normB = integrate.simps(Ffunc(delcM,delca,sM,sa),x=sa,axis=0)
    #those reside in the exponential part of eq.14
#    xmax = pow((delca-delcM),2)*pow((2*(s_Y11(mmax)-sM)),-1)
#    normB = special.gamma(0.5)*special.gammainc(0.5,xmax)/np.sqrt(np.pi)

    sa = s_Y11(ma)
    Phi = Ffunc(delcM,delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
    if N_herm==1:
        F2t = np.nan_to_num(Phi)
        F2=F2t.reshape((len(zacc),len(ma)))

    else:
        F2 = np.sum(np.nan_to_num(Phi)*wwi/np.sqrt(np.pi),axis=0)
    Na = F2*dsdm(ma,0)*dMdz(Mhost,zacc_2d,z0,sigmafac)*(1+zacc_2d)
    return Na


sigmafac = -1
ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w,surv \
    = rs_rhos_calc(M0=1.3e12,N_ma=1000,sigmalogc=0.128,
                   N_herm=10,logmamin=4,
                   logmamax=None,sigmafac=sigmafac,N_hermNa=1,Namodel=3)
np.save('data/dwarf_params/subhalo_params_1.3e12_-1sigma.npy',
        np.array([ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w]).T)

sigmafac = -2
ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w,surv \
    = rs_rhos_calc(M0=1.3e12,N_ma=1000,sigmalogc=0.128,
                   N_herm=10,logmamin=4,
                   logmamax=None,sigmafac=sigmafac,N_hermNa=1,Namodel=3)
np.save('data/dwarf_params/subhalo_params_1.3e12_-2sigma.npy',
        np.array([ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w]).T)

sigmafac = 1
ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w,surv \
    = rs_rhos_calc(M0=1.3e12,N_ma=1000,sigmalogc=0.128,
                   N_herm=10,logmamin=4,
                   logmamax=None,sigmafac=sigmafac,N_hermNa=1,Namodel=3)
np.save('data/dwarf_params/subhalo_params_1.3e12_+1sigma.npy',
        np.array([ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w]).T)

sigmafac = +2
ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w,surv \
    = rs_rhos_calc(M0=1.3e12,N_ma=1000,sigmalogc=0.128,
                   N_herm=10,logmamin=4,
                   logmamax=None,sigmafac=sigmafac,N_hermNa=1,Namodel=3)
np.save('data/dwarf_params/subhalo_params_1.3e12_+2sigma.npy',
        np.array([ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w]).T)

ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w,surv \
    = rs_rhos_calc(M0=1.3e12,N_ma=1000,sigmalogc=0.128,
                   N_herm=10,logmamin=4,
                   logmamax=None,sigmafac=0,N_hermNa=200,Namodel=3)
np.save('data/dwarf_params/subhalo_params_1.3e12.npy',
        np.array([ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w]).T)

ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w,surv \
    = rs_rhos_calc(M0=1.6e12,N_ma=1000,sigmalogc=0.128,
                   N_herm=10,logmamin=4,
                   logmamax=None,sigmafac=0,N_hermNa=200,Namodel=3)
np.save('data/dwarf_params/subhalo_params_1.6e12.npy',
        np.array([ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w]).T)

ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w,surv \
    = rs_rhos_calc(M0=1e12,N_ma=1000,sigmalogc=0.128,
                   N_herm=10,logmamin=4,
                   logmamax=None,sigmafac=0,N_hermNa=200,Namodel=3)
np.save('data/dwarf_params/subhalo_params_1e12.npy',
        np.array([ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w]).T)

ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w,surv \
    = rs_rhos_calc(M0=1.9e12,N_ma=1000,sigmalogc=0.128,
                   N_herm=10,logmamin=4,
                   logmamax=None,sigmafac=0,N_hermNa=200,Namodel=3)
np.save('data/dwarf_params/subhalo_params_1.9e12.npy',
        np.array([ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w]).T)

ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w,surv \
    = rs_rhos_calc(M0=.7e12,N_ma=1000,sigmalogc=0.128,
                   N_herm=10,logmamin=4,
                   logmamax=None,sigmafac=0,N_hermNa=200,Namodel=3)
np.save('data/dwarf_params/subhalo_params_0.7e12.npy',
        np.array([ma200,za,rs_a,rhos_a,m0,rs0,rhos0,ct0,w]).T)
"""
