import scipy.linalg
import numpy as np
import time
from numba import njit,prange

def PerformJADOC(mC,mB0=None,iT=100,iTmin=10,dTol=1E-4,dTauH=1E-2,dLambda0=1,iS=None):
    """Joint Approximate Diagonalization under Orthogonality Constraints
    (JADOC)
    
    Authors: Ronald de Vlaming and Eric Slob
    Repository: https://www.github.com/devlaming/jadoc/
    
    Input
    ------
    mC : np.ndarray with shape (iK, iN, iN)
        iK Hermitian iN-by-iN matrices to jointly diagonalize
    
    mB0 : np.ndarray with shape (iN, iN), optional
        starting value for unitary transformation matrix such
        that mB@mC[i]@(mB.conj().T) is approximately diagonal for all i
    
    iT : int, optional
        maximum number of iterations; default=100
    
    iTmin : int, optional
        minimal number of iterations before convergence is tested; default=10
    
    dTol : float, optional
        stop if average magnitude elements gradient<dTol; default=1E-4
    
    dTauH : float, optional
        minimum value of second-order derivatives; default=1E-2
    
    dLambda0 : float, optional
        regularization coefficient; default=1
    
    iS : int, optional
        replace mC[i] by rank-iS approximation; default=None
        (set to ceil(iN/iK) under the default value)
    
    Output
    ------
    mB : np.ndarray with shape (iN, iN)
        unitary matrix such that mB@mC[i]@(mB.conj().T) is
        approximately diagonal for all i
    """
    print("Starting JADOC")
    (iK,iN,_)=mC.shape
    if iS is None:
        iS=(iN/iK)
        if (iS-int(iS))>0: iS=int(iS)+1
        else: iS=int(iS)
    if iS==iN: print("Computing decomposition of input matrices")
    elif iS>iN:
        raise ValueError("Desired rank (iS) exceeds dimensionality" \
                         +" of input matrices (iN)")
    else: print("Computing low-dimensional decomposition of input matrices")
    if mB0 is None: mB=np.eye(iN)
    elif mB0.shape!=(iN,iN):
        raise ValueError("Starting value transformation matrix" \
                         +" has wrong shape")
    else: mB=mB0
    mA=np.empty((iK,iN,iS),dtype="complex128")
    dLambda=dLambda0
    print("Initial regularization coefficient = "+str(dLambda))
    for i in range(iK):
        (vD,mP)=np.linalg.eigh(mC[i])
        vD=abs(vD)
        dSD1=vD.sum()
        vLeadEVs=((vD.argsort().argsort())>=iN-iS)
        vD=vD[vLeadEVs]
        dSD2=vD.sum()
        mP=mP[:,vLeadEVs]
        dLambda+=((dSD1-dSD2)/(iN*iK))
        mA[i]=mP*(np.sqrt(vD)[None,:])
        if mB0 is not None: mA[i]=np.dot(mB,mA[i])
    print("Final regularization coefficient = "+str(dLambda))
    (mP,vD,mC)=(None,None,None)
    print("Starting quasi-Newton algorithm with line search (golden section)")
    bConverged=False
    for t in range(iT):
        (dLoss,mDiags,dRMSG,mU)=ComputeLoss(mA,dLambda,dTauH)
        if dRMSG<dTol and t>=iTmin:
            bConverged=True
            break
        dStepSize=PerformGoldenSection(mA,mU,mB,dLambda)
        print("ITER "+str(t)+": L="+str(round(dLoss,3))+", RMSD(g)=" \
              +str(round(dRMSG,6))+", step="+str(round(dStepSize,3)))
        (mB,mA)=UpdateEstimates(mA,mU,mB,dStepSize)
    if not(bConverged):
        print("WARNING: JADOC did not converge. Reconsider data or thresholds")
    print("Returning transformation matrix B")
    return mB

def ComputeLoss(mA,dLambda,dTauH=None,bLossOnly=False):
    mDiags=((np.real(mA)**2).sum(axis=2))+((np.imag(mA)**2).sum(axis=2))\
        +dLambda
    (iK,iN,iS)=mA.shape
    dLoss=0.5*(np.log(mDiags).sum())/iK
    if bLossOnly:
        return dLoss
    else:
        mF=np.zeros((iN,iN),dtype="complex128")
        mF=ComputeF(mF,mA,mDiags,iK,iN)
        mG=(mF-(mF.conj().T))
        dRMSG=np.sqrt((((np.real(mG)**2).sum())+((np.imag(mG)**2).sum()))\
                      /(iN*(iN-1)))
        mH=(mDiags[:,:,None]/mDiags[:,None,:]).mean(axis=0)
        mH=mH+mH.T-2.0
        mH[mH<dTauH]=dTauH
        mU=-mG/mH
        return dLoss,mDiags,dRMSG,mU

@njit
def ComputeF(mF,mA,mDiags,iK,iN):
    for i in prange(iK):
        vDiags=(mDiags[i]).reshape((iN,1))
        mF+=np.dot(mA[i]/vDiags,mA[i].conj().T)
    mF=mF/iK
    return mF

def PerformGoldenSection(mA,mU,mB,dLambda):
    dTheta=2/(1+(5**0.5))
    iIter=0
    iMaxIter=15
    iGuesses=4
    (dStepLB,dStepUB)=(0,1)
    bLossOnlyGold=True
    (iK,iN,iS)=mA.shape
    mR=scipy.linalg.expm(mU)
    mAS=np.empty((iGuesses,iK,iN,iS),dtype="complex128")
    mAS[0]=mA.copy()
    mAS[1]=RotateData(mR,mA.copy())
    mAS[2]=(1-dTheta)*mAS[1]+dTheta*mAS[0]
    mAS[3]=(1-dTheta)*mAS[0]+dTheta*mAS[1]
    (mA,mR)=(None,None)
    dLoss2=ComputeLoss(mAS[2],dLambda,bLossOnly=bLossOnlyGold)
    dLoss3=ComputeLoss(mAS[3],dLambda,bLossOnly=bLossOnlyGold)
    while iIter<iMaxIter:
        if (dLoss2<dLoss3):
            mAS[1]=mAS[3]
            mAS[3]=mAS[2]
            dLoss3=dLoss2
            dStepUB=dStepLB+dTheta*(dStepUB-dStepLB)
            mAS[2]=mAS[1]-dTheta*(mAS[1]-mAS[0])
            dLoss2=ComputeLoss(mAS[2],dLambda,bLossOnly=bLossOnlyGold)
        else:
            mAS[0]=mAS[2]
            mAS[2]=mAS[3]
            dLoss2=dLoss3
            dStepLB=dStepUB-dTheta*(dStepUB-dStepLB)
            mAS[3]=mAS[0]+dTheta*(mAS[1]-mAS[0])
            dLoss3=ComputeLoss(mAS[3],dLambda,bLossOnly=bLossOnlyGold)
        iIter+=1
    return np.log(1+(dStepLB*(np.exp(1)-1)))

def UpdateEstimates(mA,mU,mB,dStepSize):
    mR=scipy.linalg.expm(dStepSize*mU)
    mB=np.dot(mR,mB)
    mA=RotateData(mR,mA)
    return mB,mA

@njit
def RotateData(mR,mData):
    iK=mData.shape[0]
    for i in prange(iK):
        mData[i]=np.dot(mR,mData[i])
    return mData

def SimulateData(iK,iN,iR,dAlpha,bComplex=False,bPSD=True):
    if bComplex: sType1="Hermitian "
    else: sType1="real symmetric "
    if bPSD: sType2="positive (semi)-definite "
    else: sType2=""
    print("Simulating "+str(iK)+" distinct "+str(iN)+"-by-"+str(iN)+" " \
          +sType1+sType2+"matrices with alpha="+str(dAlpha) \
              +", for run "+str(iR))
    iMainSeed=15348091
    iRmax=10000
    if iR>=iRmax:
        return
    rngMain=np.random.default_rng(iMainSeed)
    vSeed=rngMain.integers(0,iMainSeed,iRmax)
    iSeed=vSeed[iR]
    rng=np.random.default_rng(iSeed)
    if bComplex:
        mX=rng.normal(size=(iN,iN))+1j*rng.normal(size=(iN,iN))
        mC=np.empty((iK,iN,iN),dtype="complex128")
    else:
        mX=rng.normal(size=(iN,iN))
        mC=np.empty((iK,iN,iN))
    for i in range(0,iK):
        if bComplex:
            mXk=rng.normal(size=(iN,iN))+1j*rng.normal(size=(iN,iN))
        else:
            mXk=rng.normal(size=(iN,iN))
        mXk=dAlpha*mX+(1-dAlpha)*mXk
        mR=scipy.linalg.expm(mXk-(mXk.conj().T))
        vD=rng.normal(size=iN)
        if bPSD:
            vD=vD**2
        mC[i]=np.dot(mR*(vD[None,:]),mR.conj().T)
    return mC

def Test():
    iK=10
    iN=100
    iR=1
    dAlpha=0.9
    mC=SimulateData(iK,iN,iR,dAlpha)
    dTimeStart=time.time()
    mB=PerformJADOC(mC)
    dTime=time.time()-dTimeStart
    print("Runtime: "+str(round(dTime,3))+" seconds")
    mD=np.empty((iK,iN,iN),dtype="complex128")
    for i in range(iK):
        mD[i]=np.dot(np.dot(mB,mC[i]),mB.conj().T)
    dSS_C=0
    dSS_D=0
    for i in range(iK):
        mOffPre=mC[i]-np.diag(np.diag(mC[i]))
        mOffPost=mD[i]-np.diag(np.diag(mD[i]))
        dSS_C+=(np.real(mOffPre)**2).sum()+(np.imag(mOffPre)**2).sum()
        dSS_D+=(np.real(mOffPost)**2).sum()+(np.imag(mOffPost)**2).sum()
    dRMS_C=np.sqrt(dSS_C/(iN*(iN-1)*iK))
    dRMS_D=np.sqrt(dSS_D/(iN*(iN-1)*iK))
    print("Root-mean-square deviation off-diagonals before transformation: " \
          +str(round(dRMS_C,6)))
    print("Root-mean-square deviation off-diagonals after transformation: " \
          +str(round(dRMS_D,6)))
