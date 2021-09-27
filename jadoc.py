import scipy.linalg
import numpy as np
import time
from numba import njit,prange

def PerformJADOC(mC,iT=100,iTmin=10,dTol=1E-4,dTauH=1E-2,iS=None):
    """Joint Approximate Diagonalization under Orthogonality Constraints
    (JADOC)
    
    Authors: Ronald de Vlaming and Eric Slob
    Repository: https://www.github.com/devlaming/jadoc/
    
    Input
    ------
    mC : np.ndarray with shape (iK, iN, iN)
        iK symmetric PSD iN-by-iN matrices to jointly diagonalize
    
    iT : int, optional
        maximum number of iterations; default=100
    
    iTmin : int, optional
        minimal number of iterations before convergence is tested; default=10
    
    dTol : float, optional
        stop if average magnitude elements gradient<dTol; default=1E-4
    
    dTauH : float, optional
        minimum value of second-order derivatives; default=1E-2
        
    iS : int, optional
        replace mC[i] by rank-iS approximation; default=None
        set to ceil(iN/iK) if None provided
    
    Output
    ------
    mB : np.ndarray with shape (iN, iN)
        matrix such that mB@mC[i]@mB.T is approximately diagonal for all i
    """
    print("Starting JADOC")
    (iK,iN,_)=mC.shape
    if iS is None:
        iS=(iN/iK)
        if (iS-round(iS))>0: iS=round(iS)+1
        else: iS=round(iS)
    if iS==iN: print("Computing decomposition of input matrices")
    elif iS>iN:
        raise ValueError("Desired rank (iS) exceeds dimensionality" \
                         +" of input matrices (iN)")
    else: print("Computing low-dimensional decomposition of input matrices")
    mB=np.eye(iN)
    mA=np.empty((iK,iN,iS))
    dLambda=1
    for i in range(iK):
        if iS<iN:
            (vD,mP)=scipy.linalg.eigh(mC[i],subset_by_index=[iN-iS,iN-1])
        else: (vD,mP)=np.linalg.eigh(mC[i])
        vD[vD<0]=0
        dLambda+=((np.trace(mC[i])-vD.sum())/(iN*iK))
        mA[i]=mP*(np.sqrt(vD)[None,:])
    print("Regularization coefficient = "+str(dLambda))
    (mP,vD,mC)=(None,None,None)
    print("Starting quasi-Newton algorithm with line search (golden section)")
    bConverged=False
    for t in range(iT):
        (dLoss,mDiags,dRMSG,mU)=ComputeLoss(mA,dLambda,dTauH)
        if dRMSG<dTol and t>=iTmin:
            bConverged=True
            break
        dStepSize=PerformGoldenSection(mA,mU,mB,dLambda)
        print("ITER "+str(t)+": L="+str(round(dLoss,3))+", RMSE(g)=" \
              +str(round(dRMSG,6))+", step="+str(round(dStepSize,3)))
        (mB,mA)=UpdateEstimates(mA,mU,mB,dStepSize)
    if not(bConverged):
        print("WARNING: JADOC did not converge. Reconsider data or thresholds")
    print("Returning transformation matrix B")
    return mB

def ComputeLoss(mA,dLambda,dTauH=None,bLossOnly=False):
    mDiags=(np.power(mA,2).sum(axis=2))+dLambda
    (iK,iN,iS)=mA.shape
    dLoss=0.5*(np.log(mDiags).sum())/iK
    if bLossOnly:
        return dLoss
    else:
        mG=ComputeGradient(mA,mDiags)
        dRMSG=np.sqrt((np.power(mG,2).sum())/(iN*(iN-1)))
        mH=(mDiags[:,:,None]/mDiags[:,None,:]).mean(axis=0)
        mH=mH+mH.T-2.0
        mH[mH<dTauH]=dTauH
        mU=-mG/mH
        return dLoss,mDiags,dRMSG,mU

@njit
def ComputeGradient(mA,mDiags):
    (iK,iN,iS)=mA.shape
    mG=np.zeros((iN,iN))
    for i in prange(iK):
        mThisDiag=np.reshape(np.repeat(mDiags[i],iS),(iN,iS))
        mG+=np.dot(mA[i]/mThisDiag,mA[i].T)
    mG=(mG-mG.T)/iK
    return mG

def PerformGoldenSection(mA,mU,mB,dLambda):
    dTheta=2/(1+(5**0.5))
    iIter=0
    iMaxIter=15
    iGuesses=4
    (dStepLB,dStepUB)=(0,1)
    bLossOnlyGold=True
    (iK,iN,iS)=mA.shape
    mR=scipy.linalg.expm(mU)
    mAS=np.empty((iGuesses,iK,iN,iS))
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

def Test():
    iSeed=3426010694
    rng=np.random.default_rng(iSeed)
    iK=10
    iN=100
    print("Simulating "+str(iK)+" PSD matrices, each "+str(iN)+"-by-"+str(iN))
    mC=np.empty((iK,iN,iN))
    for i in range(iK):
        mU=rng.uniform(size=(iN,iN))
        mX=np.ones((iN,iN))
        mX[mU<0.25] = 0
        mX[mU>0.75] = 2
        vF=mX.mean(axis=0)/2
        mX=(mX-(2*vF)[None,:])/(np.sqrt(2*vF*(1-vF))[None,:])
        mC[i]=np.dot(mX,mX.T)/iN
    dTimeStart=time.time()
    mB=PerformJADOC(mC)
    dTime=time.time()-dTimeStart
    print("The JADOC algorithm took "+str(dTime)+" seconds")
    mD=np.empty((iK,iN,iN))
    for i in range(iK):
        mD[i]=np.dot(np.dot(mB,mC[i]),mB.T)
    mDiags=np.diagonal(mD,axis1=1,axis2=2).T
    dSSD=0
    for i in range(iK):
        dSSD+=np.power(mD[i]-np.diag(np.diag(mD[i])),2).sum()
    dRMSOFF = np.sqrt(dSSD/(iN*iN*iK))
    mR=np.corrcoef(mDiags.T)
    vEV=np.linalg.eigvalsh(mR)
    dKappa=max(vEV)/min(vEV)
    print("JADOC RMSD off-diagonal elements: "+str(dRMSOFF))
    print("JADOC Condition number diagonals: "+str(dKappa))

