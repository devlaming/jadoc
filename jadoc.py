import scipy.linalg
import numpy as np
import time
from numba import njit,prange

def PerformJADOC(mC,iIter=100,dTol=1E-4,iS=None):
    """Joint Approximate Diagonalisation under Orthogonality Constraints
    (JADOC)
    
    Authors: Ronald de Vlaming and Eric Slob
    Last edit: August 23, 2021
    Source: https://www.github.com/devlaming/jadoc/
    
    Input
    ------
    mC : np.ndarray with shape (iC, iN, iN)
        iC symmetric PSD iN-by-iN matrices to jointly diagonalise
    
    iIter : int, optional
        maximum number of iterations; default=100
    
    dTol : float, optional
        stop if average magnitude elements gradient<dTol; default=1E-4
        
    iS : int, optional
        replace mC[i] by rank-iS approximation; default=None
        set to ceil(iN/iC) if None provided
    
    Output
    ------
    mB : np.ndarray with shape (iN, iN)
        matrix such that mB@mC[i]@mB.T is approximately diagonal for all i
    """
    print("Starting JADOC")
    (iC,iN,_)=mC.shape
    dTheta=2/(1+(5**0.5))
    iMinIter=10
    if iS is None:
        iS=(iN/iC)
        if (iS-round(iS))>0: iS=round(iS)+1
        else: iS=round(iS)
    if iS==iN: print("Computing decomposition of input matrices")
    elif iS>iN:
        raise ValueError("Desired rank (iS) exceeds dimensionality" \
                         +" of input matrices (iN)")
    else: print("Computing low-dimensional decomposition of input matrices")
    mB=np.eye(iN)
    mA=np.empty((iC,iN,iS))
    dLambda=1
    for i in range(iC):
        if iS<iN:
            (vD,mP)=scipy.linalg.eigh(mC[i],subset_by_index=[iN-iS,iN-1])
        else: (vD,mP)=np.linalg.eigh(mC[i])
        vD[vD<0]=0
        dLambda+=((np.trace(mC[i])-vD.sum())/(iN*iC))
        mA[i]=mP*(np.sqrt(vD)[None,:])
    print("Regularisation coefficient = "+str(dLambda))
    (mP,vD,mC)=(None,None,None)
    print("Starting quasi-Newton algorithm with golden-section steps")
    for t in range(iIter):
        (dLoss,mDiags,dRMSG,mU)=ComputeLoss(mA,dLambda)
        vEigsRhoDiags=np.linalg.eigvalsh(np.corrcoef(mDiags))
        if dRMSG<dTol and t>=iMinIter:
            break
        dStepSize=PerformGoldenSection(mA,mU,mB,dLambda,dTheta)
        print("ITER "+str(t)+": L="+str(dLoss)+"; RMSE(g)=" \
              +str(dRMSG)+"; kappa(diags)=" \
                  +str(max(vEigsRhoDiags)/min(vEigsRhoDiags)) \
                      +"; step="+str(dStepSize))
        (mB,mA)=UpdateEstimates(mA,mU,mB,dStepSize,iC)
    return mB

def ComputeLoss(mA,dLambda,bLossOnly=False):
    dMinH=0.05
    mDiags=(np.power(mA,2).sum(axis=2))+dLambda
    (iC,iN,iS)=mA.shape
    dLoss=0.5*(np.log(mDiags).sum())/iC
    if bLossOnly:
        return dLoss
    else:
        mG=ComputeGradient(mA,mDiags,iC,iN,iS)
        dRMSG=np.sqrt((np.power(mG,2).sum())/(iN*(iN-1)))
        mH=(mDiags[:,:,None]/mDiags[:,None,:]).mean(axis=0)
        mH=mH+mH.T-2.0
        mH[mH<dMinH]=dMinH
        mU=-mG/mH
        return dLoss,mDiags,dRMSG,mU

@njit
def ComputeGradient(mA,mDiags,iC,iN,iS):
    mG=np.zeros((iN,iN))
    for i in prange(iC):
        mThisDiag=np.reshape(np.repeat(mDiags[i],iS),(iN,iS))
        mG+=np.dot(mA[i]/mThisDiag,mA[i].T)
    mG=(mG-mG.T)/iC
    return mG

def PerformGoldenSection(mA,mU,mB,dLambda,dTheta):
    iIter=0
    iMaxIter=15
    iGuesses=4
    (dStepLB,dStepUB)=(0,1)
    bLossOnly=True
    (iC,iN,iS)=mA.shape
    mR=scipy.linalg.expm(mU)
    mAS=np.empty((iGuesses,iC,iN,iS))
    mAS[0]=mA.copy()
    mAS[1]=RotateData(mR,mA.copy(),iC)
    mAS[2]=(1-dTheta)*mAS[1]+dTheta*mAS[0]
    mAS[3]=(1-dTheta)*mAS[0]+dTheta*mAS[1]
    (mA,mR)=(None,None)
    dLoss2=ComputeLoss(mAS[2],dLambda,bLossOnly)
    dLoss3=ComputeLoss(mAS[3],dLambda,bLossOnly)
    while iIter<iMaxIter:
        if (dLoss2<dLoss3):
            mAS[1]=mAS[3]
            mAS[3]=mAS[2]
            dLoss3=dLoss2
            dStepUB=dStepLB+dTheta*(dStepUB-dStepLB)
            mAS[2]=mAS[1]-dTheta*(mAS[1]-mAS[0])
            dLoss2=ComputeLoss(mAS[2],dLambda,bLossOnly)
        else:
            mAS[0]=mAS[2]
            mAS[2]=mAS[3]
            dLoss2=dLoss3
            dStepLB=dStepUB-dTheta*(dStepUB-dStepLB)
            mAS[3]=mAS[0]+dTheta*(mAS[1]-mAS[0])
            dLoss3=ComputeLoss(mAS[3],dLambda,bLossOnly)
        iIter+=1
    return np.log(1+(dStepLB*(np.exp(1)-1)))

def UpdateEstimates(mA,mU,mB,dStepSize,iC):
    mR=scipy.linalg.expm(dStepSize*mU)
    mB=np.dot(mR,mB)
    mA=RotateData(mR,mA,iC)
    return mB,mA

@njit
def RotateData(mR,mData,iC):
    for i in prange(iC):
        mData[i]=np.dot(mR,mData[i])
    return mData

def Test():
    iSeed=3426010694
    rng=np.random.default_rng(iSeed)
    iC=10
    iN=100
    print("Simulating "+str(iC)+" PSD matrices, each "+str(iN)+"-by-"+str(iN))
    mC=np.empty((iC,iN,iN))
    for i in range(iC):
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
    mD=np.empty((iC,iN,iN))
    for i in range(iC):
        mD[i]=np.dot(np.dot(mB,mC[i]),mB.T)
    mDiags=np.diagonal(mD,axis1=1,axis2=2).T
    dSSD=0
    for i in range(iC):
        dSSD+=np.power(mD[i]-np.diag(np.diag(mD[i])),2).sum()
    dRMSOFF = np.sqrt(dSSD/(iN*iN*iC))
    mR=np.corrcoef(mDiags.T)
    vEV=np.linalg.eigvalsh(mR)
    dKappa=max(vEV)/min(vEV)
    print("JADOC RMSD off-diagonal elements: "+str(dRMSOFF))
    print("JADOC Condition number diagonals: "+str(dKappa))
