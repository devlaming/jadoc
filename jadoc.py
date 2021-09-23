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
    mBA=np.empty((iC,iN,iS))
    dLambda=1
    for i in range(iC):
        if iS<iN:
            (vD,mP)=scipy.linalg.eigh(mC[i],subset_by_index=[iN-iS,iN-1])
        else: (vD,mP)=np.linalg.eigh(mC[i])
        vD[vD<0]=0
        dLambda+=((np.trace(mC[i])-vD.sum())/(iN*iC))
        mBA[i]=mP*(np.sqrt(vD)[None,:])
    print("Regularisation coefficient = "+str(dLambda))
    (mP,vD,mC)=(None,None,None)
    print("Starting quasi-Newton algorithm with golden-section steps")
    for t in range(iIter):
        (dLoss,mDiags,dRMSG,mU)=ComputeLoss(mBA,dLambda)
        vEigsRhoDiags=np.linalg.eigvalsh(np.corrcoef(mDiags))
        if dRMSG<dTol and t>=iMinIter:
            break
        dStepSize=PerformGoldenSection(mBA,mU,mB,dLambda,dTheta)
        print("ITER "+str(t)+": L="+str(dLoss)+"; RMSE(g)=" \
              +str(dRMSG)+"; kappa(diags)=" \
                  +str(max(vEigsRhoDiags)/min(vEigsRhoDiags)) \
                      +"; step="+str(dStepSize))
        (mB,mBA)=UpdateEstimates(mB,dStepSize,mU,mBA,iC)
    return mB

def ComputeLoss(mBA,dLambda,bLossOnly=False):
    dMinH=0.05
    mDiags=(np.power(mBA,2).sum(axis=2))+dLambda
    (iC,iN,iS)=mBA.shape
    dLoss=0.5*(np.log(mDiags).sum())/iC
    if bLossOnly:
        return dLoss
    else:
        mG=ComputeGradient(mBA,mDiags,iC,iN,iS)
        dRMSG=np.sqrt((np.power(mG,2).sum())/(iN*(iN-1)))
        mH=(mDiags[:,:,None]/mDiags[:,None,:]).mean(axis=0)
        mH=mH+mH.T-2.0
        mH[mH<dMinH]=dMinH
        mU=-mG/mH
        return dLoss,mDiags,dRMSG,mU

@njit
def ComputeGradient(mBA,mDiags,iC,iN,iS):
    mG=np.zeros((iN,iN))
    for i in prange(iC):
        mThisDiag=np.reshape(np.repeat(mDiags[i],iS),(iN,iS))
        mG+=np.dot(mBA[i]/mThisDiag,mBA[i].T)
    mG=(mG-mG.T)/iC
    return mG

def PerformGoldenSection(mBA,mU,mB,dLambda,dTheta):
    iIter=0
    iMaxIter=15
    iGuesses=4
    (dStepLB,dStepUB)=(0,1)
    bLossOnly=True
    (iC,iN,iS)=mBA.shape
    mExpU=scipy.linalg.expm(mU)
    m4BS=np.empty((iGuesses,iC,iN,iS))
    m4BS[0]=mBA.copy()
    m4BS[1]=TransformData(mExpU,mBA.copy(),iC)
    m4BS[2]=(1-dTheta)*m4BS[1]+dTheta*m4BS[0]
    m4BS[3]=(1-dTheta)*m4BS[0]+dTheta*m4BS[1]
    (mBA,mExpU)=(None,None)
    dLoss2=ComputeLoss(m4BS[2],dLambda,bLossOnly)
    dLoss3=ComputeLoss(m4BS[3],dLambda,bLossOnly)
    while iIter<iMaxIter:
        if (dLoss2<dLoss3):
            m4BS[1]=m4BS[3]
            m4BS[3]=m4BS[2]
            dLoss3=dLoss2
            dStepUB=dStepLB+dTheta*(dStepUB-dStepLB)
            m4BS[2]=m4BS[1]-dTheta*(m4BS[1]-m4BS[0])
            dLoss2=ComputeLoss(m4BS[2],dLambda,bLossOnly)
        else:
            m4BS[0]=m4BS[2]
            m4BS[2]=m4BS[3]
            dLoss2=dLoss3
            dStepLB=dStepUB-dTheta*(dStepUB-dStepLB)
            m4BS[3]=m4BS[0]+dTheta*(m4BS[1]-m4BS[0])
            dLoss3=ComputeLoss(m4BS[3],dLambda,bLossOnly)
        iIter+=1
    return np.log(1+(dStepLB*(np.exp(1)-1)))

def UpdateEstimates(mB,dStepSize,mU,mBA,iC):
    mExpU=scipy.linalg.expm(dStepSize*mU)
    mB=np.dot(mExpU,mB)
    mBA=TransformData(mExpU,mBA,iC)
    return mB,mBA

@njit
def TransformData(mT,mData,iC):
    for i in prange(iC):
        mData[i]=np.dot(mT,mData[i])
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
    mR = np.corrcoef(mDiags.T)
    vEV = np.linalg.eigvalsh(mR)
    dKappa = max(vEV)/min(vEV)
    print("JADOC RMSD off-diagonal elements: " + str(dRMSOFF))
    print("JADOC Condition number diagonals: " + str(dKappa))
