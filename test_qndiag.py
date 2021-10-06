import sys
import numpy as np
import pandas as pd
import time
import jadoc
import qndiag # source: https://github.com/pierreablin/qndiag/blob/master/qndiag/qndiag.py ; accessed: 28/09/2021

iR=int(sys.argv[1])
vAlpha=[0,0.25,0.5,0.75]
vN=[100,200,300,400,500]
vK=[2,4,8,16,32]
dfTime_N=pd.DataFrame(index=vAlpha,columns=vN)
dfRMSD_N=pd.DataFrame(index=vAlpha,columns=vN)
dfTime_K=pd.DataFrame(index=vAlpha,columns=vK)
dfRMSD_K=pd.DataFrame(index=vAlpha,columns=vK)
jadoc.Test()
for dAlpha in vAlpha:
    for iN in vN:
        iK=10
        mC=jadoc.SimulateData(iK,iN,iR,dAlpha)
        dT0=time.time()
        (mB,_)=qndiag.qndiag(mC,ortho=True)
        dT=time.time()-dT0
        mD=np.empty((iK,iN,iN))
        for i in range(iK):
            mD[i]=np.dot(np.dot(mB,mC[i]),mB.T)
        dSSD=0
        for i in range(iK):
            dSSD+=np.power(mD[i]-np.diag(np.diag(mD[i])),2).sum()
        dRMSD=np.sqrt(dSSD/(iN*(iN-1)*iK))
        dfTime_N.loc[dAlpha,iN]=dT
        dfRMSD_N.loc[dAlpha,iN]=dRMSD
    for iK in vK:
        iN=256
        mC=jadoc.SimulateData(iK,iN,iR,dAlpha)
        dT0=time.time()
        (mB,_)=qndiag.qndiag(mC,ortho=True)
        dT=time.time()-dT0
        mD=np.empty((iK,iN,iN))
        for i in range(iK):
            mD[i]=np.dot(np.dot(mB,mC[i]),mB.T)
        dSSD=0
        for i in range(iK):
            dSSD+=np.power(mD[i]-np.diag(np.diag(mD[i])),2).sum()
        dRMSD=np.sqrt(dSSD/(iN*(iN-1)*iK))
        dfTime_K.loc[dAlpha,iK]=dT
        dfRMSD_K.loc[dAlpha,iK]=dRMSD
sPre='qndiag.'
sTime='time.'
sRMSD='rmsd.'
sN='alpha_rows.n_cols.'
sK='alpha_rows.k_cols.'
sRun='run.'+str(iR)
sExt='.csv'
dfTime_N.to_csv(sPre+sTime+sN+sRun+sExt)
dfRMSD_N.to_csv(sPre+sRMSD+sN+sRun+sExt)
dfTime_K.to_csv(sPre+sTime+sK+sRun+sExt)
dfRMSD_K.to_csv(sPre+sRMSD+sK+sRun+sExt)
