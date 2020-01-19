import pandas as pd
import numpy as np

#Number of folds
k = 5

dataAddress= "CXCR4_exp.csv"

df = pd.read_csv(dataAddress, sep=",", comment="#")

for f in range(k):
    msk = np.random.rand(len(df)) < 0.8
    tr = df[msk]
    te = df[~msk]
    te.to_csv('testdataset_'+str(f)+'.csv')
    tr.to_csv('trainingdataset_'+str(f)+'.csv')

