from jmhe import jmhe 
import pandas as pd
from jmhe import score
import pickle

Couplings_model_source = '/home/zshamsi2/projects/functionalCouplings/CXCR4-CCR5/jmhe/CXCR4/Model.model'
dataAdress_source = "/home/zshamsi2/projects/functionalCouplings/CXCR4-CCR5/jmhe/CXCR4/BiFC/exp.csv"

# Train the source model based on its own DMS experiments
data = pd.read_csv(dataAdress_source, sep=",", comment="#")
model_source = jmhe(Couplings_model=Couplings_model_source)

# Predict from initial model
dx2 = model_source.predict(X=dataAdress)
s1 = score(dx2, data_exp=dataAdress)

print('Score for CXCR4 before training: ', s1)

# Predict after fitting with experimental data

model_source.sp = s1[0]
if s1[1]>0:
  model_source.positiveL = True
else:
  model_source.positiveL = False
print(model_source.positiveL)
model_source.fit(data)
pickle.dump(model_source, open('model_cxcr4.pkl', 'wb'))
#model_source = pickle.load(open('model_cxcr4.pkl', 'rb'))

dx = model_source.predict(X=dataAdress)

# Calculate the Spearman's correlation score
s2 = score(dx, data_exp=dataAdress)
print('Score for CXCR4 after training: ', s2)


# For comparison, predict the target model without transfer

# Model of target protein
Couplings_model_target='/home/zshamsi2/projects/functionalCouplings/CXCR4-CCR5/jmhe/CCR5/Model.model'

# Path of experimental DMS dataset
dataAdress_target = "/home/zshamsi2/projects/functionalCouplings/CXCR4-CCR5/jmhe/CCR5/BiFC/exp.csv"
data_target_target = pd.read_csv(dataAdress_target, sep=",", comment="#")

model_target = jmhe(Couplings_model=Couplings_model_target)

dx2 = model_target.predict(X=dataAdress_target)

# Calculate the Spearman's correlation score
s1 = score(dx2, data_exp=dataAdress_target)
print('Score for CCR5 before transfer: ',s1)


# Transfer from source protein to target protein 

# Sequence alignment file can be generated from any alignment server or program 
algn_file = '/home/zshamsi2/projects/functionalCouplings/CXCR4-CCR5/case_PMEP_jmhe/CXCR4-CCR5.aln'

model_target_transfer = model_source.transfer(Couplings_model_target=Couplings_model_target, algn_file=algn_file)
#pickle.dump(model_target_transfer, open('model_ccr5_transfered.pkl', 'wb'))

dx2_transfer = model_target_transfer.predict(X=dataAdress_target)

# Calculate the Spearman's correlation score
s1_transfer = score(dx2_transfer, data_exp=dataAdress_target)
print('Score for CCR5 after transfer from CCR5', s1_transfer)



