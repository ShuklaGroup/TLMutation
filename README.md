# TLMutation

TLmutation: predicting the effects of mutations using transfer learning

TLmutation leverages deep mutational scanning datasets to predict the functional consequences of mutations in homologous proteins. Find our publication here: https://pubs.acs.org/doi/full/10.1021/acs.jpcb.0c00197

In this example, we will use TLmutation to transfer from chemokine receptors CXCR4 to CCR5.

## 1. Building a model for both the source and target protein
TLmutation first requires a model to represent the protein sequence. The current iteration of TLmutation uses a Pott's model and is built using the EVcouplings package.

https://github.com/debbiemarkslab/EVcouplings

Below is an example script to execute the EVcoupling pipeline.
```
import os
os.environ['QT_QPA_PLATFORM']='offscreen' 
from evcouplings.utils import read_config_file, write_config_file
from evcouplings.utils import read_config_file
from evcouplings.utils.pipeline import execute

source_protein_config = read_config_file("CXCR4-config.txt")
source_protein_outcfg = execute(**source_protein_config)

target_protein_config = read_config_file("CCR5-config.txt")
target_protein_outcfg = execute(**target_protein_config)
```


## 2. Supervised transfer learning to a functional assay of the source protein
In the current iteration of TLmutation, the supervised transfer is from the evolutionary statistical energy to a functional assay. Here, deep mutational scanning (DMS) is the studied experimental assay.

The DMS dataset should be in a .csv format as shown.

```
mutant,linear
...
E2L,0.605017852
E2I,-2.179323351
E2V,0.148787296
E2A,-0.222549939
E2S,0.32657855
E2T,-1.829107102
E2N,1.136931418
E2Q,-0.18022787
E2D,-0.929420795
...
G3V,-1.085390056
G3A,-0.943412925
G3S,0.372614715
G3T,-0.321523109
G3N,-1.660989957
G3Q,1.538280976
G3D,-2.056820679
...

```

Now that we have a model, we will incorporate the experimental DMS dataset into the model of the source protein (CXCR4).

The following script is a part of ``transfer_mut.py``. 
```
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
dx2 = model_source.predict(X=dataAdress_source)
s1 = score(dx2, data_exp=dataAdress_source)

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

dx = model_source.predict(X=dataAdress_source)

# Calculate the Spearman's correlation score
s2 = score(dx, data_exp=dataAdress_source)
print('Score for CXCR4 after training: ', s2)
```

After training the model on the DMS data, we observed an increase in the Spearman's correlation, signifying the incorporation of experimental data greatly benifits the model.

```
Score for CXCR4 before training:  ('sp', 0.19921471564043003)
Score for CXCR4 after training:  ('sp', 0.51009788834287362)
```


## 3. Unsupervised transfer between proteins: Transfer the weights to the target protein model

First and as a comparison, we will evaluate how the model will fair without transfer learning.

```
# Transfer from source protein to target protein 
Couplings_model_target='/home/zshamsi2/projects/functionalCouplings/CXCR4-CCR5/jmhe/CCR5/Model.model'
model_target = jmhe(Couplings_model=Couplings_model_target)

#DMS dataset for the target protein, for evaluating purposes
dataAdress_target = "/home/zshamsi2/projects/functionalCouplings/CXCR4-CCR5/jmhe/CCR5/BiFC/exp.csv"
data_target_target = pd.read_csv(dataAdress_target, sep=",", comment="#")


# To compare, let's see how predicting the target protein without transfer will do
dx2 = model_target.predict(X=dataAdress_target)

# Calculate the Spearman's correlation score without transfer to the actual DMS dataset
s1 = score(dx2, data_exp=dataAdress_target)
print('Score for CCR5 before transfer: ',s1)
```

Without transfer, we recieve the following Spearmans' coorelation.
```
Score for CCR5 before transfer:  ('sp', 0.32721716144910645)
```


Let us now see how transfering from the source protein compare. We will now transfer the weights obtained from the source model to the target model (CCR5). This requires a sequence alignment file of the two proteins and be generated from any sequence alignment server or program.

``CXCR4-CCR5.aln``
```
MDYQVSSPIYDIN-----------YYTSEPCQKINVKQIAARLLPPLYSLVFIFGFVGNMLVILILINCKRLKSMTDIYLLNLAISDLFFLLTVPFWAHYAAAQWDFGNTMCQLLTGLYFIGFFSGIFFIILLTIDRYLAVVHAVFALKARTVTFGVVTSVITWVVAVFASLPGIIFTRSQKEGLHYTCSSHFPYSQYQFWKNFQTLKIVILGLVLPLLVMVICYSGILKTLLRCRNEKKRHRAVRLIFTIMIVYFLFWAPYNIVLLLNTFQEFF-GLNNCSSSNRLDQAMQVTETLGMTHCCINPIIYAFVGEKFRNYLLVFFQKHIAKRFCKCCSIFQQEAPERASSVYTRSTGEQEISVGL
---MEGISIYTSDNYTEEMGSGDYDSMKEPCFREENANFNKIFLPTIYSIIFLTGIVGNGLVILVMGYQKKLRSMTDKYRLHLSVADLLFVITLPFWAVDAVANWYFGNFLCKAVHVIYTVNLYSSVLILAFISLDRYLAIVHATNSQRPRKLLAEKVVYVGVWIPALLLTIPDFIFANVSEADDRYICDRFYPND---LWVVVFQFQHIMVGLILPGIVILSCYCIIISKLSHSKGHQKR-KALKTTVILILAFFACWLPYYIGISIDSFILLEIIKQGCEFENTVHKWISITEALAFFHCCLNPILYAFLGAKFKTSAQHALTSVSRGS---SLKILSKGKRGGHSSVSTESESSSFHSS--
``` 


```
# Sequence alignment file can be generated from any alignment server or program 
algn_file = '/home/zshamsi2/projects/functionalCouplings/CXCR4-CCR5/case_PMEP_jmhe/CXCR4-CCR5.aln'

model_target_transfer = model_source.transfer(Couplings_model_target=Couplings_model_target, algn_file=algn_file)
#pickle.dump(model_target_transfer, open('model_ccr5_transfered.pkl', 'wb'))

dx2_transfer = model_target_transfer.predict(X=dataAdress_target)

# Calculate the Spearman's correlation score after transfer to the actual DMS dataset
s1_transfer = score(dx2_transfer, data_exp=dataAdress_target)
print('Score for CCR5 after transfer from CCR5', s1_transfer)
```

After transfer, we obtain the Spearman's coorelation of the target protein after transfer and is greater than without the transfer of knowledge from the source protein.

```
Score for CCR5 after transfer from CCR5 ('sp', 0.34294314055843561)
```


