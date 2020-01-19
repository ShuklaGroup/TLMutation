# TLMutation

TLmutation: predicting the effects of mutations using transfer learning

TLmutation leverages deep mutational scanning datasets to predict the functional consequences of mutations in homologous proteins.

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

config = read_config_file("config-CXCR4.txt")
outcfg = execute(**config)

```


## 2. Train model on the training data for source protein

Now that we have a model, we will train the model of the source protein with the experimental deep mutational scanning (DMS) dataset.

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
```

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
