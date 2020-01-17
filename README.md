# TLMutation

TLmutation: predicting the effects of mutations using transfer learning

TLmutation leverages deep mutational scanning datasets to predict the functional consequences of mutations in homologous proteins.

## 1. Building a model for both the source and target protein
TLmutation first requires a model to represent the protein sequence. The current iteration of TLmutation uses a Pott's model and is built using the EVcouplings package. 

https://github.com/debbiemarkslab/EVcouplings

<<<<<<< HEAD
<<<<<<< HEAD

Below is an example script to execute the EVcoupling pipeline.
=======
>>>>>>> 9d7cb94aa3bad5d260e5da0d0f687ebf83e0e811
=======
>>>>>>> 9d7cb94aa3bad5d260e5da0d0f687ebf83e0e811
```
import os
os.environ['QT_QPA_PLATFORM']='offscreen' 
from evcouplings.utils import read_config_file, write_config_file
from evcouplings.utils import read_config_file
from evcouplings.utils.pipeline import execute

<<<<<<< HEAD
<<<<<<< HEAD
config = read_config_file("config-CXCR4.txt")
=======
config = read_config_file("config-ubiquitin.txt")
>>>>>>> 9d7cb94aa3bad5d260e5da0d0f687ebf83e0e811
=======
config = read_config_file("config-ubiquitin.txt")
>>>>>>> 9d7cb94aa3bad5d260e5da0d0f687ebf83e0e811
outcfg = execute(**config)

```



## 2. Train model on the training data for Source protein 
