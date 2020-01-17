import os
os.environ['QT_QPA_PLATFORM']='offscreen' 
from evcouplings.utils import read_config_file, write_config_file
from evcouplings.utils import read_config_file
from evcouplings.utils.pipeline import execute

config = read_config_file("CXCR4-config.txt")
outcfg = execute(**config)

