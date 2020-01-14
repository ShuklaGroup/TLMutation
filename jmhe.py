import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from evcouplings.couplings import CouplingsModel
import pandas as pd
from evcouplings.mutate import predict_mutation_table, single_mutant_matrix
import scipy.stats as st
import numpy as np
from copy import deepcopy
import pickle
from multiprocessing import Pool

def score(data_pred, data_exp="../exp_training.csv", exp_type='exp1'):
    dz = data_pred['effect_prediction_zar'].values.tolist()
    data_exp = pd.read_csv(data_exp, sep=",", comment="#")
    data_exp_z = data_exp[exp_type].values.tolist()
    ls = []
    for i in range(len(dz)):
        if np.isnan(dz[i]) or np.isnan(data_exp_z[i]):
            ls.append(i)
            print(i)
    for i in range(len(ls)-1,-1,-1):
        del dz[ls[i]]
        del data_exp_z[ls[i]]
    sp0 = st.spearmanr(dz, data_exp_z)[0]
    dz_r = st.rankdata(dz, method="dense")
    data_exp_z_r = st.rankdata(data_exp_z, method="dense")
    sp = st.pearsonr(dz, data_exp_z)[0]
    if np.abs(sp)>np.abs(sp0):
        return 'sp',sp
    return 'sp0',sp0

    
def load_fitness(self, dataFile="halfExp_YAP1.csv", sep=",", comment="#"): # ?!!
    data = pd.read_csv(dataFile, sep=sep, comment=comment)
    return data

class jmhe(object):
    def __init__(self, Couplings_model='YAP1_HUMAN/couplings/YAP1_HUMAN.model'):
        self.c = CouplingsModel(Couplings_model)
        self.off_j_parameters = None
        self.off_h_parameters = None
        self.positiveL = True
        self.sp = 'sp'
        print('The model is built')

    def fit(self, training_data_exp, exp_type='exp1'): 
        positiveL = self.positiveL
        # JM
        p = Pool(processes=8)
        arg = [[training_data_exp, i] for i in range(len(self.c.Jij()))]
        all_spearmanr = p.map(self._multi_run_wrapper_fitStep1_jm, arg)
        all_spearmanr = np.concatenate(all_spearmanr[1:])
        if positiveL:
            all_spearmanr_sorted = np.array(sorted(all_spearmanr, key=lambda all_spearmanr: all_spearmanr[2]))
        else:
            all_spearmanr_sorted = np.array(sorted(all_spearmanr, key=lambda all_spearmanr: -all_spearmanr[2]))
        self._fitStep2_multiP_jm(all_spearmanr_sorted, training_data_exp, exp_type=exp_type)

        ## HE
        #all_spearmanr = self._fitStep1_multiP_he(training_data_exp,  exp_type=exp_type)
        #if positiveL:
        #    all_spearmanr_sorted = np.array(sorted(all_spearmanr, key=lambda all_spearmanr: all_spearmanr[2]))
        #else:
        #    all_spearmanr_sorted = np.array(sorted(all_spearmanr, key=lambda all_spearmanr: -all_spearmanr[2]))
        #self._fitStep2_multiP_he(all_spearmanr_sorted, training_data_exp, exp_type=exp_type)
        
        ## HV
        all_spearmanr = self._fitStep1_multiP_hv(training_data_exp,  exp_type=exp_type)
        if positiveL:
            all_spearmanr_sorted = np.array(sorted(all_spearmanr, key=lambda all_spearmanr: all_spearmanr[1]))
        else:
            all_spearmanr_sorted = np.array(sorted(all_spearmanr, key=lambda all_spearmanr: -all_spearmanr[1]))
        self._fitStep2_multiP_hv(all_spearmanr_sorted, training_data_exp, exp_type=exp_type)        
### HE
    def _fitStep1_multiP_he(self, training_data_exp, exp_type='exp1'):
        processes = 10
        p = Pool(processes=processes)
        arg = [[training_data_exp, i] for i in range(len(self.c.hi()))]
        all_spearmanr = p.map(self._multi_run_wrapper_fitStep1_he, arg)
        all_spearmanr = np.concatenate(all_spearmanr[1:])
        return all_spearmanr
    
    def _multi_run_wrapper_fitStep1_he(self, args):
        return self._singleScanHE(*args) 
    
    def _singleScanHE(self, training_data_exp, iz, exp_type='exp1'):
        # make a zero file
        h_i = deepcopy(self.c.h_i)
        self._zeroh()
        # make a single change in 20x20 
        all_spearmanr = []
        for i in range(20):
            self.c.hi()[iz][i] = h_i[iz][i]
            data_pred = predict_mutation_table(self.c, training_data_exp, "effect_prediction_zar")
            dz = data_pred['effect_prediction_zar'].values.tolist()
            data_exp_z = training_data_exp[exp_type].values.tolist()
            sp = self._score(dz, data_exp_z)
            all_spearmanr.append([iz, i, sp])
            self.c.hi()[iz][i] = 0
        self.c.h_i = deepcopy(h_i)
        return all_spearmanr
  
    def _fitStep2_multiP_he(self, all_spearmanr_sorted, training_data_exp, exp_type='exp1'):
        processes = 10
        p = Pool(processes=processes)
        l = len(all_spearmanr_sorted)//processes
        start = 0
        arg = []
        for i in range(processes-1):
            arg.append([all_spearmanr_sorted, training_data_exp, start, start+l])
            start += l
        arg.append([all_spearmanr_sorted, training_data_exp, start, len(all_spearmanr_sorted)])
        res = p.map(self._multi_run_wrapper_fitStep2_he, arg)
        res = np.array(res)
        h_i_max, pair_max, positiveL, max_sp = res[:,0], res[:,1], res[:,2], res[:,3]
        argmax_SP = np.argmax(max_sp)  # all values of sp are positive at this stage, for both -1 or +1
        self.c.h_i = h_i_max[argmax_SP]
        self.off_h_parameters = all_spearmanr_sorted[:pair_max[argmax_SP]]
        return
    
    def _multi_run_wrapper_fitStep2_he(self, args):
        return self._maximizeHE(*args) 
    
    def _maximizeHE(self, all_spearmanr_sorted, training_data_exp, start, end, exp_type='exp1'):
        # assume positive sp values
        max_sp = 0
        pair_max = -1
        h_i_max = self.c.h_i
        positiveL = self.positiveL
        for pair in range(start):  
            iz = int(all_spearmanr_sorted[pair][0])
            i = int(all_spearmanr_sorted[pair][1])
            self.c.hi()[iz][i] = 0
            self.c.h_i[iz][i] = 0
            
        for pair in range(start, end):  
            iz = int(all_spearmanr_sorted[pair][0])
            i = int(all_spearmanr_sorted[pair][1])
            self.c.hi()[iz][i] = 0
            self.c.h_i[iz][i] = 0            
            data_pred = predict_mutation_table(self.c, training_data_exp, 'effect_prediction_zar')
            dz = data_pred['effect_prediction_zar'].values.tolist()
            data_exp_z = training_data_exp[exp_type].values.tolist()
            sp = self._score(dz, data_exp_z)
            if not positiveL:
                sp = -sp
            if sp>max_sp:
                h_i_max = deepcopy(self.c.h_i)
                pair_max = pair
                max_sp = sp            
        return [h_i_max, pair_max, positiveL, max_sp]

### HV
    def _fitStep1_multiP_hv(self, training_data_exp, exp_type='exp1'):
        processes = 5
        p = Pool(processes=processes)
        arg = [[training_data_exp, i] for i in range(len(self.c.hi()))]
        all_spearmanr = p.map(self._multi_run_wrapper_fitStep1_hv, arg)
        all_spearmanr = np.concatenate(all_spearmanr[1:])
        return all_spearmanr
    
    def _multi_run_wrapper_fitStep1_hv(self, args):
        return self._singleScanHV(*args) 
    
    def _singleScanHV(self, training_data_exp, iz, exp_type='exp1'):
        # make a zero file
        h_i = deepcopy(self.c.h_i)
        self._zeroh()
        # make a single change in 20x20 
        all_spearmanr = []
        for i in range(20):
            self.c.hi()[iz][i] = h_i[iz][i]
        data_pred = predict_mutation_table(self.c, training_data_exp, "effect_prediction_zar")
        dz = data_pred['effect_prediction_zar'].values.tolist()
        data_exp_z = training_data_exp[exp_type].values.tolist()
        sp = self._score(dz, data_exp_z)
        all_spearmanr.append([iz, sp])
        self.c.h_i = deepcopy(h_i)
        return all_spearmanr
  
    def _fitStep2_multiP_hv(self, all_spearmanr_sorted, training_data_exp, exp_type='exp1'):
        processes = 5
        p = Pool(processes=processes)
        l = len(all_spearmanr_sorted)//processes
        start = 0
        arg = []
        for i in range(processes-1):
            arg.append([all_spearmanr_sorted, training_data_exp, start, start+l])
            start += l
        arg.append([all_spearmanr_sorted, training_data_exp, start, len(all_spearmanr_sorted)])
        res = p.map(self._multi_run_wrapper_fitStep2_hv, arg)
        res = np.array(res)
        h_i_max, pair_max, positiveL, max_sp = res[:,0], res[:,1], res[:,2], res[:,3]
        argmax_SP = np.argmax(max_sp)  # all values of sp are positive at this stage, for both -1 or +1
        self.c.h_i = h_i_max[argmax_SP]
        self.off_h_parameters = all_spearmanr_sorted[:pair_max[argmax_SP]]
        return
    
    def _multi_run_wrapper_fitStep2_hv(self, args):
        return self._maximizeHV(*args) 
    
    def _maximizeHV(self, all_spearmanr_sorted, training_data_exp, start, end, exp_type='exp1'):
        # assume positive sp values
        max_sp = 0
        pair_max = -1
        h_i_max = self.c.h_i
        positiveL = self.positiveL
        for pair in range(start):  
            iz = int(all_spearmanr_sorted[pair][0])
            for i in range(20):
                self.c.hi()[iz][i] = 0
                self.c.h_i[iz][i] = 0
            
        for pair in range(start, end):  
            iz = int(all_spearmanr_sorted[pair][0])
            for i in range(20):
                self.c.hi()[iz][i] = 0
                self.c.h_i[iz][i] = 0            
            data_pred = predict_mutation_table(self.c, training_data_exp, 'effect_prediction_zar')
            dz = data_pred['effect_prediction_zar'].values.tolist()
            data_exp_z = training_data_exp[exp_type].values.tolist()
            sp = self._score(dz, data_exp_z)
            if not positiveL:
                sp = -sp
            if sp>max_sp:
                h_i_max = deepcopy(self.c.h_i)
                pair_max = pair
                max_sp = sp            
        return [h_i_max, pair_max, positiveL, max_sp]
    
### JM   
    def _multi_run_wrapper_fitStep1_jm(self, args):
        return self._singleScanJM(*args)
    
    def _singleScanJM(self, training_data_exp, iz, exp_type='exp1'):
        # make a zero file
        J_ij = deepcopy(self.c.J_ij)
        self._zeroJ()
        # make a single change in 20x20 
        all_spearmanr = []
        for jz in range(iz):
            for i in range(20):
                for j in range(20):
                    self.c.Jij()[iz][jz][i][j] = J_ij[iz][jz][i][j]
            data_pred = predict_mutation_table(self.c, training_data_exp, "effect_prediction_zar")
            dz = data_pred['effect_prediction_zar'].values.tolist()
            data_exp_z = training_data_exp[exp_type].values.tolist()
            sp = self._score(dz, data_exp_z)
            all_spearmanr.append([iz, jz, sp])
            for i in range(20):
                for i in range(20):
                    self.c.Jij()[iz][jz][i][j] = 0
        self.c.J_ij = deepcopy(J_ij)
        return all_spearmanr
    
    def _fitStep2_multiP_jm(self, all_spearmanr_sorted, training_data_exp, exp_type='exp1'):
        processes = 10
        p = Pool(processes=processes)
        l = len(all_spearmanr_sorted)//processes
        start = 0
        arg = []
        for i in range(processes-1):
            arg.append([all_spearmanr_sorted, training_data_exp, start, start+l])
            start += l
        arg.append([all_spearmanr_sorted, training_data_exp, start, len(all_spearmanr_sorted)])
        res = p.map(self._multi_run_wrapper_fitStep2_jm, arg)
        res = np.array(res)
        J_ij_max, pair_max, positiveL, max_sp = res[:,0], res[:,1], res[:,2], res[:,3]
        argmax_SP = np.argmax(max_sp) # all values of sp are positive at this stage, for both -1 or +1
        self.c.J_ij = J_ij_max[argmax_SP]
        self.off_parameters = all_spearmanr_sorted[:pair_max[argmax_SP]]
        return
    
    def _multi_run_wrapper_fitStep2_jm(self, args):
        return self._maximizeJM(*args) 
    
    def _maximizeJM(self, all_spearmanr_sorted, training_data_exp, start, end, exp_type='exp1'):
        # assume positive sp values
        max_sp = 0
        pair_max = -1
        J_ij_max = self.c.J_ij
        positiveL = self.positiveL
        for pair in range(start):
            iz = int(all_spearmanr_sorted[pair][0])
            jz = int(all_spearmanr_sorted[pair][1])
            for j in range(20):
                for i in range(20):
                    self.c.Jij()[iz][jz][i][j] = 0
        
        for pair in range(start, end):  
            iz = int(all_spearmanr_sorted[pair][0])
            jz = int(all_spearmanr_sorted[pair][1])
            for i in range(20):
                for j in range(20):
                    self.c.Jij()[iz][jz][i][j] = 0
            data_pred = predict_mutation_table(self.c, training_data_exp, 'effect_prediction_zar')
            dz = data_pred['effect_prediction_zar'].values.tolist()
            data_exp_z = training_data_exp[exp_type].values.tolist()
            sp = self._score(dz, data_exp_z)
            if not positiveL:
                sp = -sp
            if sp>max_sp:
                J_ij_max = deepcopy(self.c.J_ij)
                pair_max = pair
                max_sp = sp
        return [J_ij_max, pair_max, positiveL, max_sp]
### Others
    def predict(self, X="../../../Second_halfExp_YAP1.csv"):
        data = pd.read_csv(X, sep=",", comment="#")
        data_pred = predict_mutation_table(self.c, data, "effect_prediction_zar")
        return data_pred

    def _score(self, dz, data_exp_z):
        ls = []
        for i in range(len(dz)):
            if np.isnan(dz[i]) or np.isnan(data_exp_z[i]):
                ls.append(i)
        for i in range(len(ls)-1,-1,-1):
            del dz[ls[i]]
            del data_exp_z[ls[i]]
        if self.sp=='sp0':
            sp0 = st.spearmanr(dz, data_exp_z)[0]
            return sp0
        dz_r = st.rankdata(dz, method="dense")
        data_exp_z_r = st.rankdata(data_exp_z, method="dense")
        sp = st.pearsonr(dz, data_exp_z)[0]
        return sp
    
    def _zeroh(self):
        for i1 in range(len(self.c.hi())):
            for i3 in range(len(self.c.hi()[i1])):
                   self.c.hi()[i1][i3] = 0
    def _zeroJ(self):
        for i1 in range(len(self.c.Jij())):
            for i2 in range(len(self.c.Jij()[i1])):
                for i3 in range(len(self.c.Jij()[i1][i2])):
                    for i4 in range(len(self.c.Jij()[i1][i2][i3])):
                        self.c.Jij()[i1][i2][i3][i4] = 0
                        
    def _whole_seq_mapping(self, algn_file='YAP1.txt'):
        z = open(algn_file, 'r')
        a = z.readline()
        b = z.readline()
        a1 = a.split()[0]
        b1 = b.split()[0]
        map = {}
        i1, i2 = -1, -1
        for i in range(len(b1)):
            if a1[i]!='-':
                i1+=1
            if b1[i]!='-':
                i2+=1
            if a1[i]!='-' and b1[i]!='-':
                map[i1] = i2 
        return map

    def _mapping(self, target_model, algn_file='YAP1.txt'): 
        """
        Source to target
        """
        map = {}
        index_list_source = self.c.index_list
        index_list_target = target_model.c.index_list
        whole_seq_map = self._whole_seq_mapping(algn_file=algn_file)
        
        for i_s in range(len(index_list_source)):
            seq_index_s = index_list_source[i_s]
            try:
                seq_index_t = whole_seq_map[seq_index_s]  #find index_s in alignment and its corresponding index_t
                i_t = np.where(index_list_target==seq_index_t)[0][0] #find i_t corresponding to index_t
                map[i_s] = i_t
            except:
                pass
        return map
          
    def transfer(self, Couplings_model_target='Couplings_model_target.md', transferType='mutant', algn_file='YAP1.txt'):
    # different transferring types
        model_target = jmhe(Couplings_model_target)
        if transferType=='mutant':
            map = self._mapping(model_target, algn_file=algn_file)
            self._transfer_mutant(model_target, map)
        return model_target
    
    def _transfer_mutant(self, modelB, map):
        for parameter in self.off_parameters: #off_j_parameters #jm
            iz_1, jz_1, sp = int(parameter[0]), int(parameter[1]), int(parameter[2])
            try: #JM
                iz_2, jz_2 = map[iz_1], map[jz_1]
                for i in range(20):
                    for j in range(20):
                        modelB.c.Jij()[iz_2][jz_2][i][j] = 0
            except:
                pass
        for parameter in self.off_h_parameters:#off_j_parameters #he
            iz, i, sp = int(parameter[0]), int(parameter[1]), int(parameter[2])
            try:
                iz_2 = map[iz]
                modelB.c.hi()[iz_2][i] = 0
            except:
                pass
        return 
    
    def _transfer_mutant_jv(self, modelB, map):
        for parameter in self.off_parameters: #off_j_parameters #jv
            iz_1, jz_1, i = int(parameter[0]), int(parameter[1]), int(parameter[2])
            try: #JV
                iz_2, jz_2 = map[iz_1], map[jz_1]
                for j in range(20):
                    modelB.c.Jij()[iz_2][jz_2][i][j] = 0
            except:
                pass
        return 
    
    def matrx2vec(self, matrxDataFile='average_expression_YAP1.csv'):
        df = pd.read_csv(matrxDataFile)
        my_list = []
        for j in range(2,335):
            for i in range(1,22):
                w = df.at[0, str(j)]
                mut = df.at[i,'Unnamed: 1']
                val = df.at[i,str(j)]
                seq = j
                char = w+str(j)+mut
                my_list.append([char, val])
        return pd.DataFrame(my_list)
    

