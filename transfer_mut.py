from jmhe import jmhe as jmhe
import pandas as pd
from jmhe import score
import pickle

#Define couplings model of source protein and training and testing deep mutatgenesis data
Couplings_model = 'ubiquitin.model'
train_Ad = "trainingdataset.csv"
test_Ad = "testdataset.csv"


# train the source model based on its experiments
tr = pd.read_csv(train_Ad, sep=",", comment="#")
te = pd.read_csv(test_Ad, sep=",", comment="#")

model = jmhe(Couplings_model=Couplings_model)

train_predict_source = model.predict(X=train_Ad)
train_score_source = score(train_predict_source, data_exp=train_Ad)
test_predict_source = model.predict(X=test_Ad)
test_score_source = score(test_predict_source, data_exp=test_Ad)

print('Score before training on the training dataset: ', train_score_source)
print('Score before training on the test dataset: ', test_score_source) 

# Predict the traget domain (labeled as 2) after fitting with experimental data
model.sp = train_score_source[0]
if train_score_source[1]>0:
  model.positiveL = True
else:
  model.positiveL = False

print(model.positiveL)
model.fit(tr)
pickle.dump(model, open('model.pkl', 'wb'))

train_predict_target = model.predict(X=train_Ad)
train_score_target = score(train_predict_target, data_exp=train_Ad)

test_predict_target = model.predict(X=test_Ad)
test_score_target = score(test_predict_target, data_exp=test_Ad)

print('Score after training on the training dataset: ', train_score_target)
print('Score after training on the test dataset: ', test_score_target) 
