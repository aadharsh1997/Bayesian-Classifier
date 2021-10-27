#!/usr/bin/env python
# coding: utf-8

# In[1]:


from enum import auto
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pomegranate import *
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
import pandas as pd
import networkx as nx


dataset= pd.read_excel("A:\Aadharsh\Repo\Test\COVID-19_formatted_dataset.xlsx")
dataset_1= pd.read_excel("A:\Aadharsh\Repo\Test\COVID-19_formatted_dataset.xlsx")

dataset["SARS-Cov-2 exam result"].replace({"negative": 0, "positive": 1}, inplace=True)
target = dataset["SARS-Cov-2 exam result"]
dataset.drop("SARS-Cov-2 exam result", axis=1,inplace=True)

dataset_1["SARS-Cov-2 exam result"].replace({"negative": 0, "positive": 1}, inplace=True)
target_1 = dataset_1['SARS-Cov-2 exam result']
dataset_1.drop("SARS-Cov-2 exam result", axis=1,inplace=True)
dataset_1.drop("Patient ID",axis=1,inplace=True)

dataset_numpy=dataset_1.values
target_numpy=target_1.values

x_train, x_test, y_train, y_test = train_test_split(dataset_numpy, target_numpy, test_size=0.2, random_state=0,shuffle=True)
print(y_test)

column = list(dataset.columns)
features = set(column) - set(['SARS-Cov-2 exam result'])
for i in features:
    sns.histplot(dataset,x=i,hue=target)
    plt.show()


# In[2]:


disc_enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
x_train_binned = disc_enc.fit_transform(x_train)
x_test_binned = disc_enc.transform(x_test)

train_data = np.concatenate([x_train_binned, y_train.reshape(-1, 1)], axis=1)
test_data = np.concatenate([x_test_binned, np.empty_like(y_test).reshape(-1, 1)], axis=1)
test_data[:, -1] = np.nan

model = BayesianNetwork().from_samples(train_data, algorithm='exact')
pred = np.array(model.predict(test_data)).astype(int)
prediction=pred[:,-1]
  
print(cm(y_test,prediction))
print(accuracy_score(y_test,prediction))


# In[3]:


ind = np.random.randint(0, len(x_test), 5)
Sample_test = np.concatenate([x_test_binned[ind],np.empty_like(y_test[ind]).reshape(-1, 1)], axis=1)
Sample_test[:,-1]=np.nan
sample_pred=np.array(model.predict(Sample_test)).astype(int)
sample_prediction=sample_pred[:,-1]
print(y_test[ind])
print(sample_prediction)

probs = model.predict_proba(Sample_test)
print(probs)

for i in range(5):
    print('Sample {}(actual: {}): Positive Probability: {:4.2f} %'
          .format(i, y_test[ind[i]], probs[i][-1].parameters[0][1] * 100))#parameter[0][1]is the probability of covid positive


# In[4]:


Patient_blood_test = [[3.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 1.0, 1.0,3.0, 0.0, 3.0, 2.0, 0.0, None]]
sample_pred_1=np.array(model.predict(Patient_blood_test)).astype(int)
sample_prob_1=model.predict_proba(Patient_blood_test)
sample_prediction=sample_pred_1[:,-1]
print(sample_prediction)
print(sample_prob_1[0][-1].parameters[0][1])


# In[5]:


dataset_2= pd.read_excel("A:\Aadharsh\Repo\Test\COVID-19_formatted_dataset.xlsx")
dataset_2["SARS-Cov-2 exam result"].replace({"negative": 0, "positive": 1}, inplace=True)
for i in range(0,len(dataset_2["Hematocrit"])):
    if dataset_2['Hematocrit'][i] > 0:
        dataset_2['Hematocrit'][i]=1.0
    else:
        dataset_2['Hematocrit'][i]=0.0

for i in range(0,len(dataset_2["Hemoglobin"])):
    if dataset_2['Hemoglobin'][i] > 0:
        dataset_2['Hemoglobin'][i]=1.0
    else:
        dataset_2['Hemoglobin'][i]=0

for i in range(0,len(dataset_2["Lymphocytes"])):
    if dataset_2['Lymphocytes'][i] > 0:
        dataset_2['Lymphocytes'][i]=1.0
    else:
        dataset_2['Lymphocytes'][i]=0.0


def prob_calc(data, child, parent1=None, parent2=None):
    one=0
    zero=0
    zero_zero=0
    zero_one=0
    one_zero=0
    one_one=0
    zero_given_zero=0
    zero_given_one=0
    one_given_zero=0
    one_given_one=0
    zero_given_zero_zero=0
    zero_given_zero_one=0
    zero_given_one_zero=0
    zero_given_one_one=0
    one_given_zero_zero=0
    one_given_zero_one=0
    one_given_one_zero=0
    one_given_one_one=0
    if parent1==None:
        for i in data[child]:
            if i==1:
                one+=1
            else:
                zero+=1
        print([zero/598,one/598])
        return ([zero/598,one/598])
        
    else:
        if parent2==None:
            for i in data[parent1]:
                if i==1:
                    one+=1
                else:
                    zero+=1
            for i,j in zip(data[parent1],data[child]):
                if (i==0) and (j==0):
                    zero_given_zero+=1
                elif (i==0) and (j==1):
                    zero_given_one+=1
                elif (i==1) and (j==0):
                    one_given_zero+=1
                elif (i==1) and (j==1):
                    one_given_one+=1
            print([zero_given_zero/zero,zero_given_one/one,one_given_zero/zero,one_given_one/one])
            return([zero_given_zero/zero,zero_given_one/one,one_given_zero/zero,one_given_one/one])
        elif parent2!=None:
            for i,j in zip(data[parent1],data[parent2]):
                if (i==0) and (j==0):
                    zero_zero+=1
                elif (i==0) and (j==1):
                    zero_one+=1
                elif (i==1) and (j==0):
                    one_zero+=1
                elif (i==1) and (j==1):
                    one_one+=1
            for i,j,k in zip(data[parent1],data[parent2],data[child]):
                if (i==0) and (j==0) and (k==0):
                    zero_given_zero_zero+=1
                elif (i==0) and (j==0) and (k==1):
                     zero_given_zero_one+=1
                elif (i==0) and (j==1) and (k==0):
                    zero_given_one_zero+=1
                elif (i==0) and (j==1) and (k==1):
                    zero_given_one_one+=1
                elif (i==1) and (j==0) and (k==0):
                    one_given_zero_zero+=1
                elif (i==1) and (j==0) and (k==1):
                    one_given_zero_one+=1
                elif (i==1) and (j==1) and (k==0):
                    one_given_one_zero+=1
                elif (i==1) and (j==1) and (k==1):
                    one_given_one_one+=1
            print([zero_given_zero_zero/zero_zero,zero_given_zero_one/zero_one,zero_given_one_zero/one_zero,zero_given_one_one/one_one,one_given_zero_zero/zero_zero,one_given_zero_one/zero_one,one_given_one_zero/one_zero,one_given_one_one/one_one])
            return([zero_given_zero_zero/zero_zero,zero_given_zero_one/zero_one,zero_given_one_zero/one_zero,zero_given_one_one/one_one,one_given_zero_zero/zero_zero,one_given_zero_one/zero_one,one_given_one_zero/one_zero,one_given_one_one/one_one])

             

Hematocrit = BbnNode(Variable(0, 'Hematocrit', ['0.0', '1.0']), prob_calc(dataset_2,child='Hematocrit'))
Hemoglobin = BbnNode(Variable(1, 'Hemoglobin', ['0.0', '1.0']), prob_calc(dataset_2,child='Hemoglobin',parent1='Hematocrit'))
Lymphocytes = BbnNode(Variable(2, 'Lymphocytes', ['0.0', '1.0']), prob_calc(dataset_2,child='Lymphocytes'))
CovidResult = BbnNode(Variable(3, 'CovidResult', ['0.0', '1.0']), prob_calc(dataset_2,child='SARS-Cov-2 exam result',parent1='Hemoglobin',parent2='Lymphocytes'))


# In[6]:


bbn = Bbn()     .add_node(Hematocrit)     .add_node(Hemoglobin)     .add_node(Lymphocytes)     .add_node(CovidResult)     .add_edge(Edge(Hematocrit, Hemoglobin, EdgeType.DIRECTED))     .add_edge(Edge(Hemoglobin, CovidResult, EdgeType.DIRECTED))     .add_edge(Edge(Lymphocytes, CovidResult, EdgeType.DIRECTED))

join_tree = InferenceController.apply(bbn)
pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "white",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 5,
    "width": 5, }

n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, **options)

ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()

def print_probs():
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')
print(join_tree.get_bbn_nodes)
print_probs()


# In[ ]:





# In[ ]:




