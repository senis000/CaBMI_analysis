#!/usr/bin/env python
# coding: utf-8

# In[107]:


import sys
import os
import pickle
import pdb
import re
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# In[16]:


datadir = "/run/user/1000/gvfs/smb-share:server=typhos.local,share=data_01/NL/layerproject/processed/"
pattern = 'full_(IT|PT)(\d+)_(\d+)_.*\.hdf5'


# # Plotting the distribution of labeled neurons (z-plane only)

# In[53]:


#all_IT_zlocations = []
#all_PT_zlocations = []
#all_nonred_zlocations = []
#
#
## In[54]:
#
#
#for animaldir in os.listdir(datadir):
#    animal_path = datadir + animaldir + '/'
#    if not os.path.isdir(animal_path):
#        continue
#    for file_name in os.listdir(animal_path):
#        result = re.search(pattern, file_name)
#        if not result:
#            continue
#        experiment_type = result.group(1)
#        experiment_animal = result.group(2)
#        experiment_date = result.group(3)
#        f = h5py.File(animal_path + file_name, 'r')
#        redlabel = np.array(f['redlabel'])
#        com_cm = np.array(f['com_cm'])
#        red_zlocation = com_cm[redlabel, 2]
#        nonred_zlocation = com_cm[np.logical_not(redlabel), 2]
#        if experiment_type == "IT":
#            all_IT_zlocations.extend(red_zlocation)
#        elif experiment_type == "PT":
#            all_PT_zlocations.extend(red_zlocation)
#        all_nonred_zlocations.extend(nonred_zlocation)
#

# In[58]:


#all_IT_zlocations = np.array(all_IT_zlocations)
#all_PT_zlocations = np.array(all_PT_zlocations)
#all_nonred_zlocations = np.array(all_nonred_zlocations)
#
#
## In[88]:
#
#
#all_zlocations = np.hstack((
#    all_IT_zlocations, all_PT_zlocations, all_nonred_zlocations
#    ))*-1
#all_labels = np.hstack((
#    ['IT']*len(all_IT_zlocations),
#    ['PT']*len(all_PT_zlocations),
#    ['Unlabeled']*len(all_nonred_zlocations)
#    ))
#
#
## In[89]:
#
#
#pd_data = pd.DataFrame({
#    'Microns Below Cortical Surface': all_zlocations,
#    'Neuron Identity': all_labels
#    })


# In[91]:


#plt.figure(figsize=(10,10))
#sns.catplot(
#    x='Neuron Identity', y='Microns Below Cortical Surface',
#    data=pd_data, kind="boxen"
#    )
#plt.title("Depth Distribution of Neurons")
#plt.show(block=True)


# # Plotting the distribution of labeled neurons (xyz plane)

# In[151]:


all_IT_locations = []
all_PT_locations = []
all_nonred_locations = []


# In[152]:


for animaldir in os.listdir(datadir):
    animal_path = datadir + animaldir + '/'
    if not os.path.isdir(animal_path):
        continue
    for file_name in os.listdir(animal_path):
        result = re.search(pattern, file_name)
        if not result:
            continue
        experiment_type = result.group(1)
        experiment_animal = result.group(2)
        experiment_date = result.group(3)
        f = h5py.File(animal_path + file_name, 'r')
        redlabel = np.array(f['redlabel'])
        com_cm = np.array(f['com_cm'])
        red_location = com_cm[redlabel,:]
        nonred_location = com_cm[np.logical_not(redlabel),:]
        red_location[:,2]*=-1
        nonred_location[:,2]*=-1
        if experiment_type == "IT":
            all_IT_locations.extend(red_location)
        elif experiment_type == "PT":
            all_PT_locations.extend(red_location)
        all_nonred_locations.extend(nonred_location)


# In[153]:


all_IT_locations = np.array(all_IT_locations)
all_PT_locations = np.array(all_PT_locations)
all_nonred_locations = np.array(all_nonred_locations)


# In[154]:


all_locations = np.vstack((
    all_IT_locations, all_PT_locations, all_nonred_locations
    ))
all_labels = np.hstack((
    ['IT']*all_IT_locations.shape[0],
    ['PT']*all_PT_locations.shape[0],
    ['Unlabeled']*all_nonred_locations.shape[0]
    ))


# In[155]:


pd_data = pd.DataFrame({
    'X Axis Location (Microns)': all_locations[:,0],
    'Y Axis Location (Microns)': all_locations[:,1],
    'Depth Below Cortical Surface (Microns)': all_locations[:,2],
    'Neuron Identity': all_labels
    })


# In[167]:


plt.figure()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(
    all_IT_locations[:,0],
    all_IT_locations[:,1],
    all_IT_locations[:,2],
    linestyle="", marker="o",
    color='orange', markersize=1, alpha=1
    )
ax.plot(
    all_PT_locations[:,0],
    all_PT_locations[:,1],
    all_PT_locations[:,2],
    linestyle="", marker="o",
    color='red', markersize=1, alpha=1
    )
ax.plot(
    all_nonred_locations[:,0],
    all_nonred_locations[:,1],
    all_nonred_locations[:,2],
    linestyle="", marker="o",
    color='blue', markersize=0.03, alpha=.6
    )
ax.set_xlabel('\nX Axis Location (Microns)')
ax.set_ylabel('\nY Axis Location (Microns)')
ax.set_zlabel('\nDepth Below Cortical Surface (Microns)')
plt.title("Spatial Distribution of Neurons")
print(
ax.view_init(azim=-30)
plt.show(block=True)

# In[ ]:




