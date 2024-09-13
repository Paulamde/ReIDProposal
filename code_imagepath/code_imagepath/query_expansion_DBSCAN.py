# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:03:13 2019

@author: Deng
"""
from sklearn.cluster import DBSCAN
import numpy as np
import torch
import torch.nn as nn
import math
import os
import glob
import re
import xml.dom.minidom as XD
import os.path as osp
from PIL import Image

query_dir = 'D:/feature_learning-master/Feature_Learning/learning_model/data/VR/image_query/'
img_paths = glob.glob(osp.join(query_dir, '*.jpg'))
img_paths.sort()
pid_container = set()
for img_path in img_paths:
    pid = 1
    if pid == -1: continue  # junk images are just ignored
    pid_container.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_container)}

dataset = []
for img_path in img_paths:
    pid, camid = 1, 2
    if pid == -1: continue  # junk images are just ignored
    camid -= 1  # index starts from 0
    if False: pid = pid2label[pid]
    dataset.append((img_path))

# --------------------------- query expansion -----------------#
# load feature
gf = np.load(
    'D:/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_concatAicitySyntestaic_e.npy')
qf = np.load(
    'D:/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_concatAicitySyntestaic.npy')



query_feature=torch.from_numpy(qf)

q_q_dist = torch.mm(query_feature, torch.transpose(query_feature, 0, 1))
q_q_dist = q_q_dist.cpu().numpy()
q_q_dist[q_q_dist>1] = 1  #due to the epsilon
q_q_dist = 2-2*q_q_dist


eps = 0.5
min_samples =2
cluster1 = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', algorithm='auto', n_jobs=-1)
cluster1 = cluster1.fit(q_q_dist)
qlabels = cluster1.labels_
nlabel_q = len(np.unique(cluster1.labels_))

# Query Feature fusion
query_feature_clone = query_feature.clone()

#query_path = image_datasets['query'].imgs
query_path =dataset
query_shape = np.zeros(len(query_path))
count = 0
for name in query_path:
    img = np.asarray(Image.open(name))
    query_shape[count] = img.shape[0] * img.shape[1]
    count += 1
junk_index_q = np.argwhere(query_shape< 15000).flatten() # se quiere eliminar aquellas queries con menos resolucion de 150x150

for i in range(nlabel_q-1):
    index = np.argwhere(qlabels==i).flatten()  #from small to large, start from 0
    high_quality_index=index
    high_quality_index = np.setdiff1d(index, junk_index_q)
    if len(high_quality_index) == 0:
        high_quality_index = index
    qf_mean = torch.mean(query_feature_clone[high_quality_index,:], dim=0)
    for j in range(len(index)):
        query_feature[index[j],:] = qf_mean

fnorm = torch.norm(query_feature, p=2, dim=1, keepdim=True) + 1e-5
query_feature = query_feature.div(fnorm.expand_as(query_feature))
qf=query_feature.numpy()
np.save(
        'D:/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_concatAicitySyntestaic_e.npy',qf)