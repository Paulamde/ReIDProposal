import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('../')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
from data.datasets.eval_reid import eval_func
import logging
import numpy as np

from utils.re_ranking import re_ranking


def main():


    # load features

    #gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_1.npy')
    #qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_1.npy')

    #gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_2.npy')
    #qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_2.npy')

    #gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_3.npy')
    #qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_3.npy')

    #gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_multi.npy')
    #qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_multi_0.npy')

    #gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_1e.npy')
    #qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_1e_0.npy')

    #gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_1_ep120.npy')
    #qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_1_ep120.npy')


    #gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_e_1_meta.npy')
    #qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_e_1_meta.npy')


    ##gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_props6_ALL.npy')
    ##qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_props6_ALL.npy')

    ##gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_props3_ALL.npy')
    ##qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_props3_ALL.npy')

    ##gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_metadatos6_ALL.npy')
    ##qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_metadatos6_ALL.npy')

    ##gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_metadatos3_ALL.npy')
    ##qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_metadatos3_ALL.npy')


    ##gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_appstr2.npy')
    ##qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_appstr2.npy')

    ##gf_n = np.linalg.norm(gf, axis=1, keepdims=True)
    ##gf = gf / gf_n
    ##qf_n = np.linalg.norm(qf, axis=1, keepdims=True)
    ##qf = qf / qf_n



    ##gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_props3_VPU.npy')
    ##qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_props3_VPU.npy')


    """gf_n = np.linalg.norm(gf, axis=1, keepdims=True)
    gf = gf / gf_n
    qf_n = np.linalg.norm(qf, axis=1, keepdims=True)
    qf = qf / qf_n
    """

    """gf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_trainSY_testVeRi_1.npy')
    qf = np.load('/home/agm2/Downloads/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_metadatos3_VPU.npy')
    gf_n = np.linalg.norm(gf, axis=1, keepdims=True)
    gf = gf / gf_n
    qf_n = np.linalg.norm(qf, axis=1, keepdims=True)
    qf = qf / qf_n
    """

    gf = np.load('D:/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/gf_concatAicitySyntestaic_e.npy')
    qf = np.load('D:/feature_learning-master/Feature_Learning/learning_model/data/feature_expansion/qf_concatAicitySyntestaic_e.npy')


    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))

    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    indices = np.argsort(re_rank_dist, axis=1)[:, :100]
    m, n = indices.shape
    print('m: {}  n: {}'.format(m, n))
    with open('track2_AicSyntestAic2.txt', 'wb') as f_w:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f_w.write(write_line.encode())
    print(indices[0])
    print(indices.shape)

if __name__ == '__main__':
    main()
