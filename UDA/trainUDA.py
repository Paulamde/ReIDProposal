# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch.nn as nn
import scipy.io
from torch.backends import cudnn
from torch.nn import Parameter

sys.path.append('../')
from config import cfg
from data import make_data_loader, get_trainloader_uda, get_testloader_uda, make_train_loader
# from engine.trainerPLTRAIN import do_train
from engine.trainer import do_train
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR
import torch.nn.functional as F
from timm.data import Mixup
from sklearn.cluster import DBSCAN, KMeans
from utils.faiss_rerank import compute_jaccard_distance
from utils.faiss_rerank import batch_cosine_dist, cosine_dist
from utils.metrics import euclidean_distance
from utils.logger import setup_logger

import numpy as np
import torch
import random
import h5py


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        name = name.replace('module.', '')
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


def extract_features(model, data_loader, print_freq):
    model.eval()
    feats = []
    vids = []
    camids = []
    trkids = []
    with torch.no_grad():
        for i, (img, vid, camid, trkid, _) in enumerate(data_loader):
            img = img.to('cuda')
            feat = model(img)

            feats.append(feat)
            vids.extend(vid)
            camids.extend(camid)
            trkids.extend(trkid)

    feats = torch.cat(feats, dim=0)
    vids = torch.tensor(vids).cpu().numpy()
    camids = torch.tensor(camids).cpu().numpy()
    trkids = torch.tensor(trkids).cpu().numpy()

    return feats, vids, camids, trkids


def calc_distmat(feat):
    rerank_distmat = compute_jaccard_distance(feat, k1=30, k2=6, search_option=3)
    cosine_distmat = batch_cosine_dist(feat, feat).cpu().numpy()
    final_dist = rerank_distmat * 0.9 + cosine_distmat * 0.1
    # with torch.no_grad():
    #     rerank_distmat = torch.from_numpy(compute_jaccard_distance(feat, k1=30, k2=6, search_option=3))
    #     cosine_distmat = batch_cosine_dist(feat, feat)
    #     final_dist = rerank_distmat * 0.9 + cosine_distmat * 0.1
    #     del cosine_distmat,rerank_distmat

    return final_dist


def compute_P2(qf, gf, gc, la=3.0):
    X = gf
    neg_vec = {}
    u_cams = np.unique(gc)
    P = {}
    for cam in u_cams:
        curX = gf[gc == cam]
        neg_vec[cam] = torch.mean(curX, axis=0)
        tmp_eye = torch.eye(X.shape[1]).cuda()
        P[cam] = torch.inverse(curX.T.matmul(curX) + curX.shape[0] * la * tmp_eye)
    return P, neg_vec


def meanfeat_sub(P, neg_vec, in_feats, in_cams):
    out_feats = []
    for i in range(in_feats.shape[0]):
        camid = in_cams[i]
        feat = in_feats[i] - neg_vec[camid]
        feat = P[camid].matmul(feat)
        feat = feat / torch.norm(feat, p=2)
        out_feats.append(feat)
    out_feats = torch.stack(out_feats)
    return out_feats


def train(cfg):
    # prepare dataset
    val_loader, num_query, testset = get_testloader_uda(cfg)
    train_loader_train, val_loader_train, num_query_train, num_classes_train = make_train_loader(cfg)

    gallery_gt = scipy.io.loadmat('/home/pme/Desktop/rhome/pme/SubsetEvaluation/gallery_annotation2021.mat')
    gallery_gt_f = [gallery_gt['gallery_new'][i] for i in range(gallery_gt['gallery_new'].shape[0])]

    query_gt = scipy.io.loadmat('/home/pme/Desktop/rhome/pme/SubsetEvaluation/query_annotation2021.mat')
    query_gt_f = [query_gt['query_new'][i] for i in range(query_gt['query_new'].shape[0])]
    num_classes = 1802
    # prepare model
    model = build_model(cfg, num_classes)
    # cfg2=cfg.clone()
    # cfg2.defrost()
    # cfg2.INPUT.COLORJITTER="True"
    # cfg2.MODEL.PRETRAIN_PATH='/home/pme/Desktop/rhome/pme/feature_learning-master/Feature_Learning/learning_model/tools/CHECKPOINTS/SynNewAic21Baselinemodel2/densenet121_model_80.pth'
    # cfg2.freeze()
    # cfg3 = cfg.clone()
    # cfg3.defrost()
    # cfg3.DATALOADER.SOFT_MARGIN = "True"
    # cfg3.MODEL.PRETRAIN_PATH='/home/pme/Desktop/rhome/pme/feature_learning-master/Feature_Learning/learning_model/tools/CHECKPOINTS/SynNewAic21Baselinemodel3/densenet121_model_80.pth'
    # cfg3.freeze()
    # model2 = build_model(cfg2, num_classes)
    # model3 = build_model(cfg3, num_classes)
    initial_weights = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cpu')
    # initial_weights2 = torch.load(cfg2.MODEL.PRETRAIN_PATH, map_location='cpu')
    # initial_weights3 = torch.load(cfg3.MODEL.PRETRAIN_PATH, map_location='cpu')

    copy_state_dict(initial_weights, model)
    # copy_state_dict(initial_weights2, model2)
    # copy_state_dict(initial_weights3, model3)
    model.to("cuda")
    # model2.to("cuda")
    # model3.to("cuda")
    # model_ema.classifier.weight.data.copy_(model.classifier.weight.data)
    # if True:
    #     model.to(0)
    #     if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
    #         print('Using {} GPUs for training'.format(torch.cuda.device_count()))
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    #a = np.linspace(0.55, 1, 20)
    #b = a[0:10]
    #c = a[10:-1]
    # D=[0.05,0.1,0.2,0.5,0.55,0.65,0.7,0.85,0.9]
    #D = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    #D = [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]
    D= np.repeat(np.arange(0.55, 1, 0.05), 20)

    # a = np.linspace(2, 20, 19)
    # b = a[0:10]
    # c = a[10:-1]
    contador = 0
    for DBSCAN_step in D:

        for epoch in range(32):

            if epoch % 32 == 0:  # and epoch < 9) or (epoch % 6 == 0):
                target_features, target_labels, target_camids, target_trkids = extract_features(model, val_loader,
                                                                                                print_freq=100)
                target_features_train, target_labels_train, target_camids_train, target_trkids_train = extract_features(
                    model, train_loader_train,
                    print_freq=100)
                # target_features2, target_labels2, target_camids2, target_trkids2 = extract_features(model2, val_loader,
                #                                                                                 print_freq=100)
                # target_features3, target_labels3, target_camids3, target_trkids3 = extract_features(model3, val_loader,
                #                                                                                 print_freq=100)

                target_features = F.normalize(target_features, dim=1)
                P, neg_vec = compute_P2(target_features, target_features, target_camids, la=0.0005)
                target_features = meanfeat_sub(P, neg_vec, target_features, target_camids)
                target_features_train = F.normalize(target_features_train, dim=1)
                P_train, neg_vec_train = compute_P2(target_features_train, target_features_train, target_camids_train,
                                                    la=0.0005)
                target_features_train = meanfeat_sub(P_train, neg_vec_train, target_features_train, target_camids_train)
                # target_features2 = F.normalize(target_features2, dim=1)
                # P2, neg_vec2 = compute_P2(target_features2, target_features2, target_camids2, la=0.0005)
                # target_features2 = meanfeat_sub(P2, neg_vec2, target_features2, target_camids2)
                # target_features3 = F.normalize(target_features3, dim=1)
                # P3, neg_vec3 = compute_P2(target_features3, target_features3, target_camids3, la=0.0005)
                # target_features3 = meanfeat_sub(P3, neg_vec3, target_features3, target_camids3)

                # torch.save(target_features, 'target_features.pth')
                # target_features = torch.load('target_features.pth')

                gallery_trkids = target_trkids[num_query:]
                unique_trkids = sorted(list(set(gallery_trkids[gallery_trkids != -1])))
                gallery_features = target_features[num_query:]
                track_feature = []
                # gallery_trkids2 = target_trkids2[num_query:]
                # unique_trkids2 = sorted(list(set(gallery_trkids2[gallery_trkids2 != -1])))
                # gallery_features2 = target_features2[num_query:]
                # track_feature2 = []
                # gallery_trkids3 = target_trkids3[num_query:]
                # unique_trkids3 = sorted(list(set(gallery_trkids3[gallery_trkids3 != -1])))
                # gallery_features3 = target_features3[num_query:]
                # track_feature3 = []

                unique_trkids_train = sorted(list(set(target_trkids_train[target_trkids_train != -1])))
                track_feature_train = []

                for i, trkid in enumerate(unique_trkids_train):
                    track_feature_train = torch.mean(target_features_train[target_trkids_train == trkid], dim=0,
                                                     keepdim=True)
                    tmp_indices = (target_trkids_train == trkid)
                    target_features_train[tmp_indices] = target_features_train[
                                                             tmp_indices] * 0.3 + track_feature_train * 0.7

                for i, trkid in enumerate(unique_trkids):
                    track_feature = torch.mean(gallery_features[gallery_trkids == trkid], dim=0, keepdim=True)
                    tmp_indices = (gallery_trkids == trkid)
                    gallery_features[tmp_indices] = gallery_features[tmp_indices] * 0.3 + track_feature * 0.7
                target_features[num_query:] = gallery_features
                # for i, trkid in enumerate(unique_trkids2):
                #     track_feature2 = torch.mean(gallery_features2[gallery_trkids2 == trkid], dim=0, keepdim=True)
                #     tmp_indices = (gallery_trkids2 == trkid)
                #     gallery_features2[tmp_indices] = gallery_features2[tmp_indices] * 0.3 + track_feature2 * 0.7
                # target_features2[num_query:] = gallery_features2
                # for i, trkid in enumerate(unique_trkids3):
                #     track_feature3 = torch.mean(gallery_features3[gallery_trkids3 == trkid], dim=0, keepdim=True)
                #     tmp_indices = (gallery_trkids3 == trkid)
                #     gallery_features3[tmp_indices] = gallery_features3[tmp_indices] * 0.3 + track_feature3 * 0.7
                # target_features3[num_query:] = gallery_features3

                # final_dist = calc_distmat(target_features)
                # target_featuresConcat=torch.cat((target_features,target_features2,target_features3))
                #
                # target_features_n = np.linalg.norm(target_features.cpu().numpy(), axis=1, keepdims=True)
                # target_featuresnorm = target_features.cpu().numpy() / target_features_n
                # target_features2_n = np.linalg.norm(target_features2.cpu().numpy(), axis=1, keepdims=True)
                # target_features2 = target_features2.cpu().numpy() / target_features2_n
                # target_features3_n = np.linalg.norm(target_features3.cpu().numpy(), axis=1, keepdims=True)
                # target_features3= target_features3.cpu().numpy() / target_features3_n
                # target_featuresConcat = np.concatenate((target_featuresnorm, target_features2, target_features3), axis=1)/np.sqrt(3)
                # target_featuresConcat = torch.from_numpy(target_featuresConcat)
                target_featuresConcat = target_features

                final_dist = calc_distmat(target_featuresConcat)
                final_dist[final_dist < 0.0] = 0.0
                final_dist[final_dist > 1.0] = 1.0
                # final_dist2 = calc_distmat(target_features)
                # final_dist2[final_dist2 < 0.0] = 0.0
                # final_dist2[final_dist2 > 1.0] = 1.0
                # final_dist3 = calc_distmat(target_features3)
                # final_dist3[final_dist3 < 0.0] = 0.0
                # final_dist3[final_dist3 > 1.0] = 1.0
                mat = np.zeros((563, 100))

                print('bieeen')
                cluster = DBSCAN(eps=DBSCAN_step, min_samples=10, metric='precomputed', n_jobs=-1)
                pseudo_labels = cluster.fit_predict(final_dist)
                labelset = list(set(pseudo_labels[pseudo_labels >= 0]))

                # idxs_old = np.where(np.in1d(pseudo_labels, labelset))
                # idxs = [i for i in idxs_old[0] if i + 17260 in idxs_old[0] and i + 34520 in idxs_old[0]]
                idxs = np.where(np.in1d(pseudo_labels, labelset))
                psolabels = pseudo_labels[idxs]
                ### QUITANDO PICO ####
                # auxiliae, auxiliar2 = np.unique(psolabels, return_counts='True')#import matplotlib.pyplot as plt plt.plot(auxiliar2)
                # st_dev=np.std(auxiliar2)
                # maxindex = auxiliae[auxiliar2>4*st_dev]#maxindex = auxiliae[auxiliar2>3*st_dev]
                # labelsetAux = [i for i in labelset if i not in maxindex]
                # idxs = np.where(np.in1d(pseudo_labels, labelsetAux))
                # labelset=labelset[0:-maxindex.size]
                # for j in range(pseudo_labels.size):
                #     if pseudo_labels[j] in maxindex:
                #         pseudo_labels[j] = -1
                #     elif pseudo_labels[j] > maxindex[0]:
                #         NumAdded_sortLabelset = [i for i, x in enumerate(maxindex) if x < pseudo_labels[j]]
                #         pseudo_labels[j] = pseudo_labels[j] - NumAdded_sortLabelset.__len__()
                # psolabels = pseudo_labels[idxs]
                ### QUITANDO PICO ####
                ### con GT ####
                # labelset = list(set(target_labels[target_labels >= 0]))
                # #
                # indices = dict()
                # for i in range(len(labelset)):
                #     indices[labelset[i]] = i
                #
                # for i in range(len(target_labels)):
                #     target_labels[i] = indices[target_labels[i]]
                # labelset = range(len(labelset))
                # idxs = np.where(np.in1d(target_labels, labelset))
                # psolabels = target_labels[idxs]
                ### con GT ####
                flag = 0
                if len(labelset) == 0:
                    flag = 1
                    break
                file = open("./Epoch_{}_Th_{}.txt".format(epoch, DBSCAN_step), "w+")
                write_line = 'number of IDs: ' + str(labelset.__len__()) + ' Number of images: ' + str(
                    idxs[0].size) + '\n'
                file.write(write_line)
                file.close()
                # OtroDBSCAN
                # cluster = DBSCAN(eps=0.55, min_samples=10, metric='precomputed', n_jobs=-1)
                # pseudo_labels2 = cluster.fit_predict(final_dist2)
                # labelset2= list(set(pseudo_labels2[pseudo_labels2 >= 0]))
                # idxs2 = np.where(np.in1d(pseudo_labels2, labelset2))
                # psolabels2 = pseudo_labels2[idxs2]
                # cluster = DBSCAN(eps=0.55, min_samples=10, metric='precomputed', n_jobs=-1)
                # pseudo_labels3 = cluster.fit_predict(final_dist3)
                # labelset3 = list(set(pseudo_labels3[pseudo_labels3 >= 0]))
                # idxs3 = np.where(np.in1d(pseudo_labels3, labelset3))
                # psolabels3 = pseudo_labels3[idxs3]

                # psofeatures = target_features[idxs]
                matriz_aux = final_dist[idxs[0], :]
                matriz_aux = matriz_aux[:, idxs[0]]
                a = np.argsort(final_dist, axis=1)
                for i in range(563):
                    cont = 0
                    for j in range(17260):
                        if cont == 100:
                            break
                        if (a[i][j] > 562):
                            mat[i, cont] = int(a[i][j])
                            cont = cont + 1
                        # if (a[i][j] > 562 and a[i][j] in idxs[0]):
                        #     mat[i, cont] = int(a[i][j])
                        #     cont = cont + 1
                m = 563

                with open('./track2_SDBSCAN_{}_{}.txt'.format(DBSCAN_step, contador), 'wb') as f_w:
                    for i in range(m):
                        write_line = mat[i] - 562
                        write_line = ' '.join(map(str, write_line.tolist())) + '\n'
                        f_w.write(write_line.encode())
                f_w.close()
                # file = open("indices.txt", "w+")
                #
                # # Saving the array in a text file
                # file = open("indices.txt", "w+")
                # indices = idxs[0][0:287]
                # content = str(indices)
                # file.write(content)
                # file.close()

                psofeatures = target_featuresConcat[idxs]
                mean_features = []
                for label in labelset:
                    mean_indices = (psolabels == label)
                    mean_features.append(torch.mean(psofeatures[mean_indices], dim=0))

                labelset_train = np.unique(target_labels_train)
                for label in labelset_train:
                    mean_indices = (target_labels_train == label)
                    mean_features.append(torch.mean(target_features_train[mean_indices], dim=0))
                # mean_features.append(torch.mean(target_features_train, dim=0))
                mean_features = torch.stack(mean_features).cuda()

                num_classes = len(mean_features)
                numclasespl = len(labelset)
                numclasestrain = len(labelset_train)

                model.num_classes = len(mean_features)
                model.classifier = nn.Linear(model.in_planes, len(mean_features), bias=False)
                model.classifier.weight = nn.Parameter(mean_features)
                classifier_gt = nn.Linear(model.in_planes, numclasestrain, bias=False)
                classifier_gt.weight = nn.Parameter(mean_features[numclasespl:])
                classifier_pl = nn.Linear(model.in_planes, numclasespl, bias=False)
                classifier_pl.weight = nn.Parameter(mean_features[0:numclasespl])
                # model.classifier = nn.Linear(model.in_planes, len(mean_features), bias=False)
                # model.classifier = nn.Linear(mean_features.shape[1], len(mean_features), bias=False)
                # model.classifier.weight = nn.Parameter(mean_features[:,0:1024])

                del target_featuresConcat, target_features  # ,target_features2,target_features3

                pids = []
                new_dataset = []
                new_dataset_pseudo = []
                new_dataset_train = []
                max_track = 0
                for i, (item, label) in enumerate(zip(testset, target_labels)):
                    # for i, (item, label) in enumerate(zip(testset, pseudo_labels)):
                    if label == -1 or label not in labelset:
                        continue
                    pids.append(label)
                    # new_dataset.append((item[0], label, item[2], item[3]))
                    if max_track < item[1]:
                        max_track = item[1]
                    new_dataset_pseudo.append((item[0], label, item[2], item[1]))
                for item in (train_loader_train.dataset.dataset):
                    # new_dataset_train.append((item[0], item[1]+pseudo_labels.max(), item[2], item[3]+max_track))#impath,pid,camID,trackid
                    new_dataset_train.append((item[0], item[1], item[2],
                                              item[3]))  # impath,pid,camID,trackid

                print('new class are {}, length of new dataset is {}'.format(len(set(pids)), len(new_dataset)))

            train_loader_pl = IterLoader(get_trainloader_uda(cfg, new_dataset_pseudo, numclasespl))
            train_loader_pl.new_epoch()
            train_loader_gt = IterLoader(get_trainloader_uda(cfg, new_dataset_train, numclasestrain))
            train_loader_gt.new_epoch()
            optimizer = make_optimizer(cfg, model, classificators=[classifier_gt, classifier_pl])
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

            ## multi-gpu
            # model = nn.DataParallel(model, device_ids=[0,1])

            loss_func = make_loss(cfg)

            arguments = {}

            do_train(
                epoch,
                cfg,
                model,
                train_loader_pl,
                # train_loader_gt,
                val_loader,
                optimizer,
                scheduler,
                loss_func,
                num_query,
                DBSCAN_step,
                contador,
                # classifier_gt,
                # classifier_pl,

            )
        if flag == 1:
            flag = 0
            continue
        contador = contador + 1


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    set_seed(1234)
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
