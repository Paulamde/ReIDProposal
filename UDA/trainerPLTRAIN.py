import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda import amp
import torch.distributed as dist
import logging
import os
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from tqdm import tqdm

def do_train(epoch, cfg, model, train_loader_pl,train_loader_gt, val_loader, optimizer, scheduler, loss_fn,num_query,DBSCAN_step, contador, classifier_gt=None, classifier_pl=None):#, tensorboard=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = 30
    eval_period = cfg.SOLVER.EVAL_PERIOD
    print('lr: %f' % optimizer.param_groups[-1]['lr'])

    scaler = amp.GradScaler()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    device = 'cuda'
    logger = logging.getLogger('reid_baseline.train')


    model.train()
    # model_ema.train()
    ##
    # rober
    ##lambda_ent = 100 #Modificar
    ##iter_val = iter(val_loader)
    #for n_iter in tqdm(range(300)):
    for n_iter in range(300):
        if n_iter %2==0:
            img, vid, target_cam = train_loader_pl.next()
        else:
            img, vid, target_cam = train_loader_gt.next()
            #vid=vid-classifier_pl.out_features


        optimizer.zero_grad()
        img = img.to(device)
        target = vid.to(device)
        #target_cam = target_cam.to(device)

        if True:
        #     with amp.autocast(enabled=True):
        #         #score, feat = model(img, target, cam_label=target_cam)
        #         score, feat = model(img)
        #         #loss = loss_fn(score, feat, target, target_cam)
        #         loss = loss_fn(score, feat, target)
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
            # score, feat = model(img, target, cam_label=target_cam)
            score, feat = model(img)
            loss_concat = score.sum()




            # loss = loss_fn(score, feat, target, target_cam)
            #loss = loss_fn(score, feat, target)
            if n_iter %2 == 0:
                score = classifier_pl(feat)
                # score = score.to("cpu")
                # feat = feat.to("cpu")
                # target = target.to("cpu")
                #print(target)
                loss_ft = loss_fn(score, feat, target)
                loss_gt = score.sum()
                loss = 0*loss_concat + 0*loss_gt + loss_ft
            #     if tensorboard:
            #         tensorboard.add_scalar("Loss PL", loss.detach().cpu().item(), n_iter+epoch*300)
            #         tensorboard.add_scalar("Conf PL", score.detach().cpu().item(), n_iter + epoch * 300)
            #         aux = feat.clone().detach()
            #         entropy = torch.functional.softmax(aux,dim=1)
            #         entropy = (entropy*toch.log(entropy)).sum(1)
            #         tensorboard.add_scalar("Entropy PL", entropy.mean().detach().cpu().item(), n_iter + epoch * 300)
            else:
                score = classifier_gt(feat)
                # score = score.to("cpu")
                # feat = feat.to("cpu")
                # target = target.to("cpu")
                loss_gt = loss_fn(score, feat, target)
                loss_ft = score.sum()
                loss = 0 * loss_concat + 1*loss_gt + 0*loss_ft
                # if tensorboard:
                #     tensorboard.add_scalar("Loss Train", loss.detach().cpu().item(), n_iter + epoch * 300)
                #     tensorboard.add_scalar("Conf Train", score.detach().cpu().item(), n_iter + epoch * 300)
                #     aux = feat.clone().detach()
                #     entropy = torch.functional.softmax(aux, dim=1)
                #     entropy = (entropy * toch.log(entropy)).sum(1)
                #     tensorboard.add_scalar("Entropy PL", entropy.mean().detach().cpu().item(), n_iter + epoch * 300)
            loss.backward()
            ##img_val = iter_val.next()[0]
            ##img_val = img_val.to(device)
            ##score_tgt, feat = model(img_val)
            ##p = torch.softmax(score_tgt, 1)
            ##loss_ent = lambda_ent * torch.mean(-torch.mul(p, torch.log(p)))
            ##loses = {"Loss": loss.detach().cpu().numpy(),"Loss_entropy": loss_ent.detach().cpu().numpy()}
            ##tqdm.write(str(loses))
            ##loss_ent.backward()
            optimizer.step()
            scheduler.step()

        def update_ema_variables(model, ema_model, global_step):
            alpha = 0.999
            alpha = min(1 - 1 / (global_step + 1), alpha)
            for (ema_name, ema_param), (model_name, model_param) in zip(ema_model.named_parameters(),
                                                                        model.named_parameters()):
                ema_param.data = ema_param.data * alpha + model_param.data * (1 - alpha)

        # update_ema_variables(model, model_ema, epoch*len(train_loader) + n_iter)

        if isinstance(score, list):
            acc = (score[0].max(1)[1] == target).float().mean()
        else:
            acc = (score.max(1)[1] == target).float().mean()
        loss_meter.update(loss.item(), img.shape[0])
        acc_meter.update(acc, 1)

        torch.cuda.synchronize()

        if (n_iter + 1) % log_period == 0:
            logger.info(
                'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}'.format(epoch, (n_iter + 1), 300, loss_meter.avg,
                                                                              acc_meter.avg))

    if (epoch + 1) % checkpoint_period == 0:
        torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}_{}_{}.pth'.format(epoch,DBSCAN_step,contador)))

    if (epoch + 1) % eval_period == 0:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                                dataset=cfg.DATASETS.NAMES)  # only available for offline validation
        evaluator.reset()
        if cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                model_ema.eval()
                torch.cuda.empty_cache()
                for n_iter, (img, vid, camid, trackids, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = torch.tensor(camid, dtype=torch.int64)
                        camids = camids.to(device)
                        feat = model_ema(img, cam_label=camids)
                        evaluator.update((feat.clone(), vid, camid, trackids))

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info(cfg.OUTPUT_DIR)
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
        else:
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for n_iter, (img, vid, camid, trackids, _) in enumerate(val_loader):
                    img = img.to(device)
                    camids = torch.tensor(camid, dtype=torch.int64)
                    camids = camids.to(device)
                    feat = model(img, cam_label=camids)
                    evaluator.update((feat.clone(), vid, camid, trackids))

            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            # if tensorboard:
            #     tensorboard.add_scalar("Validation CMC", cmc, epoch) #Si da error es que le estas pasando un tensor CPU() item().
            #     tensorboard.add_scalar("Validation mAP", mAP, epoch)
            logger.info(cfg.OUTPUT_DIR)
            logger.info("Validation Results Standard Model - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

