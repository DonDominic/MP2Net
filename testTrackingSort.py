from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from re import L
from lib.utils.opts import opts
opt = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
import numpy as np
import cv2
import time
import torch
import torch.nn.functional as F
from collections import defaultdict

from lib.models.stNet import get_det_net, load_model
from lib.dataset.coco_icpr import COCO

from lib.external.nms import soft_nms

from lib.utils.decode import ctdet_decode
from lib.utils.post_process import generic_post_process

from lib.utils.sort import *

from progress.bar import Bar

def show_mask(mask):
    import matplotlib.pyplot as plt

    heatmap = mask.squeeze().detach().cpu().numpy()                  # H, W
    heatmap /= np.max(heatmap)                                  # minmax norm
    heatmap = np.uint8(255*heatmap)
    return heatmap

def show_arrow(det, dis, readname, writename):
    img = cv2.imread(readname)
    N = det.shape[0]
    for i in range(N):
        ct1 = (det[i][0] + det[i][2]) / 2
        ct2 = (det[i][1] + det[i][3]) / 2
        arrow = dis[i][-1] * 10
        img = cv2.arrowedLine(img,
                    pt1=(int(ct1), int(ct2)),
                    pt2=(int(ct1 + arrow[0]), int(ct2 + arrow[1])),
                    color=(255, 255, 0),
                    thickness=2, tipLength=0.5)
    cv2.imwrite(writename, img)
    return

def process(model, ratios, image, vid=None):
    with torch.no_grad():
        dets_all = {}
        output_all = model(image, training=False, vid=vid)[-1]
        for ratio in ratios:
            output = output_all[ratio]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            dis = output['dis']
            torch.cuda.synchronize()
            dets = ctdet_decode(hm, wh, reg=reg, tracking=dis, num_classes=opt.num_classes, K=opt.K)
            for k in dets:
                dets[k] = dets[k].detach().cpu().numpy()
            dets_all[ratio] = dets
            # mask_seq = show_mask(F.sigmoid(output['hm_seq'][0, -1]))
    return dets_all

def post_process(ratios, dets_all, meta, scale=1):
    dets_all_post = {}
    for ratio in ratios:
        dets = generic_post_process(
            dets_all[ratio], [meta['c']], [meta['s']], meta['out_height'] // ratio // opt.down_ratio, meta['out_width'] // ratio // opt.down_ratio)
        if scale != 1:
            for i in range(len(dets[0])):
                for k in ['bbox']:
                    if k in dets[0][i]:
                        dets[0][i][k] = (np.array(dets[0][i][k], np.float32) / scale).tolist()
        dets_all_post[ratio] = dets[0]  # [item1, item2 ...]
    return dets_all_post

def pre_process(image, scale=1):
    height, width = image.shape[3:5]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height ,
            'out_width': inp_width}
    return meta

def merge_outputs(ratios, dets_all_post, num_class):
    results = defaultdict(list)
    # tracking = defaultdict(list)
    for ratio in reversed(ratios):
        detections = dets_all_post[ratio]
        if ratio == 1:
            for i in range(len(detections)):
                item = detections[i]
                temp_box = [item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3], item['score'][0]]
                results[item['class']].append(temp_box)
                # tracking[item['class']].append(item['tracking'])

    for i in range(1, num_class + 1):
        if len(results[i]) != 0:
            results[i] = np.array(results[i])
    return results

def merge_tracks(dets):
    results = []
    # tracks = []
    for i, item in enumerate(dets):
        if item[4] > opt.conf_thres:
            results.append(item)
            # tracks.append(tracking[i])
    if len(results) == 0:
        return np.empty((0, 5))
    else:
        results = np.array(results)
        return results

def test(opt, split, modelPath, show_flag, results_name):
    # Logger(opt)
    print(opt.model_name)

    # ------------------load data and model------------------
    dataset = COCO(opt, split)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2, 'dis': 2}, opt.model_name)
    model = load_model(model, modelPath)
    model = model.cuda()
    model.eval()

    # some useful initialization
    num_classes = dataset.num_classes
    file_folder_pre = 'INIT'
    trackingflag = 'off'
    im_count = 0
    ratios = [1]
    dets_track = defaultdict(list)

    saveTxt = opt.save_track_results
    if saveTxt:
        track_results_save_dir = os.path.join(opt.save_results_dir, 'trackingResults'+opt.model_name)
        if not os.path.exists(track_results_save_dir):
            os.mkdir(track_results_save_dir)

    num_iters = len(data_loader)
    bar = Bar('processing', max=num_iters)
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        if(ind>len(data_loader)-1):
            break

        bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters,total=bar.elapsed_td, eta=bar.eta_td
        )

        # read images
        # file_folder_cur = pre_processed_images['file_name'][0].split('/')[-2]     # SATMTB
        file_folder_cur = pre_processed_images['file_name'][0].split('_')[0]      # ICPR
        # file_folder_cur = pre_processed_images['file_name'][0].split('/')[-3]       # AIRMOT & Skysat
        meta = pre_process(pre_processed_images['input'], scale=1)
        image = pre_processed_images['input'].cuda()

        # cur image detection
        dets_all = process(model, ratios, image, vid=file_folder_cur+'_'+str(im_count))
        dets_all_post = post_process(ratios, dets_all, meta)
        ret = merge_outputs(ratios, dets_all_post, num_classes)
        
        # update tracker
        for i in range(1, num_classes + 1):
            dets_track[i] = merge_tracks(ret[i])
        
        if file_folder_cur != file_folder_pre:
            trackingflag = 'on'
            if saveTxt and file_folder_pre != 'INIT':
                fid.close()
            file_folder_pre = file_folder_cur
            car_tracker = Sort()    # 1
            # plane_tracker = Sort()    # 2
            # ship_tracker = Sort()    # 3
            # train_tracker = Sort()    # 4
            if saveTxt:
                # load model
                model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2, 'dis': 2}, opt.model_name)
                model = load_model(model, modelPath)
                model = model.cuda()
                model.eval()
                im_count = 0
                txt_path = os.path.join(track_results_save_dir, file_folder_cur+'.txt')
                fid = open(txt_path, 'w+')

        if trackingflag is not 'off':
            for i in range(1, num_classes + 1):
                if i == 1:
                    car_track_bbs_ids = car_tracker.update(dets_track[i])
                    car_track_bbs_ids = car_track_bbs_ids[::-1,:]
                    car_track_bbs_ids[:,2:4] = car_track_bbs_ids[:,2:4] - car_track_bbs_ids[:,:2]
                    car_track_bbs_ids[:, -1] = car_track_bbs_ids[:, -1]
                # if i == 2:
                #     plane_track_bbs_ids = plane_tracker.update(dets_track[i])
                #     plane_track_bbs_ids = plane_track_bbs_ids[::-1,:]
                #     plane_track_bbs_ids[:,2:4] = plane_track_bbs_ids[:,2:4] - plane_track_bbs_ids[:,:2]
                # elif i == 3:
                #     ship_track_bbs_ids = ship_tracker.update(dets_track[i])
                #     ship_track_bbs_ids = ship_track_bbs_ids[::-1,:]
                #     ship_track_bbs_ids[:,2:4] = ship_track_bbs_ids[:,2:4] - ship_track_bbs_ids[:,:2]
                #     ship_track_bbs_ids[:, -1] = ship_track_bbs_ids[:, -1] + 1000
                # else:
                #     train_track_bbs_ids = train_tracker.update(dets_track[i])
                #     train_track_bbs_ids = train_track_bbs_ids[::-1,:]
                #     train_track_bbs_ids[:,2:4] = train_track_bbs_ids[:,2:4] - train_track_bbs_ids[:,:2]
                #     train_track_bbs_ids[:, -1] = train_track_bbs_ids[:, -1] + 2000

            if saveTxt:
                im_count += 1
                for it in range(car_track_bbs_ids.shape[0]):
                    fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,1,1\n'%(im_count,
                            car_track_bbs_ids[it,-1], car_track_bbs_ids[it,0], car_track_bbs_ids[it,1],
                                    car_track_bbs_ids[it, 2], car_track_bbs_ids[it, 3]))
                # for it in range(plane_track_bbs_ids.shape[0]):
                #     fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,2,1\n'%(im_count,
                #             plane_track_bbs_ids[it,-1], plane_track_bbs_ids[it,0], plane_track_bbs_ids[it,1],
                #                     plane_track_bbs_ids[it, 2], plane_track_bbs_ids[it, 3]))
                # for it in range(ship_track_bbs_ids.shape[0]):
                #     fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,3,1\n'%(im_count,
                #             ship_track_bbs_ids[it,-1], ship_track_bbs_ids[it,0], ship_track_bbs_ids[it,1],
                #                     ship_track_bbs_ids[it, 2], ship_track_bbs_ids[it, 3]))
                # for it in range(train_track_bbs_ids.shape[0]):
                #     fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,4,1\n'%(im_count,
                #             train_track_bbs_ids[it,-1], train_track_bbs_ids[it,0], train_track_bbs_ids[it,1],
                #                     train_track_bbs_ids[it, 2], train_track_bbs_ids[it, 3]))

        bar.next()
    bar.finish()

if __name__ == '__main__':
    split = 'val'
    show_flag = False
    if (not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    if opt.load_model != '':
        modelPath = opt.load_model
    else:
        modelPath = './checkpoints/MP2Net.pth'
    print(modelPath)

    results_name = opt.model_name+'_'+modelPath.split('/')[-1].split('.')[0]
    test(opt, split, modelPath, show_flag, results_name)