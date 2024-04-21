from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
from lib.tracking_utils.discheck import check

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
        arrow = dis[i][-2] * 10
        img = cv2.arrowedLine(img,
                    pt1=(int(ct1), int(ct2)),
                    pt2=(int(ct1 + arrow[0]), int(ct2 + arrow[1])),
                    color=(255, 255, 0),
                    thickness=2, tipLength=0.5)
    cv2.imwrite(writename, img)
    return

def process(model, image, vid=None):
    with torch.no_grad():
        output_all = model(image, training=False, vid=vid)[-1]
        output = output_all[1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        dis = output['dis'].detach().cpu()
        torch.cuda.synchronize()
        dets = ctdet_decode(hm, wh, reg=reg, num_classes=opt.num_classes, K=opt.K)
        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()
        # mask_seq = show_mask(F.sigmoid(output['hm_seq'][0, -1]))
    return dets, dis

def post_process(dets_all, meta, scale=1):
    dets = generic_post_process(
        dets_all, [meta['c']], [meta['s']], meta['out_height'] // opt.down_ratio, meta['out_width'] // opt.down_ratio)
    if scale != 1:
        for i in range(len(dets[0])):
            for k in ['bbox']:
                if k in dets[0][i]:
                    dets[0][i][k] = (np.array(dets[0][i][k], np.float32) / scale).tolist()
    return dets[0]  # [item1, item2 ...]

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

def merge_outputs(dets_all_post, num_class):
    results = []
    inds = []
    detections = dets_all_post
    for i in range(len(detections)):
        item = detections[i]
        temp_box = [item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3], item['score'][0], item['class'], item['ind']]
        if item['score'][0] > opt.conf_thres:
            results.append(temp_box)
            inds.append(item['ind'])

    # for i in range(1, num_class + 1):
    #     if len(results[i]) != 0:
    #         results[i] = np.array(results[i])
    #     else:
    #         results[i] = np.empty((0, 5))
    return results, np.array(inds)

def test(opt, split, modelPath, show_flag, results_name):
    # Logger(opt)
    print(opt.model_name)

    # ------------------load data and model------------------
    dataset = COCO(opt, split)
    num_classes = 1
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    model = get_det_net({'hm': num_classes, 'wh': 2, 'reg': 2, 'dis': 2}, opt.model_name)
    model = load_model(model, modelPath)
    model = model.cuda()
    model.eval()

    # some useful initialization
    file_folder_pre = 'INIT'
    trackingflag = 'off'
    im_count = 0
    buffer_size = opt.seqLen
    dets_track = defaultdict(list)
    dets_buffer = defaultdict(list)
    dis_buffer = defaultdict(list)
    inds_buffer = defaultdict(list)
    file_folder_buffer = defaultdict(list)

    saveTxt = opt.save_track_results
    if saveTxt:
        track_results_save_dir = os.path.join(opt.save_results_dir, 'trackingResults'+opt.model_name)
        if not os.path.exists(track_results_save_dir):
            os.mkdir(track_results_save_dir)
    # ------------------initialization ends------------------

    num_iters = len(data_loader)
    bar = Bar('processing', max=num_iters)
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        if(ind>len(data_loader)-1):
            break

        bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters,total=bar.elapsed_td, eta=bar.eta_td)

        # read images
        # file_folder_cur = pre_processed_images['file_name'][0].split('/')[-2]     # SATMTB
        file_folder_cur = pre_processed_images['file_name'][0].split('_')[0]      # ICPR
        # file_folder_cur = pre_processed_images['file_name'][0].split('/')[-3]       # AIRMOT & Skysat
        meta = pre_process(pre_processed_images['input'], scale=1)
        image = pre_processed_images['input'].cuda()

        # cur image detection
        dets_all, cur_dis = process(model, image, vid=file_folder_cur+'_'+str(im_count))
        dets_all_post = post_process(dets_all, meta)
        ret, cur_inds = merge_outputs(dets_all_post, num_classes)

        # detection buffer
        for j in range(buffer_size):
            if j != buffer_size - 1:
                file_folder_buffer[j] = file_folder_buffer[j + 1]
                dets_buffer[j] = dets_buffer[j + 1]
                dis_buffer[j] = dis_buffer[j + 1]
                inds_buffer[j] = inds_buffer[j + 1]
            else:
                file_folder_buffer[j] = file_folder_cur
                dets_buffer[j] = ret
                dis_buffer[j] = cur_dis
                inds_buffer[j] = cur_inds
        
        if file_folder_buffer[0] != file_folder_pre and len(file_folder_buffer[0]) != 0:
            trackingflag = 'on'
            if saveTxt and file_folder_pre != 'INIT':
                fid.close()
            file_folder_pre = file_folder_buffer[0]
            car_tracker = Sort()    # 1
            # plane_tracker = Sort()    # 2
            # ship_tracker = Sort()    # 3
            # train_tracker = Sort()    # 4
            if saveTxt:
                # load model
                model = get_det_net({'hm': num_classes, 'wh': 2, 'reg': 2, 'dis': 2}, opt.model_name)
                model = load_model(model, modelPath)
                model = model.cuda()
                model.eval()
                im_count = 0
                txt_path = os.path.join(track_results_save_dir, file_folder_buffer[0]+'.txt')
                fid = open(txt_path, 'w+')

        if trackingflag is not 'off':
            '''
            # input: dets_buffer, file_folder_buffer, inds_buffer
            '''
            dets_buffer, inds_buffer = check(opt, file_folder_buffer, dets_buffer, dis_buffer, inds_buffer)

            for i in range(1, num_classes + 1):
                # update tracker
                track_temp = []
                for item in dets_buffer[0]:
                    if int(item[5]) == i:
                        track_temp.append(item)
                if len(track_temp) != 0:
                    dets_track[i] = np.array(track_temp)
                else:
                    dets_track[i] = np.empty((0, 8))

                # begin tracking
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
                            car_track_bbs_ids[it,4], car_track_bbs_ids[it,0], car_track_bbs_ids[it,1],
                                    car_track_bbs_ids[it, 2], car_track_bbs_ids[it, 3]))
                # for it in range(plane_track_bbs_ids.shape[0]):
                #     fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,2,1\n'%(im_count,
                #             plane_track_bbs_ids[it,4], plane_track_bbs_ids[it,0], plane_track_bbs_ids[it,1],
                #                     plane_track_bbs_ids[it, 2], plane_track_bbs_ids[it, 3]))
                # for it in range(ship_track_bbs_ids.shape[0]):
                #     fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,3,1\n'%(im_count,
                #             ship_track_bbs_ids[it,4], ship_track_bbs_ids[it,0], ship_track_bbs_ids[it,1],
                #                     ship_track_bbs_ids[it, 2], ship_track_bbs_ids[it, 3]))
                # for it in range(train_track_bbs_ids.shape[0]):
                #     fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,4,1\n'%(im_count,
                #             train_track_bbs_ids[it,4], train_track_bbs_ids[it,0], train_track_bbs_ids[it,1],
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