from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from collections import defaultdict

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from lib.utils.image import flip, color_aug
from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils.image import draw_dense_reg
import math
from lib.utils.opts import opts

from lib.utils.augmentations import Augmentation

import torch.utils.data as data

class COCO(data.Dataset):
    opt = opts().parse()
    num_classes = opt.num_classes
    reg_offset = True
    mean = np.array([0.49965, 0.49965, 0.49965],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.08255, 0.08255, 0.08255],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(COCO, self).__init__()

        self.img_dir0 = self.opt.data_dir   #/dataset/ICPR

        self.img_dir = self.opt.data_dir + split + '_data/'

        if split == 'train':
            self.annot_path = os.path.join(
                self.img_dir0, 'annotations',
                'instances_{}_caronly.json').format(split)
        else:
            self.annot_path = os.path.join(
                self.img_dir0, 'annotations',
                'instances_{}_caronly.json').format(split)

        self.down_ratio = opt.down_ratio
        self.max_objs = opt.K
        self.seqLen = opt.seqLen

        self.class_name = [
            'car', 'airplane', 'ship', 'train']
        self._valid_ids = [
            1, 2, 3, 4]
        # self.class_name = [
        #     'car']
        # self._valid_ids = [
        #     1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}  # 生成对应的category dict

        self.split = split
        self.opt = opt

        print('==> initializing ICPR {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

        if(split=='train'):
            self.aug = Augmentation(opt)
        else:
            self.aug = None

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    # 遍历每一个标注文件解析写入detections. 输出结果使用
    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir, time_str):
        json.dump(self.convert_eval_format(results),
                  open('{}/results_{}.json'.format(save_dir,time_str), 'w'))

        print('{}/results_{}.json'.format(save_dir,time_str))

    def run_eval(self, results, save_dir, time_str):
        self.save_results(results, save_dir, time_str)
        coco_dets = self.coco.loadRes('{}/results_{}.json'.format(save_dir, time_str))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        precisions = coco_eval.eval['precision']

        return stats, precisions

    def run_eval_just(self, save_dir, time_str, iouth):
        coco_dets = self.coco.loadRes('{}/{}'.format(save_dir, time_str))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox", iouth = iouth)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_5 = coco_eval.stats
        precisions = coco_eval.eval['precision']

        return stats_5, precisions

    def _coco_box_to_bbox(self, box):
        if len(box) == 0:
            return box
        else:
            bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                            dtype=np.float32)
            return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_transoutput(self, c, s, height, width):
        trans_output = get_affine_transform(c, s, 0, [width, height])
        return trans_output

    def transform_box(self, bbox, transform, output_w, output_h):
        bbox[:2] = affine_transform(bbox[:2], transform)
        bbox[2:] = affine_transform(bbox[2:], transform)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        h = np.clip(h, 0, output_h - 1)
        w = np.clip(w, 0, output_w - 1)
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct[0] = np.clip(ct[0], 0, output_w - 1)
        ct[1] = np.clip(ct[1], 0, output_h - 1)
        return bbox, ct, radius

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']    #001_000001.jpg
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        seq_num = self.seqLen
        imIdex = file_name.split('.')[0].split('_')[-1]         #000001
        imf = file_name.split('_')[0]                           #001
        # imIdex = file_name.split('.')[0].split('/')[-1]         #000001
        # imf = file_name.split('/')[1]           #001
        imtype = '.'+file_name.split('.')[-1]                   #.jpg
        im0 = cv2.imread(self.img_dir + imf+'/img1/'+imIdex+imtype)
        img = np.zeros([im0.shape[0], im0.shape[1], 3, seq_num])
        pre_revrs_ids = []        # len(pre_img_ids) = seq_num - 1
        interval = []
        temp_id = img_id

        for ii in range(seq_num):
            # 0, 1, ..., seq_num - 1
            imIndexNew = '%06d' % max(int(imIdex) - seq_num + ii + 1, 1)
            imName = imf+'/img1/'+imIndexNew+imtype
            if ii <= int(imIdex) - 1:
                temp_id = img_id - ii
            if ii != 0:
                pre_revrs_ids.append(temp_id)
                interval.append(img_id - temp_id)
            im = cv2.imread(self.img_dir + imName)
            #normalize
            inp_i = (im.astype(np.float32) / 255.)
            inp_i = (inp_i - self.mean) / self.std
            img[:,:,:,ii] = inp_i

        pre_anns = defaultdict(list)
        for ii in range(seq_num - 1):
            pre_ann_ids = self.coco.getAnnIds(imgIds=pre_revrs_ids[seq_num - 2 - ii])
            pre_anns[ii] = self.coco.loadAnns(ids=pre_ann_ids)  # ..., N-2, N-1
        
        bbox_tol = []
        cls_id_tol = []
        ids_tol = []
        for k in range(num_objs):
            ann = anns[k]
            bbox_tol.append(self._coco_box_to_bbox(ann['bbox']))
            cls_id_tol.append(self.cat_ids[ann['category_id']])
            ids_tol.append(int(ann['obj_id']))

        # get box and id of pre annotations
        pre_bboxes = defaultdict(list)
        pre_ids = defaultdict(list)
        for i in range(seq_num - 1):
            for k in range(len(pre_anns[i])):
                ann = pre_anns[i][k]
                pre_bboxes[i + 1].append(self._coco_box_to_bbox(ann['bbox']))
                pre_ids[i + 1].append(int(ann['obj_id']))

        if self.aug is not None:
            bbox_tol = np.array(bbox_tol)
            cls_id_tol = np.array(cls_id_tol)
            ids_tol = np.array(ids_tol)
            for i in range(seq_num - 1):
                pre_bboxes[i + 1] = np.array(pre_bboxes[i + 1])
                pre_ids[i + 1] = np.array(pre_ids[i + 1])
            img, _, bbox_tol, cls_id_tol, ids_tol, pre_bboxes, pre_ids = self.aug(img, img, bbox_tol, cls_id_tol, ids_tol, pre_bboxes, pre_ids)
            bbox_tol = bbox_tol.tolist()
            cls_id_tol = cls_id_tol.tolist()
            ids_tol = ids_tol.tolist()
            for i in range(seq_num - 1):
                pre_bboxes[i + 1] = pre_bboxes[i + 1].tolist()
                pre_ids[i + 1] = pre_ids[i + 1].tolist()
            num_objs = len(bbox_tol)

        #transpose
        inp = img.transpose(3, 2, 0, 1).astype(np.float32)
        # inp = img[:,:,:,0].transpose(2, 0, 1).astype(np.float32)
        height, width = img.shape[0] - img.shape[0] % 8, img.shape[1] - img.shape[1] % 8
        inp = inp[:, :, 0:height, 0:width]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(width, height) * 1.0
        ret = {'input': inp}

        down_ratios = [1]
        for ratio in down_ratios:
            output_h = height // ratio // self.down_ratio
            output_w = width // ratio // self.down_ratio
            trans_output = self._get_transoutput(c, s, output_h, output_w)

            hm = np.zeros((seq_num, self.num_classes, output_h, output_w), dtype=np.float32)
            hm_seq = np.zeros((seq_num, 1, output_h, output_w), dtype=np.float32)
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)
            ind = np.zeros((self.max_objs), dtype=np.int64)
            reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

            ind_dis = np.zeros((seq_num - 1, self.max_objs), dtype=np.int64)
            dis_mask = np.zeros((seq_num - 1, self.max_objs), dtype=np.uint8)
            dis = np.zeros((seq_num - 1, self.max_objs, 2), dtype=np.float32)

            gt_det = []
            for k in range(num_objs):
                bbox = bbox_tol[k]
                cls_id = cls_id_tol[k]
                obj_id = ids_tol[k]
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)

                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                h = np.clip(h, 0, output_h - 1)
                w = np.clip(w, 0, output_w - 1)
                if h > 0 and w > 0:
                    for i in range(1, seq_num):
                        if i != seq_num - 1:
                            if pre_ids[i].count(obj_id) != 0 and pre_ids[i + 1].count(obj_id) != 0:
                                temp_ind_pre = pre_ids[i].index(obj_id)
                                temp_ind_later = pre_ids[i + 1].index(obj_id)
                                temp_box_pre = pre_bboxes[i][temp_ind_pre]
                                temp_box_later = pre_bboxes[i + 1][temp_ind_later]

                                temp_box_pre, ct_pre, radius_pre = self.transform_box(temp_box_pre, trans_output, output_w, output_h)
                                temp_box_later, ct_later, radius_later = self.transform_box(temp_box_later, trans_output, output_w, output_h)
                                
                                ct_int_pre = ct_pre.astype(np.int32)
                                draw_umich_gaussian(hm[i - 1][cls_id], ct_int_pre, radius_pre)
                                draw_umich_gaussian(hm_seq[i - 1][0], ct_int_pre, radius_pre)
                                ind_dis[i - 1][temp_ind_pre] = ct_int_pre[1] * output_w + ct_int_pre[0]
                                dis[i - 1][temp_ind_pre] = ct_later - ct_pre
                                dis_mask[i - 1][temp_ind_pre] = 1
                        else:
                            if pre_ids[i].count(obj_id) != 0:
                                temp_ind_pre = pre_ids[i].index(obj_id)
                                temp_box_pre = pre_bboxes[i][temp_ind_pre]
                                temp_box_later = bbox_tol[k]

                                temp_box_pre, ct_pre, radius_pre = self.transform_box(temp_box_pre, trans_output, output_w, output_h)
                                temp_box_later, ct_later, radius_later = self.transform_box(temp_box_later, trans_output, output_w, output_h)
                                
                                ct_int_pre = ct_pre.astype(np.int32)
                                draw_umich_gaussian(hm[i - 1][cls_id], ct_int_pre, radius_pre)
                                draw_umich_gaussian(hm_seq[i - 1][0], ct_int_pre, radius_pre)
                                ind_dis[i - 1][temp_ind_pre] = ct_int_pre[1] * output_w + ct_int_pre[0]
                                dis[i - 1][temp_ind_pre] = ct_later - ct_pre
                                dis_mask[i - 1][temp_ind_pre] = 1

                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct[0] = np.clip(ct[0], 0, output_w - 1)
                    ct[1] = np.clip(ct[1], 0, output_h - 1)
                    ct_int = ct.astype(np.int32)
                    draw_umich_gaussian(hm[seq_num - 1][cls_id], ct_int, radius)
                    draw_umich_gaussian(hm_seq[seq_num - 1][0], ct_int, radius)
                    wh[k] = 1. * w, 1. * h
                    ind[k] = ct_int[1] * output_w + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1
                    gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                                ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
            ret[ratio] = {'hm': hm, 'hm_seq': hm_seq, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'dis_ind': ind_dis, 'dis': dis, 'dis_mask': dis_mask}

        for kkk in range(num_objs, self.max_objs):
            bbox_tol.append([])

        ret['file_name'] = file_name

        return img_id, ret