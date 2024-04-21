import torch
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from collections import defaultdict
from lib.utils.utils import _transpose_and_gather_feat
import copy

def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)

def check(opt, file_folder_buffer, dets_buffer, dis_buffer, inds_buffer):
    if file_folder_buffer[opt.seqLen - 1] != file_folder_buffer[0]:
        return dets_buffer, inds_buffer
    # 第0个是准的
    cur_dismap = dis_buffer[opt.seqLen - 1]      # dis: B, N-1, 2, H, W
    _, _, _, H, W = cur_dismap.shape
    cur_dets = defaultdict(list)
    whs = defaultdict(list)
    hits = defaultdict(list)
    match_inds = defaultdict(list)
    unmatch_preds = defaultdict(list)
    unmatch_curs = defaultdict(list)
    inds_temp_buffer = inds_buffer.copy()
    for i in range(opt.seqLen - 1):       # 0, 1, 2
        # cur dets
        cur_dets[i] = cur_dets[i] + dets_buffer[i + 1]
        M = len(cur_dets[i])
        cur_cts = np.array([((item[0] + item[2]) / 2, (item[1] + item[3]) / 2) for item in cur_dets[i]])  # M * 2, cur_cts
        cur_whs = np.array([[(item[2] - item[0]), (item[3] - item[1]), item[4], item[5], item[6]] for item in cur_dets[i]])    # M * 4
        cur_sizes = np.array([((item[2] - item[0]) * (item[3] - item[1])) for item in cur_dets[i]])   # M, cur_det的尺寸集合

        # generate cur dis and perform predict
        # 第0帧用原始inds_buffer[0], 后续帧应该用更新的inds, 不过还是inds_buffer[i]
        if i == 0:
            pre_whs = np.array([[(item[2] - item[0]), (item[3] - item[1]), item[4], item[5], item[6]] for item in dets_buffer[0]])   # K * 4
        else:
            pre_whs = np.array(whs[i - 1])          # M+K-T * 4
        K = len(inds_temp_buffer[i])
        pre_ind = torch.from_numpy(inds_temp_buffer[i]).unsqueeze(0)     # B, K
        pre_ys = (pre_ind / W).int().float()        # B, K
        pre_xs = (pre_ind % W).int().float()        # B, K
        pre_cts = torch.cat([pre_xs.unsqueeze(-1), pre_ys.unsqueeze(-1)], dim=2)     # B, K, 2
        pre2cur_dis = _transpose_and_gather_feat(cur_dismap[:, i], pre_ind).view(1, -1, 2)  # B, K, 2
        pred_cts = (pre_cts + pre2cur_dis)[0].numpy()            # K, 2
        pred_cts[pred_cts < 0] = 0.
        pred_cts[pred_cts >= W] = W - 1
        pred_dis = np.array([(item[0]**2 + item[1]**2) for item in pre2cur_dis[0].numpy()])     # K, 
        pre_sizes = np.array([(item[0] * item[1]) for item in pre_whs])                         # K, pre_det的尺寸集合
        
        # match cur cts with pred cts
        dists = (((cur_cts.reshape(1, -1, 2) - pred_cts.reshape(-1, 1, 2)) ** 2).sum(axis=2))       # K x M
        invalid = ((dists > cur_sizes.reshape(1, M)) + (dists > pre_sizes.reshape(K, 1))) > 0        # K x M
        dists = dists + invalid * 1e18

        if i == 0:
            # 第0帧似乎只需要unmatch_preds
            match_inds[i + 1] = greedy_assignment(copy.deepcopy(dists))
            unmatch_preds[i + 1] = [d for d in range(pred_cts.shape[0]) if not (d in match_inds[i + 1][:, 0])]     # 对应K
            unmatch_curs[i + 1] = [d for d in range(cur_cts.shape[0]) if not (d in match_inds[i + 1][:, 1])]       # 对应M
        else:
            match_inds[i + 1] = greedy_assignment(copy.deepcopy(dists))
            # unmatch_preds[i + 1] = [d for d in range(pred_cts.shape[0]) if not (d in match_inds[i + 1][:, 0])]     # 对应K
            # unmatch_curs[i + 1] = [d for d in range(cur_cts.shape[0]) if not (d in match_inds[i + 1][:, 1])]       # 对应M

        # 核心 更新第1个的inds和dets
        if i == 0:
            # 第1个需要更新unmatch_pred中的ind到cur_ind中去
            inds_temp = inds_temp_buffer[i + 1].tolist()    # 检测直出, 应该是M个, 并假设M与Kmatch到T个
            whs_temp = cur_whs.tolist()                              # M * 5
            # print(inds_temp, 'inds_temp', M)

            cts_update = np.array(pred_cts[unmatch_preds[i + 1], :], np.int32)  # K-T个
            inds_update = cts_update[:, 0] + cts_update[:, 1] * W
            whs_update = pre_whs[unmatch_preds[i + 1]].tolist()                 # K-T * 5
            unmatch_pred0_update = (M + np.array(range(len(unmatch_preds[i + 1])))).tolist()
            # print(inds_update.tolist(), 'inds_update')

            inds_temp = inds_temp + inds_update.tolist()                # M+K-T个
            inds_temp_buffer[i + 1] = np.array(inds_temp, np.int64)
            whs[i] = whs_temp + whs_update                              # M+K-T * 5

            hits[i] = np.zeros_like(inds_temp_buffer[i + 1])            # M+K-T
            hits[i][match_inds[i + 1][:, 1]] += 1
        else:
            # 后续需要检验第1个更新的ind在后面是否有可以match的cur_ct
            # 如果有，则保留第一个中更新的ind，没有则不保留
            # 维护一个hits number的东西, 如果match到了对应ind位置的hit+=1, 最后把hit=0的都删了
            # match_inds[i + 1][:, 0]对应pred, match_inds[i + 1][:, 1]对应cur
            hits[i] = np.zeros_like(inds_temp_buffer[i])
            hits[i][match_inds[i + 1][:, 0]] += 1
            cts_update = np.array(pred_cts, np.int32)
            inds_update = cts_update[:, 0] + cts_update[:, 1] * W

            for j in range(match_inds[i + 1].shape[0]):
                pre_whs[match_inds[i + 1][j, 0]][0:5] = cur_whs[match_inds[i + 1][j, 1]][0:5]
                inds_update[match_inds[i + 1][j, 0]] = cur_whs[match_inds[i + 1][j, 1]][4]
            
            inds_temp_buffer[i + 1] = np.array(inds_update, np.int64)
            whs[i] = pre_whs.tolist()

    # 返回来修改第1个
    hits_sum = hits[0] + hits[1] + hits[2] + hits[3]      # seqLen == 5
    # hits_sum = hits[0] + hits[1] + hits[2]
    remain_inds = np.where(hits_sum>0)[0]
    # print(remain_inds, len(inds_temp_buffer[1]), len(inds_temp_buffer[2]), len(inds_temp_buffer[3]))
    # print(unmatch_preds[1])
    # inds_temp_buffer: 第1帧的det_buffer[1] + 第0帧的所有未匹配预测
    # inds_buffer: 与该帧det_buffer相对应
    # remain_inds: 既有原第1帧det_buffer, 又有第0帧预测的
    # unmatch_preds: 所有未匹配预测
    inds_temp1 = inds_buffer[1].tolist()
    inds_temp2 = inds_buffer[2].tolist()
    inds_temp3 = inds_buffer[3].tolist()
    inds_temp4 = inds_buffer[4].tolist()    # seqLen == 5

    FN_ind, FP_ind = [], []
    # 先对应处理FP, 没问题
    for i, ind in enumerate(inds_temp1):
        if ind not in inds_temp_buffer[1][remain_inds].tolist() and i in unmatch_curs[1]:
            FP_ind.append(i)
    for i in sorted(FP_ind, reverse=True):
        del dets_buffer[1][i]
        del inds_temp1[i]

    # 再对应处理FN
    for i, num in enumerate(unmatch_pred0_update): # 序号重排后
        temp_ind = inds_temp_buffer[1][num]
        # if ind not in inds_temp1 and i in remain_inds:
        #     FN_ind.append(i)
        if temp_ind > 0 and temp_ind < H * W:
            ct_y = temp_ind / W
            ct_x = temp_ind % W
            w, h, score, cls, _ = whs[0][num]
            if num in remain_inds:
                hits[0][num] += 1
                inds_temp1.append(temp_ind)
                dets_buffer[1].append([ct_x - w / 2., ct_y - h / 2., ct_x + w / 2., ct_y + h / 2., score, cls, temp_ind])
    inds_buffer[1] = np.array(inds_temp1)

    # 修改后面的
    hits_sum = hits[1] + hits[2] + hits[3]      # seqLen == 5
    # hits_sum = hits[1] + hits[2]
    remain_inds = np.where((hits[0]>0) & (hits_sum>1))[0]
    FN_ind = []
    for i in remain_inds:
        if hits[1][i] == 0:
            FN_ind.append(i)
    for i in FN_ind:
        temp_ind = inds_temp_buffer[2][i]
        if temp_ind > 0 and temp_ind < H * W:
            ct_y = temp_ind / W
            ct_x = temp_ind % W
            w, h, score, cls, _ = whs[1][i]
            inds_temp2.append(temp_ind)
            dets_buffer[2].append([ct_x - w / 2., ct_y - h / 2., ct_x + w / 2., ct_y + h / 2., score, cls, temp_ind])
    inds_buffer[2] = np.array(inds_temp2)

    FN_ind = []
    for i in remain_inds:
        if hits[2][i] == 0:
            FN_ind.append(i)
    for i in FN_ind:
        temp_ind = inds_temp_buffer[3][i]
        if temp_ind > 0 and temp_ind < H * W:
            ct_y = temp_ind / W
            ct_x = temp_ind % W
            w, h, score, cls, _ = whs[2][i]
            inds_temp3.append(temp_ind)
            dets_buffer[3].append([ct_x - w / 2., ct_y - h / 2., ct_x + w / 2., ct_y + h / 2., score, cls, temp_ind])
    inds_buffer[3] = np.array(inds_temp3)

    return dets_buffer, inds_buffer