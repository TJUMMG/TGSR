# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
sys.path.append('/media/HardDisk_new/wh/TGSR/')
os.system("cd /media/HardDisk_new/wh/TGSR/tools")

import cv2
import torch
import numpy as np
import pickle

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from toolkit.utils.statistics import overlap_ratio

parser = argparse.ArgumentParser(description='siammask tracking')
parser.add_argument('--dataset',default='VOT2016',
                    type=str,help='datasets')
parser.add_argument('--config', default='./experiments/siamrpn_r50_l234_dwxcorr/config.yaml',
                    type=str, help='config file')
parser.add_argument('--snapshot', default='./experiments/siamrpn_r50_l234_dwxcorr/model.pth',
                    type=str,help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,help='eval one special video')
parser.add_argument('--vis', action='store_true',help='whether visualzie result')
parser.add_argument('--model_save_name', default ='SiamRPN++_TGSR_1',
                    type = str, help = 'model_save_name')
parser.add_argument('--savepath', default = './tools/',
                    type = str,
                    help = 'savepath')
args = parser.parse_args()

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(traj_checkpoint, iou_checkpoint, refine_checkpoint,
         save_name=args.model_save_name):
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_root = os.path.join('/media/HardDisk_new/DataSet/test/', args.dataset)


    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot)
    model.pred.load_state_dict(torch.load(traj_checkpoint))

    iou_pred_model_param = torch.load(iou_checkpoint)

    model.iou_pred.load_state_dict(iou_pred_model_param, strict=True)
    model.refine_module.load_state_dict(torch.load(refine_checkpoint))
    model.cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = save_name

    result_dict_save = dict()

    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue

            # if video.name != 'Skiing':
            #     continue

            # if v_idx >= 6:
            #     continue

            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []

            result_dict_save[video.name]= dict()

            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    tracker.init(img, gt_bbox_, args.dataset)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)

                    if idx != 0:
                        # 重新初始化
                        result_dict_save[video.name][idx] = dict()
                        result_dict_save[video.name][idx]['Flag'] = False

                elif idx > frame_counter:
                    outputs = tracker.track(img, gt_bbox_)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)

                        # 将结果保存为dict
                        result_dict_save[video.name][idx] = dict()
                        result_dict_save[video.name][idx]['Flag'] = True
                        result_dict_save[video.name][idx]['bbox'] = outputs['bbox']
                        result_dict_save[video.name][idx]['crop_bbox'] = outputs['crop_bbox']
                        result_dict_save[video.name][idx]['gt_on_crop'] = outputs['gt_on_crop']
                        result_dict_save[video.name][idx]['bbox_on_crop'] = outputs['bbox_on_crop']
                        result_dict_save[video.name][idx]['gt_iou'] = outputs['gt_iou']
                        result_dict_save[video.name][idx]['pred_iou'] = outputs['pred_iou']
                        result_dict_save[video.name][idx]['is_refine'] = outputs['is_refine']
                        result_dict_save[video.name][idx]['refine_bbox_on_img'] = outputs['refine_bbox_on_img']
                        result_dict_save[video.name][idx]['original_bbox'] = outputs['original_bbox']
                        result_dict_save[video.name][idx]['pred_refine_iou'] = outputs['pred_refine_iou']
                        result_dict_save[video.name][idx]['gt_refine_iou'] = outputs['gt_refine_iou']

                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1

                        result_dict_save[video.name][idx] = dict()
                        result_dict_save[video.name][idx]['Flag'] = False

                else:
                    pred_bboxes.append(0)

                    result_dict_save[video.name][idx] = dict()
                    result_dict_save[video.name][idx]['Flag'] = False

                toc += cv2.getTickCount() - tic

                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)


            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(args.savepath,'results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number

        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue

            # if video.name != 'Skiing':
            #     continue

            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []

            result_dict_save[video.name] = dict()

            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                if idx == 0:
                    tracker.init(img, gt_bbox_, args.dataset)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    # outputs = tracker.track(img)
                    outputs = tracker.track(img, gt_bbox_)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])

                    # 将结果保存为dict
                    result_dict_save[video.name][idx] = dict()
                    result_dict_save[video.name][idx]['Flag'] = True
                    result_dict_save[video.name][idx]['bbox'] = outputs['bbox']
                    result_dict_save[video.name][idx]['crop_bbox'] = outputs['crop_bbox']
                    result_dict_save[video.name][idx]['gt_on_crop'] = outputs['gt_on_crop']
                    result_dict_save[video.name][idx]['bbox_on_crop'] = outputs['bbox_on_crop']
                    result_dict_save[video.name][idx]['gt_iou'] = outputs['gt_iou']
                    result_dict_save[video.name][idx]['pred_iou'] = outputs['pred_iou']
                    result_dict_save[video.name][idx]['is_refine'] = outputs['is_refine']
                    result_dict_save[video.name][idx]['refine_bbox_on_img'] = outputs['refine_bbox_on_img']
                    result_dict_save[video.name][idx]['original_bbox'] = outputs['original_bbox']
                    result_dict_save[video.name][idx]['pred_refine_iou'] = outputs['pred_refine_iou']
                    result_dict_save[video.name][idx]['gt_refine_iou'] = outputs['gt_refine_iou']


                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join(args.savepath,'results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join(args.savepath,'results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join(args.savepath, 'results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))

        torch.cuda.empty_cache()

    # 保存result_dict_save为pkl文件，用于分析ALTL指标
    # result_dict_save_path = 'Val1001/pkl/' + args.model_save_name + args.dataset + '.pkl'
    # with open(result_dict_save_path, 'wb') as f:
    #     pickle.dump(result_dict_save, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    traj_checkpoint = './snapshot_test/traj.pth'
    iou_checkpoint = './snapshot_test/ipn_rpnpp.pth'
    refine_checkpoint = './snapshot_test/rbn_rpnpp.pth'

    main(traj_checkpoint=traj_checkpoint,
         iou_checkpoint=iou_checkpoint,
         refine_checkpoint=refine_checkpoint)
