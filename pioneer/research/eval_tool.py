from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('/media/HardDisk_new/wh/TGSR/')
os.system("cd /media/HardDisk_new/wh/TGSR/pioneer/research/")
import argparse
import cv2
import numpy as np
import collections
import pickle
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
        VOTDataset, NFSDataset, VOTLTDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
        EAOBenchmark, F1Benchmark
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from pysot.utils.bbox import get_axis_aligned_bbox

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str,
                    default='./pioneer/research/',
                    help='tracker result path')
parser.add_argument('--result_form', type=str,
                    default='pkl',
                    help='pkl or txt')
parser.add_argument('--dataset', '-d', type=str,
                    default='VOT2016',
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', default='SiamRPN++_TGSR_VOT2016',
                    type=str, help='tracker name')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def import_results():
    '''
    导入结果
    '''
    if args.result_form == 'pkl':
        # 字典形式
        tracker_names = []
        for a in os.listdir(args.tracker_path):
            if '.pkl' in a and args.dataset in a:
                tracker_names.append(a.split('.')[0])

        if args.tracker_prefix != '':
            tracker_names = [a for a in tracker_names if args.tracker_prefix in a]

        tracker_names.sort()
        print(tracker_names)  # 输出tracker的顺序，以便确定矩形框的颜色
        tracker_last_results = tracker_names.copy()

        assert len(tracker_names) > 0

        trackers_results = dict()
        for tracker_name in tracker_names:
            tracker_results_path = os.path.join(args.tracker_path, tracker_name+'.pkl')
            with open(tracker_results_path, 'rb') as f:
                trackers_results[tracker_name] = pickle.load(f)
    else:
        # txt形式
        # 读取结果各个tracker的结果
        results_path = os.path.join(args.tracker_path, args.dataset)
        trackers_results = dict()
        tracker_names = sorted(os.listdir(results_path))

        if args.tracker_prefix != '':
            tracker_names = [a for a in tracker_names if args.tracker_prefix in a]

        print(tracker_names)  # 输出tracker的顺序，以便确定矩形框的颜色
        # tracker_last_results 在VOT中，用于保存上一帧结果。当前帧结果输出不正确时，输出上一帧结果。
        tracker_last_results = tracker_names.copy()

        assert len(tracker_names) > 0

        if 'VOT' in args.dataset:
            # VOT
            for tracker_name in tracker_names:
                tracker_results = dict()
                # VOT
                tracker_results_path = os.path.join(results_path, tracker_name, 'baseline')
                # tracker_results_path = os.path.join(results_path, tracker_name)
                for video_name in sorted(os.listdir(tracker_results_path)):
                    video_result_name = video_name + '_001.txt'
                    video_result_path = os.path.join(tracker_results_path, video_name, video_result_name)
                    with open(video_result_path, 'r') as f:
                        lines = f.readlines()
                        results = []
                        for line in lines:
                            line = line.rstrip()
                            if len(line) == 1:
                                result = [1]
                            else:
                                result = [float(x) for x in line.split(',')]
                            results.append(result)

                    tracker_results[video_name] = results
                trackers_results[tracker_name] = tracker_results
        else:
            # OTB
            for tracker_name in tracker_names:
                tracker_results = dict()
                # VOT
                # tracker_results_path = os.path.join(results_path, tracker_name, 'baseline')
                tracker_results_path = os.path.join(results_path, tracker_name)
                for video_name in sorted(os.listdir(tracker_results_path)):
                    video_name = video_name.split('.')[0]
                    video_result_path = os.path.join(tracker_results_path, video_name + '.txt')
                    with open(video_result_path, 'r') as f:
                        lines = f.readlines()
                        results = []
                        for line in lines:
                            line = line.rstrip()
                            if len(line) == 1:
                                result = [1]
                            else:
                                result = [float(x) for x in line.split(',')]
                            results.append(result)

                    tracker_results[video_name] = results
                trackers_results[tracker_name] = tracker_results

    return tracker_names, trackers_results


def analyse_continuity(dataset, tracker_names, trackers_results):
    """
    分析Refine对提高连续预测帧数
    """
    continuity_increase = dict()
    for tracker_name in tracker_names:
        continuity_increase[tracker_name]= []

        if 'VOT' in dataset.name:
            for video_name in trackers_results[tracker_name].keys():
                max_length_tem = 0
                max_length_memory = []
                for idx in trackers_results[tracker_name][video_name].keys():

                    if trackers_results[tracker_name][video_name][idx]['Flag'] == True:
                        max_length_tem += 1
                    else:
                        # 中途跟踪失败
                        max_length_memory.append(max_length_tem)
                        max_length_tem = 0
                # 整个序列成功跟踪
                max_length_memory.append(max_length_tem)
                # 记录最长帧数
                max_length = max(max_length_memory)
                continuity_increase[tracker_name].append(max_length)
        else:
            # 读取dataset数据
            for v_idx, video in enumerate(dataset):

                # if video.name != 'Trans':
                #     continue

                max_length = 0
                for idx, (img, gt_bbox) in enumerate(video):

                    if idx == 0:
                        continue

                    if len(gt_bbox) == 4:
                        gt_bbox = [gt_bbox[0], gt_bbox[1],
                                   gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                                   gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                                   gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]


                        result = trackers_results[tracker_name][video.name][idx]['original_bbox']
                        overlap = vot_overlap(result, gt_bbox, (img.shape[1], img.shape[0]))

                        if overlap > 0:
                            max_length += 1

                continuity_increase[tracker_name].append(max_length)

        print(tracker_name+':', continuity_increase[tracker_name])
        print(tracker_name +' MAEN MAX' +':', np.mean(continuity_increase[tracker_name]))
    return continuity_increase





if __name__ == '__main__':
    dataset_root = '/media/HardDisk_new/DataSet/test/' + args.dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    tracker_names, trackers_results = import_results()

    analyse_continuity(dataset, tracker_names, trackers_results)