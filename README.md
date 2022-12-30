# TGSR
This is the official implementation with training code for “Trajectory Guided Robust Visual Object Tracking with Selective Remedy”.

Introduction
--------------------------------
In the paper, we propose a generic, fast and flexible approach to improve the robustness of Siamese trackers with two light-load novel modules: Trajectory Guidance Module (TGM) and Selective Refinement Module (SRM). Specifically, TGM encourages to pay a soft attention on possible target location based on short-term historical trajectory. SRM selectively remedies the tracking results at the risk of failure with little impact on the speed. The proposed algorithm can be easily establish upon state-of-the-art Siamese trackers and obtains better performance on seven benchmarks with high real-time tracking speed.

Installation
--------------------------
Due to PreciseRoIPooling, **PLEASE USE THE COMMAND TO DOWNLOADE THE CODE:** ```git clone https://github.com/TJUMMG/TGSR.git```

You can use the following command to build your environment.

```bash
conda create -n verify python=3.7
conda activate verify
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt 
python setup.py build_ext --inplace
git clone https://github.com/vacancy/PreciseRoIPooling.git
```



Please refer to [PySOT_INSTALL.md](https://github.com/STVIR/pysot/blob/master/INSTALL.md) and [PreciseRoIPooling_README.md](https://github.com/vacancy/PreciseRoIPooling) to solve the installation problem.



## Result

raw result file in [Baidupan](https://pan.baidu.com/s/1lkb6tApeoGKJO4deVWQ8Qw), keyword: 9tu5

| Dataset | Evaluation | SiamRPN++ | SiamRPN++\_TG | SiamRPN++\_SR | SiamRPN++\_TGSR |
| :-----: | :--------: | :-------: | :-----------: | :-----------: | :-------------: |
| VOT2016 |    EAO     |   0.464   |     0.480     |     0.486     |      0.493      |
| VOT2018 |    EAO     |   0.415   |     0.435     |     0.422     |      0.440      |
| VOT2019 |    EAO     |   0.287   |     0.292     |     0.290     |      0.295      |
| OTB100  |    AUC     |   0.696   |     0.698     |     0.697     |      0.698      |
|         |    Pre     |   0.905   |     0.909     |     0.907     |      0.914      |
|   DTB   |    AUC     |   0.614   |     0.615     |     0.616     |      0.624      |
|         |    Pre     |   0.800   |     0.804     |     0.803     |      0.814      |
|  NFS30  |    AUC     |   0.507   |     0.509     |     0.518     |      0.520      |
|         |    Pre     |   0.598   |     0.600     |     0.612     |      0.614      |
|  LaSOT  |    AUC     |   0.497   |     0.502     |     0.498     |      0.502      |
|         |  NormPre   |   0.571   |     0.577     |     0.573     |      0.578      |
|         |    Pre     |   0.490   |     0.495     |     0.491     |      0.496      |

Usage
--------------------------

### Modify the path

1. modify the path in the python script (e.g., `./tools/test_SiamRPN++_VOT.py`)

   ```python
   sys.path.append('/media/HardDisk_new/wh/TGSR/')   # path to TGSR
   os.system("cd /media/HardDisk_new/wh/TGSR/tools/")	# path to current folder
   ```

2. modify the dataset path (e.g.,  `dataset_root` in`./tools/test_SiamRPN++_VOT.py`)

   ```python
   dataset_root = os.path.join('/media/HardDisk_new/DataSet/test/', args.dataset)  # path to your pysot dataset
   ```

   

### Test

1. Download models in [Baidupan](https://pan.baidu.com/s/1lkb6tApeoGKJO4deVWQ8Qw), keyword: 9tu5
   - `experiments.zip` : the model of SiamRPN++ and SiamMask, should be unzipped to `./experiments`

2. run the command

   ```python  
   python ./tools/test_SiamRPN++_VOT.py --dataset VOT2016
   ```

   

### Train
1. run the ```./pioneer/traj_predict_train.py``` to train TPN

2. run the `./pioneer/IoU_train.py`to train IPN

3. run the `./pioneer/Refine_train.py` to train BRN

   

### Eval ALTL

1. Download the pkl result in [Baidupan](https://pan.baidu.com/s/1lkb6tApeoGKJO4deVWQ8Qw), keyword: 9tu5

   - `snapshot_test.zip` : the model of TGSR, should be unzipped to `./snapshot_test`

2. run the command and get the  Average Longest Tracking Length (ALTL) of SiamRPN++_TGSR on the VOT2016

   ```
   python ./pioneer/research/eval_tool.p
   ```




Acknowledgments
------------------------------

1. [PySOT](https://github.com/STVIR/pysot)
2. [pytracking](https://github.com/visionml/pytracking)
3. [PreciseRoIPooling_README.md](https://github.com/vacancy/PreciseRoIPooling)
4. [DR_Loss](https://github.com/idstcv/DR_loss)
