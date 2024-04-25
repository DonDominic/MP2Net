# MP^2^Net

The official PyTorch implementation of our **TGRS 2024** paper:

[**MP2Net: Mask Propagation and Motion Prediction Network for Multi-Object Tracking in Satellite Videos**](https://ieeexplore.ieee.org/abstract/document/10493056)



## Highlight

![](MP2Net.pdf)

### Brief Introduction

We propose a novel joint-detection-and-tracking framework, MP^2^Net, for multi-object tracking in satellite videos. MP^2^Net utilizes mask propagation and matching mechanism assist tiny target and local region detection and introduces implicit and explicit motion prediction strategies to facilitate the interaction of detection and tracking. Experimental results on two large-scale datasets demonstrate the effectiveness and robustness of MP^2^Net, achieving state-of-the-art performance on typical moving objects in satellite videos, such as 66.9% MOTA and 76.0% IDF1 on the SatVideoDT challenge dataset.

For more detailed information, please refer to the paper.



### Overall Performance[^1]

| Variant     | Dataset    | Category | MOTA  | IDF1  | IDP   | IDR   | Rcll  | Prcn  | IDs  | FP    | FN     | MT   | ML   |
| ----------- | ---------- | -------- | ----- | ----- | ----- | ----- | ----- | ----- | ---- | ----- | ------ | ---- | ---- |
| Ours_DLADCN | SatVideoDT | car      | 66.9% | 76.0% | 88.5% | 66.6% | 71.2% | 94.5% | 520  | 11536 | 80961  | 805  | 159  |
| Ours_ResFPN | SatVideoDT | car      | 64.7% | 73.6% | 88.8% | 62.9% | 67.9% | 95.7% | 512  | 8459  | 90183  | 759  | 184  |
| Ours_DLADCN | SatMTB     | car      | 60.3% | 71.8% | 88.0% | 60.6% | 64.7% | 94.0% | 986  | 14090 | 121319 | 1468 | 647  |
|             |            | airplane | 72.4% | 83.3% | 90.3% | 77.3% | 79.0% | 92.3% | 11   | 1028  | 3275   | 51   | 7    |
|             |            | ship     | 46.6% | 64.5% | 74.0% | 57.2% | 62.2% | 80.5% | 98   | 3281  | 8227   | 45   | 18   |

[^1]: Note that the results presented here differ slightly from those in the original article since we have made some optimizations to enhance our model's performance on cars. The variations in the results from the SatMTB dataset arise from using different training and test splits. We re-trained and tested our model using the publicly released version, and the results are summarized in the table. Rest assured, these new results do not affect the conclusions drawn in the original article.



## Release

**Trained Models** [[model weights](https://1drv.ms/f/c/178ed2ad13758ece/EvLv_kW9hEJAr1tWswLYshQBi8HtW3vprtwj2NeQVJ7vUw)] and **Raw Results** [[raw results](https://1drv.ms/f/c/178ed2ad13758ece/EvLv_kW9hEJAr1tWswLYshQBi8HtW3vprtwj2NeQVJ7vUw)] for both ICPR SatVideoDT dataset and SatMTB dataset.



## Getting Started

### Environment

Our experiments are conducted with *python3.6, pytorch1.7.0*, and *CUDA 10.2*.

You can follow [CenterNet](https://github.com/xingyizhou/CenterNet) to build the conda environment and use the latested version of [DCNv2](https://github.com/CharlesShang/DCNv2) under PyTorch 1.x.

Install evaluation requirements using  ```pip install motmetrics```

### Data Preparation

The ICPR SatVideoDT dataset used here is available in https://satvideodt.github.io. (An extensive version of VISO dataset, with more data and corrected annotations)

The SatMTB dataset is available in [BaiduYun](https://pan.baidu.com/s/1TBCnflx1M_Fk30xWcsDiqg?) with password **s5y7**, and the unzip password is **CSU@2023**.

Our collected json file is available in [json](https://1drv.ms/f/c/178ed2ad13758ece/EvLv_kW9hEJAr1tWswLYshQBi8HtW3vprtwj2NeQVJ7vUw)

```
-dataset
	|-ICPR
		|-train_data
			|-001
			|-002
			...
		|-val_data
			|-001
			|-002
			...
		|-annotations
			|-instances_train_caronly.json
			|-instances_val_caronly.json
	
	|-SatMTB
		|-train
			|-img
				|-airplane
					|xx
					...
				|-car
				|-ship
				|-train
			|-label
				|-airplane
					|xx.txt
					...
				|-car
				|-ship
				|-train
		|-test
			...(same with train)
		|-instances_SatMTB_train.json
		|-instances_SatMTB_test.json
```



### Training

#### Training for SatVideoDT.

```shell
python train.py --model_name DLADCN --gpus 0,1 --lr 1.25e-4 --lr_step 14 --num_epochs 15 --batch_size 4 --seqLen 5 --datasetname ICPR --data_dir  ./data/ICPR/
```

#### Training for SatMTB.

```shell
python train_satmtb.py --model_name DLADCN --gpus 0,1 --lr 1.25e-4 --lr_step 19 --num_epochs 20 --batch_size 4 --seqLen 5 --num_classes 3 --datasetname SatMTB --data_dir ./data/SatMTB/
```



### Testing

#### Testing for SatVideoDT.

```shell
python testDis.py --model_name DLADCN --gpus 0 --load_model ./checkpoints/MP2Net_ICPR_DLADCN.pth --datasetname ICPR --data_dir  ./data/ICPR/ 
```

#### Testing for SatMTB.

```shell
python testDis_satmtb.py --model_name DLADCN --gpus 0 --load_model ./checkpoints/MP2Net_SatMTB_DLADCN.pth --datasetname SatMTB --data_dir ./data/SatMTB/ --num_classes 3
```

#### (Optional) Testing with SORT.

```bash
python testTrackingSort.py --model_name DLADCN --gpus 0 --load_model ./checkpoints/MP2Net_ICPR_DLADCN.pth --datasetname ICPR --data_dir  ./data/ICPR/
```



### Evaluation

#### Evaluation for SatVideoDT.

```shell
python eval.py
```

#### Evaluation for SatMTB.

```shell
python eval_satmtb.py
```



## Acknowledgement

Our idea is built upon the following projects. We really appreciate their excellent open-source works!

- [DSFNet](https://github.com/ChaoXiao12/Moving-object-detection-DSFNet) [[related paper](https://ieeexplore.ieee.org/document/9594855)]
- [CenterNet](https://github.com/xingyizhou/CenterNet) [[related paper](https://arxiv.org/abs/1904.07850)]
- [FairMOT](https://github.com/ifzhang/FairMOT) [[related paper](https://arxiv.org/abs/2004.01888)]



## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@article{zhao2024mp2net,
  author={Zhao, Manqi and Li, Shengyang and Wang, Han and Yang, Jian and Sun, Yuhan and Gu, Yanfeng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MP2Net: Mask Propagation and Motion Prediction Network for Multiobject Tracking in Satellite Videos}, 
  year={2024},
  volume={62},
  pages={1-15},
  publisher={IEEE}
}
```



## Contact

If you have any questions or concerns, feel free to open issues or contact me through email [zhaomanqi19@csu.ac.cn].