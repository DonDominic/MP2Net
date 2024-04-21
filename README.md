# MP^2^Net

The official PyTorch implementation of our **TGRS 2024** paper:

[**MP2Net: Mask Propagation and Motion Prediction Network for Multi-Object Tracking in Satellite Videos**](https://ieeexplore.ieee.org/abstract/document/10493056)



## Brief Introduction

We propose a novel joint-detection-and-tracking framework, MP^2^Net, for multi-object tracking in satellite videos. MP^2^Net utilizes mask propagation and matching mechanism assist tiny target and local region detection and introduces implicit and explicit motion prediction strategies to facilitate the interaction of detection and tracking. Experimental results on two large-scale datasets demonstrate the effectiveness and robustness of MP^2^Net, achieving state-of-the-art performance on typical moving objects in satellite videos, such as 66.7% MOTA and 75.9% IDF1 on the SatVideoDT challenge dataset.

For more detailed information, please refer to the paper.



## Release

**Trained Models** (model weights for ICPR SatVideoDT dataset) [[model weights](https://1drv.ms/f/c/178ed2ad13758ece/EvLv_kW9hEJAr1tWswLYshQBi8HtW3vprtwj2NeQVJ7vUw)]



## Getting Started

-   ### Environment

    Our experiments are conducted with *python3.6, pytorch1.7.0*, and *CUDA 10.2*.

    You can follow [CenterNet](https://github.com/xingyizhou/CenterNet) to build the conda environment and use the latested version of [DCNv2](https://github.com/CharlesShang/DCNv2) under PyTorch 1.x.

    Install evaluation requirements using  ```pip install motmetrics```

-   ### Data Preparation

    The ICPR SatVideoDT dataset used here is available in https://satvideodt.github.io.

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




-   ### Training

    ```shell
    python train.py --model_name DLADCN --gpus 0,1 --lr 1.25e-4 --lr_step 14 --num_epochs 15 --batch_size 4 --seqLen 5 --datasetname ICPR --data_dir  ./data/ICPR/
    ```

    

-   ### Testing

    ```shell
    python testDis.py --model_name DLADCN --gpus 0 --load_model ./checkpoints/MP2Net_DLADCN.pth --datasetname ICPR --data_dir  ./data/ICPR/ 
    ```

    #### (Optional) Testing with SORT.

    ```bash
    python testTrackingSort.py --model_name DLADCN --gpus 0 --load_model ./checkpoints/MP2Net_DLADCN.pth --datasetname ICPR --data_dir  ./data/ICPR/
    ```

    

-   ### Evaluation

    ```shell
    python eval.py
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