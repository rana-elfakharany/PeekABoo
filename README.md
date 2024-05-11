# Peek A Boo: Towards Smart Search and Rescue

**Course:** COE 49413 - Computer Vision

**Instructor:** Dr. Omar Arif

**Term:** Spring 2024

**Group members:**
| **Name** | **ID** | 
| :----------------------------------------------------------- | ---- | 
| Mohammed Al Shafai | b00087311 |
| Rana Elfakharany | g00087725 |
| Hiba Saleem | g00087239 |


## Objective
The primary objective of this project is to execute object detection in marine environments through several computer vision techniques and detect the presence of humans in these environments. The purpose of identifying humans is to improve the surveillance and monitoring systems in tasks such as search and rescue missions, thereby ensuring maritime security.

## Dataset
The dataset to be used will be the 
[SeaDronesSee Dataset](https://openaccess.thecvf.com/content/WACV2022/html/Varga_SeaDronesSee_A_Maritime_Benchmark_for_Detecting_Humans_in_Open_Water_WACV_2022_paper.html). The choice behind this dataset is supported by a number of reasons:
- High number of samples available for rigorous training, validation, and testing phases
- High complexity due to the presence of a variety of objects
- Assurance of robustness of the model


## Models


| Model                                                        | Size | mAP<sup>val<br/>0.5:0.95 | Speed<sup>T4<br/>trt fp16 b1 <br/>(fps) | Speed<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | ---- | :----------------------- | --------------------------------------- | ---------------------------------------- | -------------------- | ------------------- |
| [**YOLOv6-N6**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6n6.pt) | 1280 | 44.9                     | 228                                     | 281                                      | 10.4                 | 49.8                |
| [**YOLOv6-S6**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s6.pt) | 1280 | 50.3                     | 98                                      | 108                                      | 41.4                 | 198.0               |
| [**YOLOv6-M6**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6m6.pt) | 1280 | 55.2                     | 47                                      | 55                                       | 79.6                 | 379.5               |
| [**YOLOv6-L6**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6l6.pt) | 1280 | 57.2                     | 26                                      | 29                                       | 140.4                | 673.4               |
<details>
<summary>Table Notes</summary>

- All checkpoints are trained with self-distillation except for YOLOv6-N6/S6 models trained to 300 epochs without distillation.  
- Results of the mAP and speed are evaluated on [COCO val2017](https://cocodataset.org/#download) dataset with the input resolution of 640×640 for P5 models and 1280x1280 for P6 models.
- Speed is tested with TensorRT 7.2 on T4.
- Refer to [Test speed](./docs/Test_speed.md) tutorial to reproduce the speed results of YOLOv6.
- Params and FLOPs of YOLOv6 are estimated on deployed models.
</details>


## Quick Start

<details>
<summary> Install</summary>


```shell
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
```
</details>



<details>
<summary> Training</summary>


Single GPU

```shell
# P5 models
python tools/train.py --batch 32 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --fuse_ab --device 0
# P6 models
python tools/train.py --batch 32 --conf configs/yolov6s6_finetune.py --data data/dataset.yaml --img 1280 --device 0
```

Multi GPUs (DDP mode recommended)

```shell
# P5 models
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --fuse_ab --device 0,1,2,3,4,5,6,7
# P6 models
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 128 --conf configs/yolov6s6_finetune.py --data data/dataset.yaml --img 1280 --device 0,1,2,3,4,5,6,7
```
- fuse_ab: add anchor-based auxiliary branch and use Anchor Unified Training Mode(Not supported on P6 models)
- conf: select config file to specify network/optimizer/hyperparameters. We recommend to apply yolov6n/s/m/l_finetune.py when training on your custom dataset.
- data: prepare [COCO](http://cocodataset.org) dataset, [YOLO format coco labels](https://github.com/meituan/YOLOv6/releases/download/0.1.0/coco2017labels.zip) and specify dataset paths in data.yaml
- make sure your dataset structure as follows:
```
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
│   ├── LICENSE
│   ├── README.txt
```


Reproduce our results on COCO ⭐️ [Train COCO Dataset](./docs/Train_coco_data.md)

<details>
<summary>Resume training</summary>

If your training process is corrupted, you can resume training by
```
# single GPU training.
python tools/train.py --resume

# multi GPU training.
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --resume
```
Above command will automatically find the latest checkpoint in YOLOv6 directory, then resume the training process. 

Your can also specify a checkpoint path to `--resume` parameter by
```
# remember to replace /path/to/your/checkpoint/path to the checkpoint path which you want to resume training.
--resume /path/to/your/checkpoint/path
```
This will resume from the specific checkpoint you provide.

</details>
</details>

<details>
<summary> Evaluation</summary>

Reproduce mAP on COCO val2017 dataset with 640×640 or 1280x1280 resolution ⭐️

```shell
# P5 models
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s.pt --task val --reproduce_640_eval
# P6 models
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s6.pt --task val --reproduce_640_eval --img 1280
```
- verbose: set True to print mAP of each classes.
- do_coco_metric: set True / False to enable / disable pycocotools evaluation method.
- do_pr_metric: set True / False to print or not to print the precision and recall metrics.
- config-file: specify a config file to define all the eval params, for example: [yolov6n_with_eval_params.py](configs/experiment/yolov6n_with_eval_params.py)
</details>


<details>
<summary>Inference</summary>

First, download a pretrained model from the YOLOv6 [release](https://github.com/meituan/YOLOv6/releases/tag/0.3.0) or use your trained model to do inference.

Second, run inference with `tools/infer.py`

```shell
# P5 models
python tools/infer.py --weights yolov6s.pt --source img.jpg / imgdir / video.mp4
# P6 models
python tools/infer.py --weights yolov6s6.pt --img 1280 --source img.jpg / imgdir / video.mp4
```
</details>

<details>
<summary> Deployment</summary>

*  [ONNX](./deploy/ONNX)
*  [OpenCV Python/C++](./deploy/ONNX/OpenCV)
*  [OpenVINO](./deploy/OpenVINO)
*  [TensorRT](./deploy/TensorRT)
</details>

<details open>
<summary> Tutorials</summary>

*  [Train COCO Dataset](./docs/Train_coco_data.md)
*  [Train custom data](./docs/Train_custom_data.md)
*  [Test speed](./docs/Test_speed.md)
*  [Tutorial of Quantization for YOLOv6](./docs/Tutorial%20of%20Quantization.md)
</details>

