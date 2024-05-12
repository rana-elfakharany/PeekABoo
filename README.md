# Peek A Boo: Towards Smart Search and Rescue

**Presentation Link:**
[CV Presentation](https://www.canva.com/design/DAGFBrdaVhI/Bwyfl3nJT_WqUEDw_G7l5g/edit?utm_content=DAGFBrdaVhI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

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
[SeaDronesSee Dataset](https://universe.roboflow.com/ntnu-2wibj/seadronessee-odv2/dataset/11/images) used in the [WACV 2022 paper](https://openaccess.thecvf.com/content/WACV2022/html/Varga_SeaDronesSee_A_Maritime_Benchmark_for_Detecting_Humans_in_Open_Water_WACV_2022_paper.html). The choice behind this dataset is supported by a number of reasons:
- High number of samples available for rigorous training, validation, and testing phases
- High complexity due to the presence of a variety of objects
- Assurance of robustness of the model

## Models
| Model |Params<br/><sup> (M) | mAP<sup>test<br/>0.5:0.95 | mAP<sup>test<br/>0.5 | Precision | Recall | F1-Score | Inference Time |
| :------------------ | --------- | -------- | --------- | ---------- | -------- | --------- | -------- |
| [**YOLOv6-S6**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s6.pt) | 41.32 | 0 | 0 | 0 | 0 | 0 | 0 |
| [**YOLOv6-N6**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6n6.pt) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| [**YOLOv6-M6**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6m6.pt) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| [**YOLOv6-L6**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6l6.pt) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

[Our Checkpoints](https://drive.google.com/drive/folders/11luhl9lqkqAX4W7mA_q3b1aLuFCi3QOE?usp=sharing)
## Steps

For the steps of running the models, we followed the given [Github](https://github.com/meituan/YOLOv6/tree/0.3.0).

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
  
- After downloading the dataset, place it in the 'data' directory as YOLO format (.yaml)
- After choosing the model that you would like to run from the above table, download the .pt file and place it in the weights folder
- Navigate to 'tools' and open the 'train.py' file to change the needed parameters in the 'get_args_parser' function
- Navigate to the data.yaml folder and specify the correct location of the images
- Run the following command:
  
```shell
 python tools/train.py --batch 32 --conf configs/yolov6m6_finetune.py --data /localHome/cloudies/PeekABoo/SeaDronesSee-Yolov8/data.yaml --img 1280 --device 3 --epochs 50
```

If your training is interrupted, you can continue the training by running the following:
```shell
# single GPU training.
python tools/train.py --resume

# multi GPU training.
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --resume
```
Doing so would automatically find the latest checkpoint in YOLOv6 directory and resume training from there.

</details>

<details>
<summary> Evaluation</summary>

- Run the following command to evaluate the model:
```shell
python tools/eval.py --data ../SeaDronesSee-Yolov8/test.yaml --batch 32 --weights ./runs/train/LaS2n/weights/best_ckpt.pt --task val --reproduce_640_eval --img 1280 --name yolov6n6
```

</details>

<details>
<summary> Inference </summary>

- Use your trained model to do the inference
- Run inference using the 'infer.py' file found in the 'tools' folder

```shell
python tools/infer.py --weights yolov6s6.pt --img 1280 --source img.jpg / imgdir / video.mp4
```
</details>
