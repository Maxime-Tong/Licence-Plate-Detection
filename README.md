# Licence-Plate-Detection
This repo shows how to detect licence plate with opencv, we use tranditional method + dp method to recognite the plate.
## trainning PNet
```shell
python train.py
```
## Detection
```shell
python main.py
```
## PNet网络结构
| model   |      hyper-parameters      |
|----------|:-------------:|
| 输入图像 |  20 x 20 |
| conv | in channels=1, out channels=8, kernel size = 5, padding = 2    | 
| BatchNorm | num features = 8 |
| maxPooling | kernel size=2, stride=2 |
| conv | in channels=8, out channels=16, kernel size = 5, padding = 2  | 
| BatchNorm | num features = 8 |
| maxPooling | kernel size=2, stride=2 |
| Linear | in features=5*5*16, out features=128|
| Dropout | p=0.5 |
| Linear | in features=128, out features=64|
| Dropout | p=0.5 |
| Linear | in features=64, out features=N_classes|
|Softmax|
