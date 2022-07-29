# 제1회 NIA/AILAB 인공지능 경진대회 

행사명: AI-Hub 공개 데이터 활성화를 위한 공개데이터 활용 경진대회

목표: 동의보감에 포함된 약초 포함 128 종의 식물 이미지 데이터 분류

Link: [Kaggle](https://www.kaggle.com/competitions/gist-ailab-sample-competition)

+ Team name: random

Source code: [GitHub](https://github.com/jgs03177/gist-nia-competition)

## Environment

+ Anaconda
+ Python 3.9
+ TensorFlow-gpu 2.4.1
+ Keras 2.4.3

```
name: tfgpu
channels:
  - defaults
dependencies:
  - tensorflow-gpu
  - keras
  - scikit-learn
  - jupyter
  - notebook
  - matplotlib
  - tqdm
```

## Dataset

[How to download](./doc/README.md)

## Execution

- Validate downloaded image dataset (optional).

```bash
python verify.py /dataset/training
```

- Train AI model.

```
python train.py --model mobilenetv2
```

- If you stopped training (e.g. using ctrl+c), then convert checkpoint to model.

```
python chk2model.py --model mobilenetv2 --frm keras_best_weight.h5 --to model.h5
```

- Evaluate and create a csv file.

```
python evaluate.py --model model.h5 --data ~/dataset/validation
```

- Reformat csv for kaggle submission.

```
python tokaggle.py src.csv dst.csv
```

## Trained Model Parameters

+ densenet201 (pre-trained with imagenet) : `densenet201_pre.h5`

```
python train.py --model densenet201
python chk2model.py --model densenet201 --frm keras_best_weight.h5 --to densenet201_pre.h5
python evaluate.py --model densenet201_pre.h5 --data ~/dataset/validation
```

+ mobilenetv2 (without pre-training) : `mobilenet.h5`

```
python train.py --model mobilenetv2
python chk2model.py --model mobilenetv2 --frm keras_best_weight.h5 --to mobilenet.h5
python evaluate.py --model mobilenet.h5 --data ~/dataset/validation
```
