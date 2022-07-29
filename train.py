# This python script is based on the source code in AI-Hub:
# https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=151

from __future__ import print_function

import argparse
import csv
import os

import numpy as np
from keras import backend as K
from keras.applications import DenseNet169
from keras.applications import DenseNet201
from keras.applications import InceptionV3
from keras.applications import MobileNetV2
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from learning_rate import choose

# pre-parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="enter imagenet pretrained model name")
parser.add_argument("--epoch", type=int, default=100, help="enter epoch number")
args = parser.parse_args()

img_height, img_width = 224, 224

if K.image_data_format() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

input_tensor = Input(shape=(img_width, img_height, 3))

batch_size = 16
epochs = args.epoch

train_data_dir = os.path.join(os.getcwd(), 'dataset/training')
validation_data_dir = os.path.join(os.getcwd(), 'dataset/validation')

# 128 class labels
labels = ['칡', '등칡', '만주감초', '애기나팔꽃', '유럽감초', '강황', '나팔꽃', '메꽃', '둥근잎나팔꽃', '둥근잎미국나팔꽃', '분꽃', '결명자',
          '고본', '당근', '회향', '배초향', '방아풀', '산박하', '구기자나무', '부추', '파(실파)', '금불초', '망강남(석결명)', '도라지', '백도라지', '층층잔대', '더덕',
          '미국자리공', '참당귀', '왜당귀', '바디나물', '엉겅퀴', '바늘엉겅퀴', '지느러미엉겅퀴', '고려엉겅퀴', '독활', '두릅나무', '쇠비름', '비름', '맥문동',
          '개맥문동', '소엽맥문동', '모란', '작약', '백작약', '으름덩굴', '통탈목', '지리강활(개당귀)', '멀꿀', '반하', '대반하', '은조롱', '이엽우피소', '하수오', '박주가리',
          '구릿대', '궁궁이', '삽주', '당백출(큰꽃삽주)', '미치광이풀', '산수유나무', '산딸나무', '생강나무', '자리공', '석창포', '창포', '꽃창포', '노랑꽃창포', '갯기름나물',
          '갯방풍', '중국방풍', '약모밀', '삼백초', '오미자', '흑오미자', '남오미자', '우엉', '쇠무릎', '파리풀', '환삼덩굴', '호프', '율무', '염주', '익모초',
          '인동덩굴', '들깨', '참깨', '지황', '곰보배추', '질경이', '갯질경이', '왕질경이', '도꼬마리', '큰도꼬마리', '천궁', '토천궁', '치자', '질경이택사', '택사',
          '소귀나물', '벗풀', '민들레', '흰민들레', '서양민들레', '붉은씨서양민들레', '부들', '애기부들', '꼬마부들', '큰잎부들', '향유', '꽃향유', '좀향유',
          '가는잎향유', '현삼', '큰개현삼', '토현삼', '섬현삼', '호도나무', '가래나무', '속새', '쇠뜨기', '황금', '골무꽃', '황기', '고삼', '사철쑥', '더위지기',
          '차즈기']

num_classes = 128
nb_train_samples = 529043
nb_validation_samples = 66079
nb_test_samples = 66247

print(f"MODEL: {args.model}")

if args.model == "resnet50":
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
elif args.model == "densenet169":
    base_model = DenseNet169(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
elif args.model == "densenet201":
    base_model = DenseNet201(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
elif args.model == "mobilenetv2":
    base_model = MobileNetV2(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
elif args.model == "inceptionv3":
    base_model = InceptionV3(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
else:
    base_model = MobileNetV2(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')

x = base_model.output
x = Dense(512, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(base_model.input, outputs=x, name=f'{args.model}')
for layer in model.layers:
    layer.trainable = True

optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)
# test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# save best model weights based on validation accuracy
save_dir = os.path.join(os.getcwd(), f'model/best_weights')
weights_name = 'keras_best_weights.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, weights_name)

# set learning rate schedule
lr_monitorable = True
lr_reduce = choose(lr_monitorable=lr_monitorable)

checkpoint = ModelCheckpoint(filepath=save_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode=max,
                             save_weights_only=True,
                             save_freq='epoch')
# early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)

# set callbacks for model fit
callbacks = [lr_reduce, checkpoint, early_stopping]

# model fit
hist = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks)

model.save(f"model/dongam_{args.model}.h5")
