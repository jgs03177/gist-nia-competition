from keras.applications import DenseNet169
from keras.applications import DenseNet201
from keras.applications import InceptionV3
from keras.applications import MobileNetV2
from keras.applications import ResNet50
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,  help="model name (e.g. densenet201)")
parser.add_argument("--frm", type=str, required=True, help="target checkpoint file (e.g. keras_best_weight.h5")
parser.add_argument("--to", type=str, required=True, help="output model file (e.g. model.h5)")
args = parser.parse_args()

input_tensor = Input(shape=(224, 224, 3))

# modelname = "densenet201"
modelname = args.model
model_path = args.frm
model_out = args.to

print(f"MODEL: {modelname}")

if modelname == "resnet50":
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
elif modelname == "densenet169":
    base_model = DenseNet169(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
elif modelname == "densenet201":
    base_model = DenseNet201(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
elif modelname == "mobilenetv2":
    base_model = MobileNetV2(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
elif modelname == "inceptionv3":
    base_model = InceptionV3(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
else:
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
x = base_model.output
x = Dense(512, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(128, activation='softmax')(x)
model = Model(base_model.input, outputs=x, name=modelname)
for layer in model.layers:
    layer.trainable = True
model.load_weights(model_path)
model.save(model_out)