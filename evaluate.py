# This python script is based on the source code in AI-Hub:
# https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=151

import os
import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import DenseNet201
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
import csv
from PIL import Image
from tqdm import tqdm
import argparse


def get_fullfilepathlist(rootpath):
    fli, li = [], []
    rootpath = os.path.expanduser(rootpath)
    subfoldernames = sorted(os.listdir(rootpath))
    for subfoldername in subfoldernames:
        print(f"going inside {subfoldername}...\n")
        fullsubfoldername = os.path.join(rootpath, subfoldername)
        filenames = sorted(os.listdir(fullsubfoldername))
        for filename in filenames:
            extension = os.path.splitext(filename)[1]
            if extension==".jpg":
                fullsourcename = os.path.join(fullsubfoldername, filename)
                print(f"{filename} added.\n")
                li += [filename]
                fli += [fullsourcename]
    return fli, li


def manual_evaluator(model_path, class_labels, img_lists, dst_path, batch_size=16):
    print("Running model inference, it might take more than 1 hour")
    model = load_model(model_path, compile=False)

    full_img_list, img_list = img_lists

    y_pred = list()
    with open(f'{dst_path}/predictions.csv', 'w', encoding='UTF-8') as file:
        writer = csv.writer(file)
        writer.writerow(["filepath", "prediction"])
        nprocessed = 0
        i = 0
        with tqdm(total=len(img_list)) as pbar:
            while nprocessed < len(img_list):
                current_batch = min(batch_size, len(img_list)-nprocessed)
                imgs = []
                for i in range(nprocessed, nprocessed+current_batch):
                    img = img_to_array(load_img(full_img_list[i], target_size=(224, 224))) / 255.
                    imgs += [img]
                x = np.stack(imgs, axis=0)
                predictions = model.predict(x)
                predictions = predictions.argmax(axis=1)
                for i in range(nprocessed, nprocessed+current_batch):
                    y_pred.append(class_labels[predictions[i-nprocessed]])
                    writer.writerow([img_list[i], class_labels[predictions[i-nprocessed]]])
                nprocessed += current_batch
                pbar.update(current_batch)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="directory of the dataset (e.g ~/dataset/validation)")
    parser.add_argument("--model", type=str, required=True, help="path of the model (e.g /model/model.h5)")
    args = parser.parse_args()

    rootpath = args.data
    model_path = args.model
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    img_list = get_fullfilepathlist(rootpath)

    class_labels = ['1', '2', '3', '4', '5', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', 
                    '19', '20', '21', '22', '23', '26', '27', '28', '29', '30', '31', '32', '33', '34', '36', 
                    '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', 
                    '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', 
                    '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', 
                    '82', '83', '85', '86', '87', '88', '89', '90', '91', '92', '93', '95', '96', '97', '98', 
                    '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', 
                    '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', 
                    '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136']

    dst_path = os.path.join(os.getcwd(), "tta_report")
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    manual_evaluator(model_path, class_labels, img_list, dst_path)
