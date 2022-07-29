import os
import sys
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="directory to check the validity of the dataset (e.g. ~/dataset/training/")
args = parser.parse_args()

rootpath = args.path

out = sys.stdout #open(1, "w")
sys.stderr = out
rootpath = os.path.expanduser(rootpath)
subfoldernames = sorted(os.listdir(rootpath))
for subfoldername in subfoldernames:
    out.write(f"going inside {subfoldername}\n")
    fullsubfoldername = os.path.join(rootpath, subfoldername)
    filenames = sorted(os.listdir(fullsubfoldername))
    for filename in filenames:
        out.write(f"processing {filename}\n")
        extension = os.path.splitext(filename)[1]
        if extension==".jpg":
            fullsourcename = os.path.join(fullsubfoldername, filename)
            with Image.open(fullsourcename) as test_img:
                try:
                    test_img.load()
                except IOError as e:
                    out.write(f"Bad image: {e}\n")

