#!/usr/bin/python
from PIL import Image
import os, sys

path = "./data/train/compost"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((512,384), Image.ANTIALIAS)
            imResize.save('compost.jpg', 'JPEG', quality=90)

resize()