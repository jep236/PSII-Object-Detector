import sys
import os
import argparse
import subprocess as sp
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from getpass import getpass
from PIL import Image
import numpy as np
import os
import random
import cv2
from pascal_voc_writer import Writer
from detecto.core import Dataset
from detecto.visualize import show_labeled_image
from detecto import core, utils, visualize
import os
import random 
import glob
import matplotlib.pyplot as plt
import cv2
from xml.dom import minidom#, parse
import numpy
import json
from torchvision import transforms
from detecto.utils import normalize_transform

def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  
    parser.add_argument('-j',
                        '--json',
                        help='JSON exported from labelbox labels',
                        type=str,
                        required=True)

    parser.add_argument('-e',
                        '--epoch',
                        help='Number of epochs to run',
                        type=int,
                        default=5)

    parser.add_argument('-lr',
                        '--learning_rate',
                        help='What the learning rate should be',
                        type=float,
                        default=.001)

    parser.add_argument('-b',
                        '--batch_size',
                        help='Decides the number of images processed before updating the model.',
                        type=int,
                        default=3)
    parser.add_argument('-th',
                        '--randH',
                        help='Decides the likehood of a random horizontal tranformation.',
                        type=float,
                        default=.5)
    parser.add_argument('-tc',
                        '--colorJit',
                        help='the rate the brightness, saturation, and contrast change when jitter is applied',
                        type=float,
                        default=.5)
    parser.add_argument('-tv',
                        '--randV',
                        help='Decides the likehood of a random vertical tranformation.',
                        type=float,
                        default=.5)

    return parser.parse_args()

def get_labels(json_path):
    labels = open(json_path);
    labels = json.loads(labels.read());
    return labels;

def download_set(work_path, set_list, img_dict):

  test_type = work_path.split('/')[-1]

  if not os.path.isdir(work_path):
    os.makedirs(work_path)
  else: 
    print(f'{test_type.capitalize()} set already exists.')

  for item in set_list: 
    url = img_dict.get(item)

    if not os.path.isfile(f'{os.path.join(os.getcwd(), work_path, item)}'):

      print(f'Downloading {item}.')
      sp.call(f'wget "{url}" -O {os.path.join(work_path, item)}', shell=True) 
#-------------------------------------------------------------------------------

def split_data(labels): 

  img_list = [item['Labeled Data'] for item in labels if item['Skipped']==False]
  name_list = [item['External ID'] for item in labels if item['Skipped']==False]
  id_list = [item['ID'] for item in labels if item['Skipped']==False]
  img_dict = dict(zip(name_list, img_list))
  label_dict = dict(zip(name_list, id_list))

  train, val, test = np.split(name_list, [int(.8*len(name_list)), int(.9*len(name_list))])
  return train, val, test, img_dict
    
def create_labels(data,test,train,val):
  

    for i in range(len(data)):
        try:
            file_name = data[i]['External ID'].replace('.png', '.txt')
            name = data[i]['External ID']
            out_name = name.replace('.png', '.xml')
            print(f'Creating {out_name}.')

            if name in test:
                file_type = 'test'

            elif name in train: 
                file_type = 'train'
          
            else:
                file_type = 'val'

            img = cv2.imread(os.path.join('lettuce_object_detection', file_type, name))

            h, w, _ = img.shape
            label_list, x, y = [], [], []
            for a in range(len(data[i]['Label']['objects'])):  
                points = data[i]['Label']['objects'][a]['bbox']
                label = data[i]['Label']['objects'][a]['value']
                label_list.append(label)
                x.append([points['left'], (points['left'] + points['width'])])
                y.append([points['top'], (points['top'] + points['height'])])
            final = list(zip(label_list, x, y))
            if not final:
                print('empty')
              
            name = os.path.join('lettuce_object_detection', file_type, name)
            writer = Writer(name, w, h)
            for item in final:

                min_x, max_x = item[1]
                min_y, max_y = item[2]
                writer.addObject(item[0], min_x, min_y, max_x, max_y)
            writer.save(os.path.join('lettuce_object_detection', file_type, out_name))
        except:
            pass

        print('Done creating labels.')

def main():
    args = get_args()
    labels = get_labels(args.json)
    train, val, test, img_dict = split_data(labels)
    inv_dict = {v: k for k, v in img_dict.items()}
    if(not(os.path.exists('lettuce_object_detection/train'))):
        download_set ('lettuce_object_detection/train', train, img_dict)
        download_set ('lettuce_object_detection/val', val, img_dict)
        download_set ('lettuce_object_detection/test', test, img_dict)
        create_labels(labels,test,train,val)
    fileName = ""
    transformList = [transforms.ToPILImage()]
    if(args.randH):
        transformList.append(transforms.RandomHorizontalFlip(p=args.randH))
        fileName += "randH" + str(args.randH)
    if(args.colorJit):
        transformList.append(transforms.ColorJitter(brightness=args.colorJit, contrast=args.colorJit, saturation=args.colorJit, hue=0.1))
        fileName += "ColorJit" + str(args.colorJit)
    if(args.randV):
        transformList.append(transforms.RandomHorizontalFlip(p=args.randV))
        fileName += "randV" + str(args.randV)
    transformList.append(transforms.ToTensor());
    t = transforms.Compose(transformList);
    if(fileName == ""):
        fileName = "NoTransforms"
    dataset = core.Dataset('lettuce_object_detection/train',transform= t)
    loader = core.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    fileName+= "epochs" + str(args.epoch) + "learn_rate" + str(args.learning_rate);
    val_dataset = core.Dataset('lettuce_object_detection/val')
    if(os.exist(fileName)):
        model = core.Model.load(fileName +"/"+fileName +".pth",['whole_plant', 'edge_plant'])
    else:
        model = core.Model(['whole_plant', 'edge_plant'])
    #model = core.Model(['plant'])
    losses = model.fit(loader, val_dataset, epochs=args.epoch, learning_rate=args.learning_rate, verbose=True)
    model.save(fileName);
    plt.plot(losses)
    plt.title('Faster R-CNN losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(fileName +"/"+fileName+'.png')
    plt.show()

    