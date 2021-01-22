# -*- coding: utf-8 -*-
import os
from pprint import pprint

import numpy
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("Using backend CUDA")
else:
    print("Using backend CPU")


def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    image = Transform(image)
    image = image.unsqueeze(0)
    return image.to(device)


def predict(image, model=None):
    if model == None:
        model = torchvision.models.resnet50()
        # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
        model.fc = torch.nn.Linear(in_features=2048, out_features=1)
        model.load_state_dict(torch.load(
            'model/model-resnet50.pth', map_location=device))
        model.eval().to(device)
    image = prepare_image(image)
    with torch.no_grad():
        preds = model(image)
    return preds.item()


def arg(x):
    return x[1]


def main():
    model = torchvision.models.resnet50()
    # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
    model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    model.load_state_dict(torch.load(
        'model/model-resnet50.pth', map_location=device))
    model.eval().to(device)
    # image = Image.open("images/0.jpg")
    # x = predict(image)

    top = input("输入路径：")

    fpsc = []
    list_dir = []
    
    for dirpath, dirnames, filenames in os.walk(top):
        for name in filenames:
            filepath = os.path.join(dirpath, name)
            list_dir.append(filepath)
    
    print("正在计算……")
    for filepath in tqdm(list_dir):
        # print(filepath)
        if filepath.endswith(".jpg") or filepath.endswith(".png"):
            image = Image.open(filepath)
            prediction = predict(image, model)
            print("image:", filepath, "score:", prediction)
            fpsc.append([filepath, prediction])

    # fpsc = [['faces/00001_0.jpg', 1.0490654706954956], ['faces/00002_0.jpg', 1.0693581104278564], ['faces/00003_0.jpg', 1.1973750591278076], ['faces/00004_0.jpg', 1.1499651670455933], ['faces/00005_0.jpg', 1.085140347480774], ['faces/00006_0.jpg', 1.2823302745819092], ['faces/00007_0.jpg', 1.152990698814392],
    #         ['faces/00008_0.jpg', 1.1538634300231934], ['faces/00009_0.jpg', 1.1667460203170776], ['faces/00010_0.jpg', 1.066475749015808], ['faces/00011_0.jpg', 0.9527475237846375], ['faces/00012_0.jpg', 1.12521493434906], ['faces/00013_0.jpg', 2.0621581077575684], ['faces/00014_0.jpg', 1.76711106300354], ]
    fpsc.sort(key=arg, reverse=True)
    
    for i, each in enumerate(fpsc):
        print("index: %3d score: %.4f"%(i,each[1]))

        
    fpsc = [i[0] for i in fpsc]
    old2order = {file:i for i, file in enumerate(fpsc)}
    for dirpath, dirnames, filenames in os.walk(top):
        for name in filenames:
            filepath = os.path.join(dirpath, name)
            try:
               
                os.rename(filepath,
                          os.path.join(
                              os.path.split(filepath)[0] ,
                              str(old2order[filepath]) +
                              "."  + 
                              filepath.split(".")[-1]
                            )
                      )
            except KeyError:
                pass
            # except:
            #     pass
    # input("请按回车键继续. . .")


if __name__ == "__main__":
    try:
        main()
        input("请按回车键继续. . .")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("错误")
        print(repr(e))
        input("请按回车键继续. . .")
