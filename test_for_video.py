import os
from pprint import pprint

import cv2
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import time
import pickle
# from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


if __name__ == "__main__":

    model = torchvision.models.resnet50()
    # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
    model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    model.load_state_dict(torch.load(
        'model/model-resnet50.pth', map_location=device))
    model.eval().to(device)

    # 用cv2读取视频
##    path = r"C:\Users\zdwxx\Downloads\183883713_nb2-1-64.flv"
    path = input("请输入路径：")

    Video = cv2.VideoCapture(path)
    assert Video.isOpened()


    fiscs = []
    rval = True
    count = 0
    best = -999
    while rval:
        rval, frame = Video.read()
        if count % 10 != 0:
            count+=1
            continue
        try:
            frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pframe = Image.fromarray(frame_RGB)
            
            score = predict(pframe, model)
            fiscs.append([count, score])
    ##        print("frame:", count, "score:", score)
            if score >= best - 0.5 or score >= 4:
                if score > best:
                    best = score
                    save_path = "fucking_goods/"+path.split("\\")[-1]+"_"+str(count)+".png"
                    pframe.save(save_path)
                cv2.imshow('img', frame)
                cv2.waitKey(100)
                
                print("frame:", count, "score:", score)
                
        except Exception as exc:
            print(exc)
            count += 1
            continue
        count+=1

    fiscs.sort(key=arg, reverse=True)
    #pprint(fiscs)
    with open(path.split("\\")[-1]+"_fiscs.pkl", "wb") as f:
        pickle.dump(fiscs, f)
