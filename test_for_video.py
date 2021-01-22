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
import win32ui
# import sys

# cv2.setNumThreads(16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(".\\goods"):
    os.mkdir("goods")

if torch.cuda.is_available():
    print("Using backend CUDA")
else:
    print("Using backend CPU")


def prepare_image(image):
    # if image.mode != 'RGB':
    #     image = image.convert("RGB")
    Transform = transforms.Compose([
        # transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    image = Transform(image.copy())
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

model = torchvision.models.resnet50()
# model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
model.fc = torch.nn.Linear(in_features=2048, out_features=1)
model.load_state_dict(torch.load(
    'model/model-resnet50.pth', map_location=device))
model.eval().to(device)

def main():

    

    # 用cv2读取视频
    path = r"C:\Users\zdwxx\Videos\2020-09-20 11-22-08.mp4"
    print("请选择一个视频：")
##    while True:
##        dlg = win32ui.CreateFileDialog(1)
##        if dlg.DoModal() == 2:
##            sys.exit()
##        path = dlg.GetPathName()
##        if os.path.exists(path):
##            break
##        else:
##            print("文件似乎不存在，请再次选择。")
    
##    path = input("请输入路径：")

    # path = r"C:\Users\zdwxx\Videos\2020-09-20 11-22-08.mp4"

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
            oframe = frame.copy()
            # frame = cv2.cuda_GpuMat(frame)
            # frame = cv2.cuda.resize(frame, (224,224),interpolation=cv2.INTER_LINEAR)
            # frame = frame.download()
            frame = cv2.resize(frame,(224,224),interpolation=cv2.INTER_LINEAR)
            frame_RGB = frame[: , : , ::-1]

            # frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # pframe = Image.fromarray(frame_RGB)
            
            score = predict(frame_RGB, model)
            fiscs.append([count, score])
    ##        print("frame:", count, "score:", score)
            if score >= best - 0.5 or score >= 4:
                if score > best:
                    best = score
                    save_path = "goods/"+path.split("\\")[-1]+"_"+str(count)+".png"
                    # cv2.imwrite(save_path, oframe,)
                    cv2.imencode('.png', oframe)[1].tofile(save_path)
                    # frame_RGB.save(save_path)
                cv2.imshow('img', oframe)
                cv2.waitKey(1)
                
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
    print("全部完成。")
##    input("请按回车键继续. . .")
    # sys.exit()

if __name__ == "__main__":
    try:
        import time
        b = time.time()
        main()
        a = time.time()
        print(a - b)
    except KeyboardInterrupt:
        # sys.exit()
        pass
    except Exception as e:
        print("错误")
        print(repr(e))
        input("请按回车键继续. . .")
        # sys.exit()
