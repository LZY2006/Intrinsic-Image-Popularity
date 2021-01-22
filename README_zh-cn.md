# 估计图片本身的流行性

这是论文[估计图片本身的流行性](https://arxiv.org/abs/1907.01985)的PyTorch实现。

这项工作可以定量预测图片在Instagram上的吸引力。

它可以帮助用户找到最受公众欢迎的图像。

运行 ```python test.py --image_path <image_path>``` 以评估你的照片本身在Instagram上受欢迎程度。

这是 [在线演示](https://iipa.ngrok2.xiaomiqiu.cn) (不稳定).

### 数据集
我们以“简码”和图像对的形式提供“可区分人气的图像”的数据集。 您可以使用URL下载图像 ```"https://www.instagram.com/p/<shortcode>/media/?size=l"``` 。

*请注意，某些网址现在可能无效。*

### 引用
```
@inproceedings{ding2019intrinsic,
  title={Intrinsic Image Popularity Assessment},
  author={Ding, Keyan and Ma, Kede and Wang, Shiqi},
  booktitle={ACM International Conference on Multimedia},
  pages={1979--1987},
  year={2019},
  publisher={ACM}
}
```
