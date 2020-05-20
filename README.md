# Omnisearch Operators
[TOC]

## 什么是 Operator

概念介绍 （可跳转至 search 仓库的概念介绍页）

术语Operator是指在Omnisearch的一次查询中，将图片转换成特征向量的过程。Operator一般基于一个训练好的模型进行推理。Omnisearch的Operator当前支持使用TensorFlow/Pytorch/caffe的模型进行推理得到特征向量。

Omnisearch当前内置了一些Operator可以完成最简单的使用。如果你想创建一个Omnisearch未支持的流程，你可以尝试自己编写一个Operator，并且用来组合出自己的search流程。

为了对输入和输出进行区分，我们将Operator再分成了两类，分别是Encoder和Detector。Encoder类Operator接收图片作为输入，经过内部的模型得到能表示图片特征的特征向量作为输出。Detector类Operator接收图片作为输入，经过内部的模型推理，识别出图片上的所有对象，输出一组物体的图片。

我们在Omnisearch中内置了以下Operator，按encoder和detector分类。

## Encoder

- Vgg16
  - 镜像名： vgg16-encoder
  - 向量维度： 512
  - 计算方式： 需要测试
  - 使用场景：
  - 功能： 对输入的图片进行 embedding，得到表征图片的特征向量

> 以 Keras Application 中 Vgg16 实现该 encoder。
- Xception
  - 镜像名：xception-encoder
  - 向量维度： 2048
  - 计算方式： 需要测试
  - 使用场景：
  - 功能： 对输入的图片进行 embedding，得到表征图片的特征向量

> 以 Keras Application 中 Xception 实现该 encoder。

- Face-encoder
  - 镜像名：face-encoder
  - 向量维度： 128
  - 计算方式： 需要测试
  - 使用场景：
  - 功能： 对识别出来的人脸图片进行 embedding，得到表征人脸特征的向量
  
> Implemented by facenet model. 附上链接

- SSD-encoder
  - 镜像名： ssd-encoder
  - 标签：MSCOCO 的90种类别
  - 计算方式： 结构化数据
  - 使用场景：
  - 功能： 对输出的图片进行物体检测，得到表征图片中的物品信息的标签。
> 附上 github 链接

## Detector
- MTCNN-face-detector
  - 镜像名： face-detector
  - 功能： 识别输入图片中的人脸
  - 接受： image
  - 返回： 识别出的一组人脸图片
  - 样例 pipeline：mtcnn_detect_face -> face_embedding

> 以 facenet 实现，使用 github 项目 https://github.com/davidsandberg/facenet.git

- Mask-RCNN-object-detector
  - 镜像名： mask-rcnn-detector
  - 功能： 识别输入图片中的物体
  - 接受： image
  - 返回： 识别出的一组物体图片
  - 样例 pipeline：mask_rcnn -> vgg/xception

> 附上 mask-rcnn github 链接

- SSD-object-detector
  - 镜像名： ssd-detector
  - 功能： 识别输入图片中的物体
  - 接受： image
  - 返回： 识别出的一组物体图片
  - 样例pipeline：ssd -> vgg/xception

>

- YOLOv3-object-detector
  - 镜像名：yolov3-detector
  - 功能： 识别输入图片中的物体
  - 接受： image
  - 返回： 识别出的一组物体图片
  - 样例 pipeline：yolo -> vgg/xception

> 以 paddlepaddle yolo v3 模型实现，附链接


## 快速开始 
PS: 待 docker hub 确定后上传拉取 （需要修改相关参数）

```bash
# 拉取对应版本的 docker 镜像, ${tag}应替换为可选的tag
docker pull zilliz/face-encoder:${tag}
# 以该镜像快速启动一个容器,同时设置容器配置:
# 1. 将容器的50004端口映射到本机
# 2. 将容器的 /app/tmp 目录映射到本机,以方便查看/调试 encoder 内部图片缓存
docker run -p 50004:50004 -v `pwd`/tmp:/app/tmp -d zilliz/face-encoder:${tag}
```
更多的,更详细的方式可参考[快速开始](./QuickStart.md)

## 如何实现自定义的 Operator
### 灵活地实现定制化 Operator
以下列出的是实现定制化 Operator 的必要事项，可以实现这些步骤进行接入。
1. 准备模型。
2. 实现 rpc 目录下的 grpc 接口，以便正常接入 Search。
3. 编写必要的编译文件以便在多数环境下正常运行。（推荐编写 makefile、 dockerfile）

P.S. 更详细的定制化 Operator 以及 快速实现的样例可参考[如何添加自定义 Operator ](./HowToAddAOperator.md)