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


## 快速开始 (docker pull)
PS: 待 docker hub 确定后上传拉取 （需要修改相关参数）

```bash
# 拉取对应版本的 docker 镜像, ${tag}应替换为可选的tag
docker pull zilliz/face-encoder:${tag}
# 以该镜像快速启动一个容器,同时设置容器配置:
# 1. 将容器的50004端口映射到本机
# 2. 将容器的 /app/tmp 目录映射到本机,以方便查看/调试 encoder 内部图片缓存
docker run -p 50004:50004 -v `pwd`/tmp:/app/tmp -d zilliz/face-encoder:${tag}
```
其他快速开始方式可参考[文件](./QuickStart.md)

## 如何实现自定义的 Operator
是否有这些疑问：
- 内置的 Operator 阈值设置不符合我的需求， 怎么实现定制化的 Operator？
- 内置的 Operator 模型不符合我的精度要求，怎么实现定制化的 Operator？
- 我实现了一篇最新论文中的SOTA模型，怎么实现定制化的 Operator？
- 我对常用模型进行了特定领域的 fine-tuning，怎么实现定制化的 Operator？

如何实现定制化的 Operator？
### 快速实现验证接入效果
1. 复制 ```example-custom-operaotor``` 目录。
2. 实现 ```data``` 目录下 ```prepare_model.sh``` 中关于准备自定义模型的代码逻辑。
3. 在 ```custom_operator.py``` 中定义关于 ```CustomOperator``` 的所有逻辑。
4. ```[Optional]``` : 修改 ```server.py``` 中 ```ENDPOINT``` 默认端口. 推荐同时更改 ```Dockerfile``` 中的 Expose 的端口, 以求更完善的 docker 使用体验。
5. ```[Optional]``` : 修改 ```Makefile``` 中的 IMAGE_NAME 以及 TAG 以便个性化定制 docker 镜像。
6. ```[Optional]``` : 一切不影响运行的修改。
7. ```[Recommeded]```: 测试 使用```custom_operator.py```中的```run``` 方法 调用 ```CuomstomOperator```。
8. ```[Recommeded]```: 使用 ```grpcurl``` 工具测试 ```server.py``` 中的 grpc服务。 [如何使用grpc进行测试]()
9. 运行 ```make cpu``` 命令来构建 docker 镜像。
10. 运行 ```docker run -p 52001:52001 -v `pwd`/tmp:/app/tmp
   -d zilliz/custom-operator:${tag}``` 命令来启动 docker 镜像。
11. ```[Recommeded]```: 重复 step 8 测试 docker 提供的 grpc 服务。

P.S. 以上所有流程都是为了能够快速实现而列出的事项。 如果对现有流程不满意，欢迎任何建设意见的 issue。

### 更大灵活度地实现定制化 Operator
以下列出的是实现定制化 Operator 的必要事项，可以实现这些步骤进行接入。
1. 准备模型。
2. 实现 rpc 目录下的 grpc 接口，以便正常接入 Search。
3. 编写必要的编译文件以便在多数环境下正常运行。（推荐编写 makefile、 dockerfile）
