# Omnisearch Operators

## Encoder
- Vgg16
- Xception
- Face-embedding
> Implemented by facenet model.
- SSD-tag-encoder
## Detector
- MTCNN-face-detector
- Mask-RCNN-object-detector
- SSD-object-detector
- YOLOv3-detector

## 快速开始 (docker pull)
PS: 待 docker hub 确定后上传拉取 
### CPU 版本
```bash
# 拉取对应版本的 docker 镜像, ${tag}应替换为可选的tag
docker pull zilliz/face-encoder:${tag}
# 以该镜像快速启动一个容器,同时设置容器配置:
# 1. 将容器的50004端口映射到本机
# 2. 将容器的 /app/tmp 目录映射到本机,以方便查看/调试 encoder 内部图片缓存
docker run -p 50004:50004 -v `pwd`/tmp:/app/tmp -d zilliz/face-encoder:${tag}
```
### GPU 版本

```bash
# 拉取对应版本的 docker 镜像, ${tag}应替换为可选的tag
docker pull zilliz/face-encoder-gpu:${tag}
# 以该镜像快速启动一个容器,同时设置容器配置:
# 1. 将容器的50004端口映射到本机
# 2. 将容器的 /app/tmp 目录映射到本机,以方便查看/调试 encoder 内部图片缓存
# 3. 开放容器对 GPU 的可见权限, 以只开放 device 0 的 GPU 为例
docker run --gpus="device=0" -e device_id="/device:GPU:0" \
    -p 50004:50004 -v `pwd`/tmp:/app/tmp -d zilliz/face-encoder-gpu:${tag}
```


## 快速开始 (docker build)
以 face embedding encoder 为例, 本节旨在3分钟之内以 docker build 的方式搭建一个最简单的 encoder 服务.

```bash
# 切换到工作目录
cd face-encoder
# 1. 准备模型, 以加速后续镜像构建, 比较耗时(可选)
cd data && ./prepare_model.sh && cd ..
# 2. 构建 docker 镜像
make cpu
# 3. 启动 docker 容器 
# (提示: ${tag} 为 刚构建的镜像 tag, 可通过 docker images 查看)
docker run -p 50004:50004 -v `pwd`/tmp:/app/tmp \
    -d zilliz/face-encoder:${tag}
```

## 快速开始 (source build)
以 face embedding encoder 为例, 本节旨在3分钟之内以 source build 的方式搭建一个最简单的 encoder 服务.

## 如何实现自己的 Operator
当遇到这种问题
- 内置的 Operator 阈值设置不符合我的需求?
- 内置的 Operator 不符合我的精度要求?

我想实现定制化的 Operator, 我该怎么做?

1. 复制 ```example-custom-operaotor``` 目录
2. 实现 ```data``` 目录下 ```prepare_model.sh``` 中关于准备自定义模型的代码逻辑
3. 在 ```custom_operator.py``` 中定义关于 ```CustomOperator``` 的所有逻辑
4. ```[Optional]``` : 修改 ```server.py``` 中 ```ENDPOINT``` 默认端口. 推荐同时更改 ```Dockerfile``` 中的 Expose 的端口, 以求更完善的 docker 使用体验.
5. ```[Optional]``` : 修改 ```Makefile``` 中的 IMAGE_NAME 以及 TAG 以便个性化定制 docker 镜像.
6. ```[Optional]``` : 一切不影响运行的修改.
7. ```[Recommeded]```: 测试 使用```custom_operator.py```中的```run``` 方法 调用 ```CuomstomOperator```.
8. ```[Recommeded]```: 使用 ```grpcurl``` 工具测试 ```server.py``` 中的 grpc服务. [如何使用grpc进行测试]()
9. 运行 ```make cpu``` 命令来构建 docker 镜像
10. 运行 ```docker run -p 52001:52001 -v `pwd`/tmp:/app/tmp 
    -d zilliz/custom-operator:${tag}``` 命令来启动 docker 镜像
11. ```[Recommeded]```: 重复 step 8 测试 docker 提供的 grpc 服务.
