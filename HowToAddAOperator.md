## 如何实现自定义的 Operator
是否有这些疑问：
- 内置的 Operator 阈值设置不符合我的需求， 怎么实现定制化的 Operator？
- 内置的 Operator 模型不符合我的精度要求，怎么实现定制化的 Operator？
- 我实现了一篇最新论文中的SOTA模型，怎么实现定制化的 Operator？
- 我对常用模型进行了特定领域的 fine-tuning，怎么实现定制化的 Operator？

如何实现定制化的 Operator？

### 实现定制化 Operator 必要步骤
以下列出的是实现定制化 Operator 的必要事项，可以实现这些步骤进行接入。
1. 准备模型。
2. 实现 rpc 目录下的 grpc 接口，以便正常接入 Search。
3. 编写必要的编译文件以便在绝大多数环境下正常运行。（推荐编写 makefile、 dockerfile）

### 示例：实现一个 VGG19 的 Operator
1. 复制 ```example-custom-operaotor``` 目录，并改名为 ```vgg19-encoder```。
    ``` bash
    cp -rf example-custom-operator vgg19-encoder
    ```

2. 在 ```data/prepare_model.sh``` 中实现准备模型的代码逻辑。
    1. 找到vgg19的notop模型地址：```https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5```
    2. 在 ```prepare_model.sh``` 中通过 wget 命令下载该模型，并进行相关判断。以下是一种实现方式：
        ```shell
        file=vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
        url=https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
        
        if [[ ! -f "${file}" ]]; then
          echo "[INFO] Model tar package does not exist, begin to download..."
          wget ${url}
          echo "[INFO] Model tar package download successfully!"
        fi
        
        if [[ -f "${file}" ]];then
          echo "[INFO] Model has been prepared successfully!"
          exit 0
        fi
        ```
    3. 运行该脚本确保可以正常下载该模型文件。
    4. 如果进行本地调试，执行以下步骤以便使 keras-application 可以导入该模型：
        ```bash
        mkdir -p ~/.keras/models
        cp data/*.h5 ~/.keras/models
        ```
       此步骤非必须。可以理解为将模型移动到机器学习框架默认导入位置。
       
3. 在 ```custom_operator.py``` 中定义关于 ```CustomOperator``` 的所有逻辑；并根据实际需要完善 requirements.txt 和 requirements-gpu.txt。
    1. 下面列出的是 vgg19 的一种实现方式, 主要参考了 Keras-Application 中 vgg19 的样例:
    （这部分代码后续会想办法折叠起来）
        ```python
        import os
        import uuid
        import logging
        import time
        import numpy as np
        import tensorflow as tf
        from keras.applications.vgg19 import VGG19
        from keras.preprocessing import image
        from keras.applications.vgg19 import preprocess_input
        import keras.backend.tensorflow_backend as KTF
        from numpy import linalg as LA
        from utils import save_tmp_file
        
        
        class CustomOperator:
           def __init__(self):
               self.model_init = False
               self.user_config = self.get_operator_config()
        
               self.graph = tf.Graph()
               with self.graph.as_default():
                   with tf.device(self.device_str):
                       self.session = tf.Session(config=self.user_config)
                       KTF.set_session(self.session)
                       self.model = VGG19(weights='imagenet', include_top=False, pooling='avg')
                       self.graph = KTF.get_graph()
                       self.session = KTF.get_session()
                       self.model.trainable = False
                       # warmup
                       self.model.predict(np.zeros((1, 224, 224, 3)))
               logging.info("Succeeded to warmup, Now grpc service is available.")
        
           def get_operator_config(self):
               try:
                   self.device_str = os.environ.get("device_id", "/cpu:0")
                   config = tf.ConfigProto(allow_soft_placement=True)
                   config.gpu_options.allow_growth = True
                   gpu_mem_limit = float(os.environ.get("gpu_mem_limit", 0.3))
                   config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_limit
                   # for device debug info print
                   if os.environ.get("log_device_placement", False):
                       self.user_config.log_device_placement = True
                   logging.info("device id %s, gpu memory limit: %f",
                                self.device_str, gpu_mem_limit)
        
               except Exception as e:
                   logging.error(
                       "unexpected error happen during read config",
                       exc_info=True)
                   raise e
               logging.info(
                   "Model device str: %s, session config: %s",
                   self.device_str, config)
               return config
        
           def execute(self, img_path):
               img = image.load_img(img_path, target_size=(224, 224))
               x = image.img_to_array(img)
               x = np.expand_dims(x, axis=0)
               x = preprocess_input(x)
               with self.graph.as_default():
                   with tf.device(self.device_str):
                       with self.session.as_default():
                           features = self.model.predict(x)
                           norm_feature = features[0] / LA.norm(features[0])
                           norm_feature = [i.item() for i in norm_feature]
                           return norm_feature
        
           def bulk_execute(self, img_paths):
               result = []
               for img_path in img_paths:
                   result.append(self.execute(img_path))
               return result
        
           def run(self, images, urls):
               result_images = []
               start = time.time()
               try:
                   if images:
                       for img in images:
                           file_name = "{}-{}".format("processor", uuid.uuid4().hex)
                           image_path = save_tmp_file(file_name, file_data=img)
                           if image_path:
                               result_images.append(self.execute(image_path))
                   else:
                       for url in urls:
                           file_name = "{}-{}".format("processor", uuid.uuid4().hex)
                           image_path = save_tmp_file(file_name, url=url)
                           if image_path:
                               result_images.append(self.execute(image_path))
               except Exception as e:
                   logging.error("something error: %s", str(e), exc_info=True)
                   pass
               end = time.time()
               logging.info('%s cost: {:.3f}s, get %d results'.format(end - start),
                            "custom processor", len(result_images))
               return result_images
        
           @property
          def name(self):
               return "vgg19"
        
           @property
          def type(self):
               return "encoder"
        
           @property
          def input(self):
               return "image"
        
           @property
          def output(self):
               return "vector"
        
           @property
          def dimension(self):
               return "512"
        
           @property
          def metric_type(self):
               return "L2"
        ```
    
        上述代码实现主要完成了以下工作:
        
        - 实现了 CustomOperator 中的几个 property
        - 实现了模型的初始化流程\gpu 设备支持
        - 实现了 run 方法接口.
    
    2. 添加程序依赖. 下面列出的是 requirements.txt, requirements-gpu.txt 可以参考添加.
        ```requirements.txt
        Keras
        tensorflow==1.14.0
        grpcio==1.27.2
        pillow
        ```
4. 调整```server.py```中对 customer_operator 的调用逻辑，使 grpc 服务返回正确的结果。
因为是encoder类型的,直接删除Execute方法中关于processor的模板代码.

    ```python
    def Execute(self, request, context):
       logging.info("execute")
       # encoder code which returns vectors
       grpc_vectors = []
       vectors = self.operator.run(request.datas, request.urls)
       for vector in vectors:
           v = rpc.rpc_pb2.Vector(element=vector)
           grpc_vectors.append(v)
       return rpc.rpc_pb2.ExecuteReply(nums=len(vectors),
                                       vectors=grpc_vectors,
                                       metadata=[])
    ```

5. 调整编译选项, 构建 docker 镜像, 并启动容器进行测试。
    以下是 cpu 版本的相关事项, gpu版本可以参照进行修改:
    
    1. 因为 keras application 会到指定位置读取模型, 需要在 Dockerfile 中添加相关逻辑, 以下是一种实现方式:
        ```dockerfile
        RUN apt-get update --fix-missing \
           && apt-get install -y python3 \
           python3-pip wget \
           libglib2.0-0 libsm6 \
           libxext6 libxrender1 \
           && apt-get clean \
           && rm -rf /var/lib/apt/lists/* \
           && cd /app/data \
           && ./prepare_model.sh \
           && cd - \
           && mkdir tmp \
           && mkdir -p /root/.keras/models && mv /app/data/*.h5 /root/.keras/models
        ```
    2. 运行 ```make cpu``` 命令构建 docker 镜像。
    3. 运行 ```make test-cpu``` 命令创建一个容器, 并测试容器暴露出的 grpc 服务.下面是一次成功的测试结果:
        ```bash
        $ make test-cpu                                   
        docker run -p 53001:53001 \
        -e OP_ENDPOINT=127.0.0.1:53001 -v `pwd`/tmp:/app/tmp \
        --name "custom-operator-cpu-test" -d zilliz/custom-operator:3be7e9e
        bbfc5a87e9e3d28be867be63fb092e14449129a1b03a052eb0b1988a7d692855
        echo "sleep 15s for waiting container to init and warmup" && sleep 15s
        sleep 15s for waiting container to init and warmup
        python3 ../test_grpc.sh.py -e 127.0.0.1:53001 || echo "[ERROR] test grpc failed"
        [*] Endpoint is  127.0.0.1:53001
        Begin to test: endpoint-127.0.0.1:53001
        Endpoint information:  {'name': 'vgg19', 'endpoint': '127.0.0.1:53001', 'type': 'encoder', 'input': 'image', 'output': 'vector', 'dimension': '512', 'metric_type': 'L2'}
        Endpoint health:  healthy
        Result :
         vector size: 1;  data size: 0
         vector dim:  512
        All tests over.
        docker rm -f "custom-operator-cpu-test"
        custom-operator-cpu-test
        ```
进阶自定义选项：
- 修改 ```server.py``` 中 ```ENDPOINT``` 默认端口. 推荐同时更改 ```Dockerfile``` 中的 Expose 的端口, 以求更完善的 docker 使用体验。
- 修改 ```Makefile``` 中的 IMAGE_NAME 以及 TAG 以便个性化定制 docker 镜像。
- 一切不影响运行的修改。

### todo: add more result pictures and code details pictures

