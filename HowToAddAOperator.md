
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
3. 在 ```custom_operator.py``` 中定义关于 ```CustomOperator``` 的所有逻辑；并根据实际需要完善 requirements.txt 和 requirements-gpu.txt。

4. ```[Recommeded]```: 测试 使用```custom_operator.py```中的```run``` 方法 调用 ```CuomstomOperator```。
5. ```[Optional]``` : 修改 ```server.py``` 中 ```ENDPOINT``` 默认端口. 推荐同时更改 ```Dockerfile``` 中的 Expose 的端口, 以求更完善的 docker 使用体验。
6. ```[Optional]``` : 修改 ```Makefile``` 中的 IMAGE_NAME 以及 TAG 以便个性化定制 docker 镜像。
7. ```[Optional]``` : 一切不影响运行的修改。
8. ```[Recommeded]```: 使用 ```grpcurl``` 工具测试 ```server.py``` 中的 grpc服务。 [如何使用grpc进行测试]()
9. 运行 ```make cpu``` 命令来构建 docker 镜像。
10. 运行 ```docker run -p 52001:52001 -v `pwd`/tmp:/app/tmp
   -d zilliz/custom-operator:${tag}``` 命令来启动 docker 镜像。
11. ```[Recommeded]```: 重复 step 8 测试 docker 提供的 grpc 服务。

P.S. 以上所有流程都是为了能够快速实现而列出的事项。 如果对现有流程不满意，欢迎任何建设意见的 issue。

#### 举例：如何添加 VGG19 的 Opertaor
使用 Keras-Application 中内置的 VGG19 应用来实现一个 Operator。
1. 复制 ```example-custom-operaotor``` 目录，并改名为 ```vgg19-encoder```。
2. 实现 ```data``` 目录下 ```prepare_model.sh``` 中关于准备自定义模型的代码逻辑。
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
    
3. 在 ```custom_operator.py``` 中定义关于 ```CustomOperator``` 的所有逻辑。
    具体逻辑参考了 [keras 官网教程](https://keras.io/zh/applications/#vgg19)
    1. 在python文件添加必要的 import 逻辑
        ```python
        from keras.applications.vgg19 import VGG19
        from keras.preprocessing import image
        from keras.applications.vgg19 import preprocess_input
        import keras.backend.tensorflow_backend as KTF
        ```
    
    2. 在 ```__init__``` 函数中warmup逻辑之前、初始化逻辑之后添加 VGG19的模型初始化逻辑
        ```python
        self.model = VGG19(weights='imagenet', include_top=False)
        ```
    3. 实现 execute 和 bulk_execute 方法。
        ```python
        def execute(self, img_path):
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            with self.graph.as_default():
            with tf.device(self.device_str):
                with self.session.as_default():
                    features = self.model.predict(img)
                    return features
    
        def bulk_execute(self, img_paths):
            result = []
            for img_path in img_paths:
                result.append(self.execute(img_path))
            return result
        ```
    4. 确保 run 方法正确调用执行方法。此例中不难发现，传入执行方法的应该是图片地址而非 cv2 读取后的 numpy 格式，故删除该逻辑，以下是调整后的代码：
        ```python
        def run(processor, images, urls):
            result_images = []
            start = time.time()
            try:
                if images:
                    for img in images:
                        file_name = "{}-{}".format("processor", uuid.uuid4().hex)
                        image_path = save_tmp_file(file_name, file_data=img)
                        if image_path:
                            result_images.append(processor.execute(image_path))
                else:
                    for url in urls:
                        file_name = "{}-{}".format("processor", uuid.uuid4().hex)
                        image_path = save_tmp_file(file_name, url=url)
                        if image_path:
                            result_images.append(processor.execute(image_path))
            except Exception as e:
                logging.error("something error: %s", str(e), exc_info=True)
                pass
            end = time.time()
            logging.info('%s cost: {:.3f}s, get %d results'.format(end - start),
                         "vgg19 encoder", len(result_images))
            return result_images
         ```
         
    5. 与此同时，注意到 warmup 逻辑也不应该传入 ndarray，为了方便，可以注释该逻辑，或构造相关 warmup 逻辑.
        ```python
        self.execute('./test.jpg') # notice here need exist a pic named 'test.jpg' in the same folder

        # or 
        self.model.predict(np.zeros((1, 224, 224, 3))) # 224 x 224 is the input shape of vgg19.
        ```

    6. 实现 CustomOperator 中的几个property，以下是一种定义方式。
        ```python
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
       
4. ```[Recommeded]```: 测试 使用```custom_operator.py```中的```run``` 方法 调用 ```CuomstomOperator```。
    1. 新建一个test_run.py文件,并编写运行run方法的逻辑。以下是一种实现方式：
        ```python
        from custom_operator import run, CustomOperator

        if __name__ == "__main__":
        test_url = 'http://a3.att.hudong.com/14/75/01300000164186121366756803686.jpg'
        operator = CustomOperator()
        res = run(operator, None, [test_url])
        print(res)
        ```
    2. 如果输出结果为一组特征向量，则表明模型逻辑基本上没有错误了。

5. ```[Optional]``` : 修改 ```server.py``` 中 ```ENDPOINT``` 默认端口. 推荐同时更改 ```Dockerfile``` 中的 Expose 的端口, 以求更完善的 docker 使用体验。
    
    这里将端口信息都更改为51009
6. ```[Optional]``` : 修改 ```Makefile``` 中的 IMAGE_NAME 以及 TAG 以便个性化定制 docker 镜像。
    
    这里将IMAGE_NAME 设为 vgg19-encoder
7. ```[Optional]``` : 一切不影响运行正确性的修改。

8. ```[Recommeded]```: 使用 ```grpcurl``` 工具或者编写 grpc 客户端代码测试 ```server.py``` 中的 grpc 服务。 具体方式可参考[如何对 grpc 服务进行测试]()。

9. 运行 ```make cpu``` 命令来构建 docker 镜像。

10. 运行 ```docker run -p 51009:51009 -v `pwd`/tmp:/app/tmp -d zilliz/vgg19-encoder:${tag}``` 命令来启动 docker 镜像。

11. ```[Recommeded]```: 重复 step 8 测试 docker 提供的 grpc 服务。
 

### 更大灵活度地实现定制化 Operator
以下列出的是实现定制化 Operator 的必要事项，可以实现这些步骤进行接入。
1. 准备模型。
2. 实现 rpc 目录下的 grpc 接口，以便正常接入 Search。
3. 编写必要的编译文件以便在多数环境下正常运行。（推荐编写 makefile、 dockerfile）
