From ubuntu:bionic-20200219
RUN mkdir -p /app

COPY . /app

WORKDIR /app

RUN mkdir -p /root/.keras/models && mv /app/data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

ENV TF_XLA_FLAGS --tf_xla_cpu_global_jit

RUN pip3 install -r /app/requirements.txt  -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com


CMD ["python3", "server.py"]
