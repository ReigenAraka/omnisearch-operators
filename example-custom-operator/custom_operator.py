import os
import uuid
import base64
import logging
import urllib.request
import time
import numpy as np
import tensorflow as tf


def temp_directory():
    return os.path.abspath(os.path.join('.', 'data'))


COCO_MODEL_PATH = os.path.join(temp_directory(), "yolov3_darknet")
LOCAL_TMP_PATH = "./tmp/"


class CustomOperator:
    def __init__(self):
        self.model_init = False
        self.user_config = self.get_operator_config()
        self.model_path = COCO_MODEL_PATH
        # warmup
        self.execute(np.zeros((300, 300, 3), dtype='float32'))

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

    def execute(self, image):
        pass

    def bulk_execute(self, images):
        pass

    @property
    def name(self):
        raise NotImplementedError("Must define name")

    @property
    def type(self):
        raise NotImplementedError("Must define type")

    @property
    def input(self):
        raise NotImplementedError("Must define input")

    @property
    def output(self):
        raise NotImplementedError("Must define output")

    @property
    def dimension(self):
        raise NotImplementedError("Must define dimension")

    @property
    def metric_type(self):
        raise NotImplementedError("Must define metric_type")


def save_tmp_file(name, file_data=None, url=None):
    start = time.time()
    extension = 'jpg'
    file_path = os.path.join(LOCAL_TMP_PATH, name + '.' + extension)
    if file_data:
        img_data = file_data.split(",")
        if len(img_data) == 2:
            posting = img_data[0]
            data_type = posting.split("/")[1]
            extension = data_type.split(";")[0]
            encode_method = data_type.split(";")[1]
            if encode_method != "base64":
                logging.error("Encode method not base64")
                raise
            imgstring = img_data[1]
        else:
            imgstring = img_data[0]
        file_path = os.path.join(LOCAL_TMP_PATH, name + '.' + extension)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(imgstring))
    if url:
        try:
            urllib.request.urlretrieve(url, file_path)
        except Exception as e:
            logging.error("Download file from url error : %s", str(e), exc_info=True)
            raise
    end = time.time()
    logging.info('  save_tmp_file cost: {:.3f}s'.format(end - start))
    return file_path


def run(processor, images, urls):
    result_images = []
    start = time.time()
    try:
        if images:
            for img in images:
                file_name = "{}-{}".format("processor", uuid.uuid4().hex)
                image_path = save_tmp_file(file_name, file_data=img)
                if image_path:
                    image = cv2.imread(image_path)
                    result_images.append(processor.execute(image))
        else:
            for url in urls:
                file_name = "{}-{}".format("processor", uuid.uuid4().hex)
                image_path = save_tmp_file(file_name, url=url)
                if image_path:
                    image = cv2.imread(image_path)
                    result_images.append(processor.execute(image))
    except Exception as e:
        logging.error("something error: %s", str(e), exc_info=True)
        pass
    end = time.time()
    logging.info('%s cost: {:.3f}s, get %d results'.format(end - start),
                 "custom processor", len(result_images))
    return result_images
