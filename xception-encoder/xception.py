import os
import numpy as np
from numpy import linalg as LA
from keras.applications.xception import Xception as KerasXception
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.preprocessing import image
import base64
import cv2
import uuid
import urllib.request, urllib.error, urllib.parse


LOCAL_TMP_PATH = "./tmp"


class Xception:
    def __init__(self):
        self.input_shape = (299, 299, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model_xception = KerasXception(include_top=False, weights=self.weight, input_tensor=None,
                                            input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                            pooling=self.pooling)
        self.model_xception.predict(np.zeros((1, 299, 299, 3)))

    def extract_feature(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_xception(img)
        feat = self.model_xception.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        norm_feat = [i.item() for i in norm_feat]
        return norm_feat

    @property
    def name(self):
        return "xception"

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
        return "2048"

    @property
    def metric_type(self):
        return "L2"


def save_from_url(path, name, url):
    try:
        urllib.request.urlretrieve(url, path + name)
        return path + name
    except Exception:
        return ""


def save_from_base64(name, file_data=None):
    file_name = os.path.join(LOCAL_TMP_PATH, name)
    try:
        with open(file_name, "wb") as f:
            f.write(base64.decodebytes(file_data.encode('utf-8')))
        return file_name
    except Exception as e:
        # raise DecodeError("Decode string error", e)
        return ""


def run(images, urls):
    xception = Xception()
    vectors = []
    if images:
        for img in images:
            file_name = "{}-{}.{}".format("processor", uuid.uuid4().hex, "jpg")
            image_path = save_from_base64(file_name, img)
            vector = xception.extract_feature(image_path)
            vectors.append(vector)
    else:
        for url in urls:
            file_name = "{}-{}.{}".format("processor", uuid.uuid4().hex, "jpg")
            image_path = save_from_url(LOCAL_TMP_PATH, file_name, url)
            if image_path:
                vector = xception.extract_feature(image_path)
                vectors.append(vector)
    return vectors
