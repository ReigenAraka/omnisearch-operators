import os
import uuid
import base64
import logging
import time
import urllib.request
import numpy as np
import cv2
import tensorflow as tf


def temp_directory():
    return os.path.abspath(os.path.join('.', 'data'))


MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
GRAPH_PATH = os.path.join(temp_directory(), MODEL_NAME,
                          'frozen_inference_graph.pb')
LOCAL_TMP_PATH = "./tmp/"


def get_mscoco_label_dict():
    mscoco_label_dict = {1: ['/m/01g317', 'person'], 2: ['/m/0199g', 'bicycle'], 3: ['/m/0k4j', 'car'],
                         4: ['/m/04_sv', 'motorcycle'], 5: ['/m/05czz6l', 'airplane'], 6: ['/m/01bjv', 'bus'],
                         7: ['/m/07jdr', 'train'], 8: ['/m/07r04', 'truck'], 9: ['/m/019jd', 'boat'],
                         10: ['/m/015qff', 'traffic light'], 11: ['/m/01pns0', 'fire hydrant'],
                         13: ['/m/02pv19', 'stop sign'], 14: ['/m/015qbp', 'parking meter'], 15: ['/m/0cvnqh', 'bench'],
                         16: ['/m/015p6', 'bird'], 17: ['/m/01yrx', 'cat'], 18: ['/m/0bt9lr', 'dog'],
                         19: ['/m/03k3r', 'horse'], 20: ['/m/07bgp', 'sheep'], 21: ['/m/01xq0k1', 'cow'],
                         22: ['/m/0bwd_0j', 'elephant'], 23: ['/m/01dws', 'bear'], 24: ['/m/0898b', 'zebra'],
                         25: ['/m/03bk1', 'giraffe'], 27: ['/m/01940j', 'backpack'], 28: ['/m/0hnnb', 'umbrella'],
                         31: ['/m/080hkjn', 'handbag'], 32: ['/m/01rkbr', 'tie'], 33: ['/m/01s55n', 'suitcase'],
                         34: ['/m/02wmf', 'frisbee'], 35: ['/m/071p9', 'skis'], 36: ['/m/06__v', 'snowboard'],
                         37: ['/m/018xm', 'sports ball'], 38: ['/m/02zt3', 'kite'], 39: ['/m/03g8mr', 'baseball bat'],
                         40: ['/m/03grzl', 'baseball glove'], 41: ['/m/06_fw', 'skateboard'],
                         42: ['/m/019w40', 'surfboard'], 43: ['/m/0dv9c', 'tennis racket'],
                         44: ['/m/04dr76w', 'bottle'], 46: ['/m/09tvcd', 'wine glass'], 47: ['/m/08gqpm', 'cup'],
                         48: ['/m/0dt3t', 'fork'], 49: ['/m/04ctx', 'knife'], 50: ['/m/0cmx8', 'spoon'],
                         51: ['/m/04kkgm', 'bowl'], 52: ['/m/09qck', 'banana'], 53: ['/m/014j1m', 'apple'],
                         54: ['/m/0l515', 'sandwich'], 55: ['/m/0cyhj_', 'orange'], 56: ['/m/0hkxq', 'broccoli'],
                         57: ['/m/0fj52s', 'carrot'], 58: ['/m/01b9xk', 'hot dog'], 59: ['/m/0663v', 'pizza'],
                         60: ['/m/0jy4k', 'donut'], 61: ['/m/0fszt', 'cake'], 62: ['/m/01mzpv', 'chair'],
                         63: ['/m/02crq1', 'couch'], 64: ['/m/03fp41', 'potted plant'], 65: ['/m/03ssj5', 'bed'],
                         67: ['/m/04bcr3', 'dining table'], 70: ['/m/09g1w', 'toilet'], 72: ['/m/07c52', 'tv'],
                         73: ['/m/01c648', 'laptop'], 74: ['/m/020lf', 'mouse'], 75: ['/m/0qjjc', 'remote'],
                         76: ['/m/01m2v', 'keyboard'], 77: ['/m/050k8', 'cell phone'], 78: ['/m/0fx9l', 'microwave'],
                         79: ['/m/029bxz', 'oven'], 80: ['/m/01k6s3', 'toaster'], 81: ['/m/0130jx', 'sink'],
                         82: ['/m/040b_t', 'refrigerator'], 84: ['/m/0bt_c3', 'book'], 85: ['/m/01x3z', 'clock'],
                         86: ['/m/02s195', 'vase'], 87: ['/m/01lsmm', 'scissors'], 88: ['/m/0kmg4', 'teddy bear'],
                         89: ['/m/03wvsk', 'hair drier'], 90: ['/m/012xff', 'toothbrush']}
    return mscoco_label_dict


def cv2base64(image):
    try:
        tmp_file_name = os.path.join(LOCAL_TMP_PATH, "%s.jpg" % uuid.uuid1())
        cv2.imwrite(tmp_file_name, image)
        with open(tmp_file_name, "rb") as f:
            base64_data = base64.b64encode(f.read())
            base64_data = base64_data.decode("utf-8")
        return base64_data
    except Exception as e:
        err_msg = "Convert cv2 object to base64 failed: "
        logging.error(err_msg, e, exc_info=True)
        raise e


class BoundingBox:
    def __init__(self, x1, y1, x2, y2, score, label=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.score = score
        self.label = label


class SSDDetectObject:
    def __init__(self):
        self.fetch_resources()
        self.model_init = False
        self.user_config = self.get_operator_config()
        self.label_dict = get_mscoco_label_dict()
        # initialize model
        try:
            self.graph = self.build_graph()
            with self.graph.as_default():
                with tf.device(self.device_str):
                    self.session = tf.Session(
                        config=self.user_config, graph=self.graph)
                    with self.session.as_default():
                        self.bulk_execute(np.zeros((1, 300, 300, 3)))
        except Exception as e:
            logging.error(
                "unexpected error happen during build graph",
                exc_info=True)
            raise e

    def get_operator_config(self):
        try:
            self.device_str = os.environ.get("device_id", "/cpu:0")
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            gpu_mem_limit = float(os.environ.get("gpu_mem_limit", 0.3))
            config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_limit
            # for device debug info print
            if os.environ.get("log_device_placement", False):
                config.log_device_placement = True
            logging.info("device id %s, gpu memory limit: %f",
                         self.device_str, gpu_mem_limit)

        except Exception as e:
            logging.error(
                "unexpected error happen during read config",
                exc_info=True)
            raise e
        logging.info("Model device str: %s, session config: %s",
                     self.device_str, config)
        return config

    def build_graph(self):
        dnn = tf.Graph()
        return dnn

    def load_model(self):
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(GRAPH_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.model_init = True

    def fetch_resources(self):
        # model_tar_path = download_temp_file(DOWNLOAD_BASE + MODEL_FILE)
        # with tarfile.open(model_tar_path) as f:
        #     f.extractall(temp_directory())
        # download_temp_file(LABEL_URL)
        pass

    def get_bboxes(self, images, boxes, scores, classes, threshold=0.4):
        bboxes = []
        for i in range(len(images)):
            bbox = []
            for (box, score, cls) in zip(boxes[i, :, :].reshape(100, 4),
                                         scores[i, :].reshape(100, 1),
                                         classes[i, :].reshape(100, 1)):
                if score > threshold:
                    bbox.append(BoundingBox(
                        x1=box[1], y1=box[0], x2=box[3], y2=box[2], score=score,
                        label=self.label_dict[int(cls[0])][1]))
            bboxes.append(bbox)
        return bboxes

    @staticmethod
    def get_obj_image(images, bboxes):
        obj_images = []
        for i, frame_bboxes in enumerate(bboxes):
            frame_object = []
            [h, w] = images[i].shape[:2]
            for j, bbox in enumerate(frame_bboxes):
                tmp = images[i][int(bbox.y1 * h):int(bbox.y2 * h), int(bbox.x1 * w):int(bbox.x2 * w)]
                frame_object.append(cv2base64(tmp))
            obj_images.append(frame_object)
        return obj_images

    @staticmethod
    def get_label(bboxes):
        obj_labels = []
        for i, frame_bboxes in enumerate(bboxes):
            frame_object = []
            for j, bboxes in enumerate(frame_bboxes):
                tmp_label = bboxes.label
                frame_object.append(tmp_label)
            obj_labels.append(frame_object)
        return obj_labels

    def execute(self, image):
        objs = self.bulk_execute([image])
        return objs[0]

    def bulk_execute(self, images):
        with self.graph.as_default():
            with tf.device(self.device_str):
                if not self.model_init:
                    self.load_model()
                image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
                boxes = self.graph.get_tensor_by_name('detection_boxes:0')
                scores = self.graph.get_tensor_by_name('detection_scores:0')
                classes = self.graph.get_tensor_by_name('detection_classes:0')
                with self.session.as_default():
                    (boxes, scores, classes) = self.session.run([boxes, scores, classes], feed_dict={
                        image_tensor: np.concatenate(np.expand_dims(images, axis=0), axis=0)})
                    bboxes = self.get_bboxes(images, boxes, scores, classes)
                    logging.debug(bboxes)
                    # objs = self.get_obj_image(images, bboxes)
                    objs = self.get_label(bboxes)
                    return objs

    @property
    def name(self):
        return "ssd"

    @property
    def type(self):
        return "processor"

    @property
    def accept_filetype(self):
        return ["png", "jpg", "jepg"]

    @property
    def input(self):
        return "image"

    @property
    def output(self):
        return "tags"

    @property
    def dimension(self):
        return "-1"

    @property
    def metric_type(self):
        return "-1"


def save_tmp_file(name, file_data=None, url=None):
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
                # raise DecodeError("Encode method not base64")
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
            # raise DownloadFileError("Download file from url %s" % url, e)
    return file_path


def run(detector, images, urls):
    result_images = []
    start = time.time()
    try:
        if images:
            for img in images:
                file_name = "{}-{}".format("processor", uuid.uuid4().hex)
                image_path = save_tmp_file(file_name, file_data=img)
                if image_path:
                    image = cv2.imread(image_path)
                    result_images.extend(detector.bulk_execute([image]))
        else:
            for url in urls:
                file_name = "{}-{}".format("processor", uuid.uuid4().hex)
                image_path = save_tmp_file(file_name, url=url)
                if image_path:
                    image = cv2.imread(image_path)
                    result_images.extend(detector.bulk_execute([image]))
    except Exception as e:
        logging.error("something error: %s", str(e), exc_info=True)
        pass
    end = time.time()
    logging.info('%s cost: {:.3f}s'.format(end - start), "ssd detector")
    return result_images


# todo: just for test, delete it when ok
if __name__ == "__main__":
    detector = SSDDetectObject()
    # url = "https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=1895204225,2410466361&fm=26&gp=0.jpg"
    # url = 'https://ss0.bdstatic.com/94oJfD_bAAcT8t7mm9GUKT-xh_/timg?image&quality=100&size=b4000_4000&sec=1589339704&di=b1af59201e401849ae9898ef0d854cb2&src=http://upload.fjii.com/2018/1020/1540020616285.jpg'
    # url = 'https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=1080913669,3502446330&fm=26&gp=0.jpg'
    # url = 'http://b-ssl.duitang.com/uploads/blog/201307/18/20130718170434_GNAGv.jpeg'
    # url = 'http://gss0.baidu.com/-4o3dSag_xI4khGko9WTAnF6hhy/zhidao/pic/item/9213b07eca806538b01f3a3696dda144ad34821a.jpg'
    # url = 'http://00.minipic.eastday.com/20160714/20160714000053_d41d8cd98f00b204e9800998ecf8427e_17.jpeg'
    # url = 'http://pic4.zhimg.com/50/v2-76f385cf0b54a8a2c7e826e8ee0decc4_hd.jpg'
    # url = 'https://up.enterdesk.com/edpic_source/0a/98/c1/0a98c103d73f5fab915a447860701e53.jpg'
    # url = 'http://pic1.win4000.com/mobile/2020-04-17/5e995f7e6a194.jpg'
    # url = 'http://pic1.win4000.com/mobile/2020-04-17/5e995f81a7e76.jpg'
    # url = 'http://pic1.win4000.com/mobile/2020-04-17/5e995f85ba60f.jpg'
    # url = "http://5b0988e595225.cdn.sohucs.com/images/20180303/8e09f1dfa6c1484697d50124c368182a.jpeg"
    url = "http://thumbs.dreamstime.com/z/%E4%BA%BA%E5%92%8C%E7%8B%97-50640419.jpg"
    a = run(detector, None, urls=[url])
    print(a)
