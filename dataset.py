import random
import re
from pathlib import Path

import cv2
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset

y_k_size = 6
x_k_size = 6


class CamImage:
    def __init__(self, target_img_path: Path):
        self.target_path = target_img_path
        self.annotation_path = None
        self.__get_information()

    def __get_information(self):
        target_name = self.target_path.stem
        name_items = str(target_name).split("_")
        self.waypoint = name_items[0]
        self.type = name_items[1]
        self.weather = name_items[5]
        self.town = self.target_path.parents[2].name

    def get_regular_annotation_name(self, suffix=""):
        target_name = self.target_path.stem
        name_items = str(target_name).split("_")
        reg_exp = f"{name_items[0]}_{name_items[1]}_" + "[^_]+" + f"_{name_items[3]}_{name_items[4]}_{name_items[5]}" + "\w*" + suffix
        return reg_exp

    def set_annotation(self, annotation_img_path):
        self.annotation_path = annotation_img_path

    def load_images(self):
        target = self.__read_image(str(self.target_path))
        annotation = self.__read_image(str(self.annotation_path))
        return target, annotation

    def __read_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        return img


class BaseDataset(Dataset):
    def __init__(self,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 512),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image, city=True):
        if city:
            image = image.astype(np.float32)[:, :, ::-1]
        else:
            image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype(np.uint8)

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label, edge):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))
        edge = self.pad_image(edge, h, w, self.crop_size,
                              (0.0,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        edge = edge[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label, edge

    def multi_scale_aug(self, image, label=None, edge=None,
                        rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                edge = cv2.resize(edge, (new_w, new_h),
                                  interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label, edge = self.rand_crop(image, label, edge)

        return image, label, edge

    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True, edge_pad=True, edge_size=4, city=True):

        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size, y_k_size), (x_k_size, x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0

        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label, edge = self.multi_scale_aug(image, label, edge,
                                                      rand_scale=rand_scale)

        image = self.input_transform(image, city=city)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            edge = edge[:, ::flip]

        return image, label, edge

    def inference(self, config, model, image):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        return pred.exp()


class CarlaDataset(BaseDataset):
    def __init__(self, wp_root, waypoint_size=None, real_cam='rgb', annotation_cam='seg', color_annotation=False,
                 random_state=42):
        super().__init__()
        random.seed(random_state)

        annotation_suffix = "color" if color_annotation else "raw"
        # Получим список путевых точек
        waypoint_path = []
        for p in sorted(wp_root) if isinstance(wp_root, (list, tuple)) else [wp_root]:
            p = Path(p).resolve()
            if p.is_dir():
                waypoint_path.extend(sorted([path for path in p.iterdir()]))
            else:
                raise FileNotFoundError(f'{str(p)} does not exist')
        random.shuffle(waypoint_path)
        if waypoint_size is not None:
            waypoint_path = waypoint_path[:waypoint_size]
        # Cписок камер
        real_cam = real_cam
        annotation_cam = annotation_cam
        # Получим список изображений
        cam_image_list = []
        for wp_path in waypoint_path:
            # Список путей до изображений для текущей точки
            target_path_list = [path for path in Path(wp_path, real_cam).iterdir()]
            # Список путей до аннотаций для текущей точки
            annotation_path_list = [path for path in Path(wp_path, annotation_cam).iterdir()]

            # Для каждого изображения ищем соответствующую ему аннотацию
            for target_path in target_path_list:
                image = CamImage(target_path)
                # Регулярное выражение искомого названия аннотации
                annotation_name_reg_exp = image.get_regular_annotation_name() + annotation_suffix

                # Проходимся по каждой аннотации и ищем подходащий под регулярку файл
                for i, annotation_path in enumerate(annotation_path_list):
                    annotation_name = annotation_path.stem
                    if re.fullmatch(annotation_name_reg_exp, annotation_name) is not None:
                        annotation_path_list.pop(i)
                        image.set_annotation(annotation_path)
                        cam_image_list.append(image)
                        break

        random.shuffle(cam_image_list)
        self.image_list = cam_image_list
        self.image_idx = 0

        self.label_mapping = {0: 1, 1: 2, 2: 3,
                              3: 4, 4: 5, 5: 6,
                              6: 7, 7: 8, 8: 9,
                              9: 10, 10: 11, 11: 12,
                              12: 13, 13: 14, 14: 15,
                              15: 16, 16: 17, 17: 18,
                              18: 19, 19: 20, 20: 21,
                              21: 22, 22: 23}

    def __iter__(self):
        self.image_idx = 0
        return self

    def __next__(self):
        if self.image_idx >= len(self.image_list):
            raise StopIteration

        image = self.image_list[self.image_idx]
        img, label = image.load_images()
        self.image_idx += 1

        return img, label

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, idx):
        image = self.image_list[idx]
        img, label = image.load_images()
        label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        img, label, edge = self.gen_sample(img, label, True, True, edge_size=4)
        label = self.convert_label(label)
        size = img.shape
        return img.copy(), label.copy(), edge.copy(), np.array(size), f'{img}_{idx}'

    def __len__(self):
        return len(self.image_list)
