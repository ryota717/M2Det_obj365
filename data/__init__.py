# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .coco import COCODetection
from .objects365 import obj365Detection
from .data_augment import *
from .anchors import *
