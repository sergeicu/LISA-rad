import glob
import json
import os
import random

import re 

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST,SHORT_OBJ_LIST

DEFAULT_IMAGE_TOKEN = "<image>"


def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    with open("utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ade20k",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels

def extract_json(f,grounded=True):
    
    with open(f) as file:
        jk = json.load(file)
    
    q=jk['formatted_response']['question']
    a=jk['formatted_response']['answer']
    gp=jk['formatted_response']['grounding_phrase']
    
    if grounded: 
        
        # # OPTION 1: replace 
        # if not gp in a:
        #     # print(gp)
        
        #     # Split both strings into words
        #     gp_words = gp.split()
        #     a_words = a.split()

        #     # Keep only words in gp that are also in a
        #                                     # gp_filtered = []
        #                                     # for word in a_words:
        #                                     #     gp_filtered.append(word)
        #                                     # gp_filtered = ' '.join(gp_filtered)
        #     gp = ' '.join(word for word in gp_words if word in a_words)
        # a = a.replace(gp, "[OBJ]" + gp+ "[OBJ]")
        
        # OPTION 2: append 
        a = "<p>" + gp+ "</p> " + a 
    

    
    
    # import transformers
    # model_max_length=512
    # version="xinlai/LISA-7B-v1"
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     version,
    #     cache_dir=None,
    #     model_max_length=model_max_length,
    #     padding_side="right",
    #     use_fast=False,
    # )    
    # https://www.reddit.com/r/StableDiffusion/comments/zc65l4/rare_tokens_for_dreambooth_training_stable/
    
    
    return q,a
    

def init_bchwrist(base_image_dir):
    bchwrist_classes = []
    with open("utils/bchwrist_classes.txt") as f:
        for line in f.readlines()[0:]:
            bchwrist_classes.append(line.strip().split(": ")[-1])
    bchwrist_classes = np.array(bchwrist_classes)
    bchwrist_images = []

    bchwrist_labels = glob.glob(
        os.path.join(base_image_dir, "bchwrist_labels", "train", "*.png")
    )
    bchwrist_images = [
        x.replace("bchwrist_labels", "bchwrist_images") for x in bchwrist_labels
    ]

    bchwrist_reports = [re.match(r'^\d+', os.path.basename(filename))[0]+'_report_v8.json' for filename in bchwrist_labels]
    bchwrist_reports = [
        base_image_dir + '/bchwrist_reports/train/'+x for x in bchwrist_reports
    ]        
    
    
    qa = [ extract_json(f) for f in bchwrist_reports]
    

    print("bchwrist: ", len(bchwrist_images))
    return bchwrist_classes, bchwrist_images, bchwrist_labels, qa


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class SemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary||bchwrist",
        grounded=True,
        deterministic=False,
        shorten=False,
    ):

        # Only initialize CLIPImageProcessor if vision_tower is not None
        if vision_tower is not None:
            from transformers import CLIPImageProcessor
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        else:
            self.clip_image_processor = None

        self.shorten = shorten
        self.deterministic=deterministic
        self.grounded=grounded
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.obj_list = SHORT_OBJ_LIST

        self.data2list = {}
        self.data2classes = {}
        self.data2qa = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            if 'bchwrist' in ds:
                classes, images, labels,qa = eval("init_{}".format(ds))(base_image_dir)
                if self.shorten:
                    classes = classes[:4]
                    images = images[:4]
                    labels = labels[:4]
                    qa = qa[:4]
                self.data2qa[ds] = qa
            else:
                classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }
            
        if "bchwrist" in self.sem_seg_datas:
            self.bchwrist_class2index = {
                c: i for i, c in enumerate(self.data2classes["bchwrist"])
            }          
            

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1) if not self.deterministic else 0
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self.__getitem__(0)
            if len(anns) >= self.num_classes_per_sample:
                sampled_anns = np.random.choice(
                    anns, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = anns
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append(name)

        elif ds in ["ade20k", "cocostuff", "mapillary", "bchwrist"]:
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1) if not self.deterministic else 0
            
            # sampled from gpt 
            qa_from_gpt = self.data2qa[ds]
            q_=qa_from_gpt[idx][0]
            a_=qa_from_gpt[idx][1]
                        
            image_path = image[idx]
            label_path = labels[idx]
            if self.clip_image_processor is None:
                sampled_classes = ['fracture']
            else:
                label = Image.open(label_path)
                label = np.array(label)
                if ds == "ade20k":
                    label[label == 0] = 255
                    label -= 1
                    label[label == 254] = 255
                elif ds == "cocostuff"      :
                    for c, i in self.cocostuff_class2index.items():
                        if "-" in c:
                            label[label == i] = 255
                elif ds == "bchwrist":
                    # print("CHECK IF this is oke here - save the image")
                    # from IPython import embed; embed()
                    # print("CHECK IF this is oke here - save the image")
                    
                    for c, i in self.bchwrist_class2index.items():
                        if "-" in c:
                            label[label == i] = 255                        
                img = cv2.imread(image_path)
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # preprocess image for clip
                image_clip = self.clip_image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0]
                image = self.transform.apply_image(image)  # preprocess image for sam
                resize = image.shape[:2]
                unique_label = np.unique(label).tolist()
                if 255 in unique_label:
                    unique_label.remove(255)
                if len(unique_label) == 0:
                    return self.__getitem__(0)

                classes = [self.data2classes[ds][class_id] for class_id in unique_label]
                if len(classes) >= self.num_classes_per_sample:
                    sampled_classes = np.random.choice(
                        classes, size=self.num_classes_per_sample, replace=False
                    ).tolist()
                else:
                    sampled_classes = classes

        questions = []
        answers = []
        class_ids = []
        # sv407 - this is where questions get asked 
        for sampled_cls in sampled_classes:
            text = sampled_cls
            
            # get object 
            obj_template = random.choice(self.obj_list) if not self.deterministic else self.obj_list[0]
            # obj_template = obj_template + "Use the following format to highlight the description of the fracture type and its location in the image <p> fracture_description </p>"
            

            assert len(text.split("||")) == 1
            
            # NEW QUESTIONS ANSWERS
            if self.grounded: 
                
                # print("we want to make sure that we give the grounding labels here")
                # from IPython import embed; embed() 
                q = DEFAULT_IMAGE_TOKEN + "\n" + q_ + ' ' + obj_template 
                questions.append(q.format(class_name=text.lower()))   # should be:  DEFAULT_IMAGE_TOKEN + "\n" + <q from llava> + "Pinpoint its location in the report.",

                answer_template = random.choice(self.answer_list) if not self.deterministic else self.answer_list[0]
                # answers.append(answer_template + ' ' + a_ ) # should be:  
                answers.append("[SEG] " + a_ ) # should be:  
                
                # print(f"questions are:\n{questions}")
                # print(f"answers are:\n{answers}")

            # OLD QUESTIONS ANSWERS
            else:            
            
                question_template = random.choice(self.short_question_list) if not self.deterministic else self.short_question_list[0]
                questions.append(question_template.format(class_name=text.lower()))
       
                answers.append(random.choice(self.answer_list))





            if ds in ["paco_lvis", "pascal_part"]:
                continue

            class_id = self.data2classes[ds].tolist().index(sampled_cls)
            class_ids.append(class_id)

        conversations = []
        if self.grounded:
            conv = conversation_lib.conv_bch_v1.copy()
        else:
            conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        if self.clip_image_processor is not None:
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for ann in sampled_anns:
                try:
                    masks.append(coco_api.annToMask(ann))
                except Exception as e:
                    print(e)
                    return self.__getitem__(0)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        else:
            if self.clip_image_processor is not None:
                label = torch.from_numpy(label).long()
                masks = []
                for class_id in class_ids:
                    masks.append(label == class_id)
                masks = torch.stack(masks, dim=0)
            else:
                image = None
                image_clip = None
                masks = None
                label = None
                resize=None
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )
