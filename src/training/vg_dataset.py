import logging
import json
import os
import random
from PIL import Image
import numpy as np
import torch
from open_clip import tokenize
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import math

def repair_text(text):
    text_arr = text.split()
    new_text_arr = []
    for i in range(len(text_arr) - 1):
        if text_arr[i] != text_arr[i+1]:
            new_text_arr.append(text_arr[i])
    new_text_arr.append(text_arr[-1])
    new_text = " ".join(new_text_arr)
    return new_text


class VgDataset(Dataset):
    def __init__(self, vg_path, split, transforms, num_objects, num_samples):
        logging.debug(f'Loading data from visual genome.')
        f = open(os.path.join(vg_path,"image_data_" + split +  ".json"))
        self.image_data = json.load(f)
        f = open(os.path.join(vg_path,"attributes_" + split +  ".json"))
        self.object_attributes = json.load(f)
        f = open(os.path.join(vg_path,"objects_" + split +  ".json"))
        self.object = json.load(f)
        f = open(os.path.join(vg_path,"relationships_" + split +  ".json"))
        self.relationships = json.load(f)
        self.num_objects = num_objects
        self.vg_path = vg_path
        self.clip_image_size = 224
        self.transforms = transforms
        self.num_samples = num_samples
        self.split = split
        logging.debug('Done loading data.')

    def __len__(self):
        if self.num_samples is None:
            return len(self.object_attributes)
        else:
            return self.num_samples

    def __getitem__(self, idx):
        #load image
        image_url = self.image_data[idx]["url"]
        image_h = self.image_data[idx]["height"]
        image_w = self.image_data[idx]["width"]
        url_parts = image_url.split("/")
        folder = url_parts[5]
        filename = url_parts[6]
        image_path = os.path.join(self.vg_path,folder,folder,filename)

        image = Image.open(image_path)
        image = self.transforms(image)



        #load objects and choose randomly
        attributes = self.object_attributes[idx]["attributes"]
        objects = self.object[idx]["objects"]
        objects = [obj for obj in objects if obj["x"] + obj["w"] > obj["x"] and obj["y"] + obj["h"] > obj["y"]]
        if len(objects) > self.num_objects:
            objects = random.sample(objects,self.num_objects)
        missing_objects = self.num_objects - len(objects)
        valid_objects = torch.tensor(self.num_objects - missing_objects, dtype=torch.long)
        
        #prepare bounding boxes
        bounding_boxes = [[(ob["x"] + 0.5*ob["w"])/image_w,(ob["y"] + 0.5*ob["h"])/image_h,min((ob["w"])/image_w,1.0),min((ob["h"])/image_h,1.0)] for ob in objects]
        if missing_objects > 0:
            bounding_boxes += [[0.0,0.0,0.0,0.0] for i in range(missing_objects)]
        bounding_boxes = torch.tensor(bounding_boxes)
        attr = []
        found = False
        object_ids = [ob["object_id"] for ob in objects]
        for id in object_ids:
            for a in attributes:
                if a["object_id"] == id:
                    found = True
                    attr.append(a)
            if not found:
                print("couldnt find match")
            

        #prepare object descriptions
        object_names = [a["names"] for a in attr]
        object_attributes = [" ".join(a["attributes"]) if "attributes" in a else "" for a in attr]
        object_descriptions = [text1 + " " +  " ".join(text2) for text1, text2 in zip(object_attributes,object_names)]
        if missing_objects > 0:
            object_descriptions += ["" for i in range(missing_objects)]
        if self.split == "val":
            max_chars = 150
            object_text = [[ord(c) for c in s] for s in object_descriptions]
            padding = [[int(36) for i in range(max_chars - len(s))] for s in object_descriptions]
            object_texts = [lst1 + lst2 for lst1, lst2 in zip(object_text,padding)]
            text_lengths = [len(s) for s in object_descriptions]
        object_descriptions = tokenize(object_descriptions)


        if self.split == "train":
            return image, valid_objects, bounding_boxes, object_descriptions
        else:
            return image, valid_objects, bounding_boxes, object_descriptions, torch.tensor(object_texts), text_lengths


class VgDatasetText(Dataset):
    def __init__(self, vg_path, split, transforms, num_objects, num_samples):
        logging.debug(f'Loading data from visual genome.')
        f = open(os.path.join(vg_path,"vg_text_dataset_" + split +  ".json"))
        self.data = json.load(f)
        self.vg_path = vg_path
        self.clip_image_size = 224
        self.transforms = transforms
        self.num_samples = num_samples
        self.split = split
        self.num_objects = num_objects if num_objects > 0 else 10
        logging.debug('Done loading data.')

    def __len__(self):
        if self.num_samples is None:
            return len(self.data)
        else:
            return self.num_samples

    def __getitem__(self, idx):
        #load image
        image_url = self.data[idx]["image_data"]["url"]
        crop_dimensions = self.data[idx]["to_crop"]
        min_x = crop_dimensions[0]
        min_y = crop_dimensions[1]
        max_x = crop_dimensions[2]
        max_y = crop_dimensions[3]
        image_h = max_y - min_y
        image_w = max_x - min_x
        url_parts = image_url.split("/")
        folder = url_parts[5]
        filename = url_parts[6]
        image_path = os.path.join(self.vg_path,folder,folder,filename)

        image = Image.open(image_path)
        image = image.crop(crop_dimensions)
        image = self.transforms(image)

        text = self.data[idx]["text"]
        text = repair_text(text)



        #load objects and choose randomly
        objects = self.data[idx]["objects"]
        missing_objects = self.num_objects - len(objects)
        valid_objects = torch.tensor(self.num_objects - missing_objects, dtype=torch.long)
        
        #prepare bounding boxes
        objects_bbs = [[ob["x"],ob["y"],ob["w"], ob["h"]] for ob in objects]
        for obj in objects_bbs:
            new_x1 = obj[0]
            new_y1 = obj[1]
            new_x2 = obj[0] + obj[2]
            new_y2 = obj[1] + obj[3]
            if obj[0] < min_x:
                new_x1 = min_x
            if obj[1] < min_y:
                new_y1 = min_y
            if obj[0] + obj[2] > max_x:
                new_x2 = max_x
            if obj[1] + obj[3] > max_y:
                new_y2 = max_y
            obj[0] = new_x1 - min_x
            obj[1] = new_y1 - min_y
            obj[2] = new_x2 - new_x1
            obj[3] = new_y2 - new_y1
            
        bounding_boxes = [[(ob[0] + 0.5*ob[2])/image_w,(ob[1] + 0.5*ob[3])/image_h,min((ob[2])/image_w,1.0),min((ob[3])/image_h,1.0)] for ob in objects_bbs]
        if missing_objects > 0:
            bounding_boxes += [[0.0,0.0,0.0,0.0] for i in range(missing_objects)]
        bounding_boxes = torch.tensor(bounding_boxes)

        #prepare object descriptions
        object_descriptions = [obj["attributes"][0] + " " + obj["names"][0] if "attributes" in obj else obj["names"][0] for obj in objects]
        object_descriptions = [repair_text(desc)for desc in object_descriptions]
        if missing_objects > 0:
            object_descriptions += ["" for i in range(missing_objects)]
        if self.split == "val":
            max_chars = 150
            object_text = [[ord(c) for c in s] for s in object_descriptions]
            padding = [[int(36) for i in range(max_chars - len(s))] for s in object_descriptions]
            object_texts = [lst1 + lst2 for lst1, lst2 in zip(object_text,padding)]
            text_lengths = [len(s) for s in object_descriptions]
        object_descriptions = tokenize(object_descriptions)
        text = tokenize([text])[0]


        if self.split == "train":
            return image, text, valid_objects, bounding_boxes, object_descriptions
        else:
            return image, text, valid_objects, bounding_boxes, object_descriptions, torch.tensor(object_texts), text_lengths

class VgDatasetIterable(IterableDataset):
    def __init__(self, vg_path, transforms, num_objects,shard_id,shard_size):
        logging.debug(f'Loading data from visual genome.')
        f = open(os.path.join(vg_path,"image_data.json"))
        self.image_data = json.load(f)
        f = open(os.path.join(vg_path,"attributes.json"))
        self.object_attributes = json.load(f)
        f = open(os.path.join(vg_path,"objects.json"))
        self.object = json.load(f)
        f = open(os.path.join(vg_path,"relationships.json"))
        self.relationships = json.load(f)
        self.num_objects = num_objects
        self.vg_path = vg_path
        self.clip_image_size = 224
        self.transforms = transforms
        self.shard_id = shard_id
        self.start= shard_size * shard_id
        self.end = shard_size * (shard_id + 1)
        logging.debug('Done loading data.')

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
             # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        for idx in range(iter_start,iter_end):
            yield self.getitem(idx)


    def getitem(self, idx):
        #load image
        image_url = self.image_data[idx]["url"]
        image_h = self.image_data[idx]["height"]
        image_w = self.image_data[idx]["width"]
        url_parts = image_url.split("/")
        folder = url_parts[5]
        filename = url_parts[6]
        image_path = os.path.join(self.vg_path,folder,folder,filename)

        image = Image.open(image_path)
        image = self.transforms(image)



        #load objects and choose randomly
        attributes = self.object_attributes[idx]["attributes"]
        objects = self.object[idx]["objects"]
        objects = [obj for obj in objects if obj["x"] + obj["w"] > obj["x"] and obj["y"] + obj["h"] > obj["y"]]
        if len(objects) > self.num_objects:
            objects = random.sample(objects,self.num_objects)
        missing_objects = self.num_objects - len(objects)
        valid_objects = torch.tensor(self.num_objects - missing_objects, dtype=torch.long)
        
        #prepare bounding boxes
        bounding_boxes = [[(ob["x"] + 0.5*ob["w"])/image_w,(ob["y"] + 0.5*ob["h"])/image_h,min((ob["w"])/image_w,1.0),min((ob["h"])/image_h,1.0)] for ob in objects] + [[0,0,0,0] for i in range(missing_objects)]
        bounding_boxes = torch.tensor(bounding_boxes)
        
        attr = []
        object_ids = [ob["object_id"] for ob in objects]
        for id in object_ids:
            for a in attributes:
                if a["object_id"] == id:
                    attr.append(a)
            

        #prepare object descriptions
        object_names = [a["names"] for a in attr]
        object_attributes = [" ".join(a["attributes"]) if "attributes" in a else "" for a in attr]
        object_descriptions = [text1 + " " +  " ".join(text2) for text1, text2 in zip(object_attributes,object_names)] + ["" for i in range(missing_objects)]
        object_descriptions = tokenize(object_descriptions)



        return image, valid_objects, bounding_boxes, object_descriptions

def get_vg_loader(dataset, args, vg_batch_size):
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=vg_batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler
    )
    return dataloader

def get_vg_loader_it(dataset, args, vg_batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=vg_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
    )
    return dataloader

