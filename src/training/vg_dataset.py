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


class VgDataset(Dataset):
    def __init__(self, vg_path, transforms, num_objects):
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
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.object_attributes)

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
        object_descriptions = tokenize(object_descriptions)



        return image, valid_objects, bounding_boxes, object_descriptions

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

