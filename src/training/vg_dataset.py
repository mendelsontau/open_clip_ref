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



class VgDataset(Dataset):
    def __init__(self, vg_path, transforms, num_objects):
        logging.debug(f'Loading data from visual genome.')
        f = open(os.path.join(vg_path,"image_data.json"))
        self.image_data = json.load(f)
        f = open(os.path.join(vg_path,"attributes.json"))
        self.object_attributes = json.load(f)
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
        objects = self.object_attributes[idx]["attributes"]
        objects = random.sample(objects,self.num_objects)
        object_ids = [ob["object_id"] for ob in objects]
        bounding_boxes = [[ob["x"]/image_w,ob["y"]/image_h,(ob["x"] + ob["w"])/image_w,(ob["y"] + ob["h"])/image_h] for ob in objects]
        bounding_boxes = torch.tensor(bounding_boxes)
        object_names = [ob["names"] for ob in objects]
        object_attributes = [" ".join(ob["attributes"]) if "attributes" in ob else "" for ob in objects]
        object_descriptions = [text1 + " " +  " ".join(text2) for text1, text2 in zip(object_attributes,object_names)]
        object_descriptions = tokenize(object_descriptions)



        return image, bounding_boxes, object_descriptions

def get_vg_loader(dataset, args, vg_batch_size):
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=vg_batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    return dataloader

