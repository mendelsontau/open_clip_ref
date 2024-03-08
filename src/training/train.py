import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from PIL import Image

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss, tokenize, HNLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast

from datasets import load_dataset
from tqdm import tqdm
from torchvision import transforms as transforms
from torchvision.transforms.functional import convert_image_dtype
from detr.util.box_ops import box_cxcywh_to_xyxy
from torchvision.utils import draw_bounding_boxes


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def organize_batch_classes(object_descriptions, valid_objects, vg_bbs, args, device):
    class_tokens = []
    object_samples = []
    tgt_boxes = []
    for i in range (valid_objects.shape[0]):
        valid_samples = valid_objects[i].item()
        invalid_samples = args.objects - valid_samples
        valid_for_sample = valid_samples * [True] + invalid_samples*[False]
        object_samples.append(valid_for_sample)

        if valid_samples == 0:
            tgt_boxes.append(torch.tensor([]).to(device=device, non_blocking=True))
        else:
            mask = torch.tensor(valid_for_sample).unsqueeze(1).expand(-1,4)
            boxes_for_sample = torch.masked_select(vg_bbs[i],mask).view(-1,4).to(device=device, non_blocking=True)
            tgt_boxes.append(boxes_for_sample)
    
    
    tgt_labels = []
    for i in range(object_descriptions.shape[0]):
        labels_for_sample = []
        for j in range(object_descriptions.shape[1]):
            if object_samples[i][j] == False:
                continue
            desc = object_descriptions[i][j]
            exists = False
            for k in range (len(class_tokens)):
                if torch.equal(desc,class_tokens[k]):
                    exists = True
                    labels_for_sample.append(k)
                    break
            if not exists:
                labels_for_sample.append(len(class_tokens))
                class_tokens.append(desc)
        tgt_labels.append(torch.tensor(labels_for_sample).type(torch.int64).to(device=device, non_blocking=True))
    class_tokens = torch.stack(class_tokens).to(device,non_blocking=True)
    targets = [{"labels": l, "boxes": b} for l,b in zip(tgt_labels,tgt_boxes)]
    return class_tokens, targets


def organize_batch_classes_relations(relations_descriptions, valid_relations, relations_bbs, args, device):
    class_tokens = []
    relation_samples = []
    tgt_boxes = []
    for i in range (valid_relations.shape[0]):
        valid_samples = valid_relations[i].item()
        invalid_samples = args.relation_tokens - valid_samples
        valid_for_sample = valid_samples * [True] + invalid_samples*[False]
        relation_samples.append(valid_for_sample)

        if valid_samples == 0:
            tgt_boxes.append(torch.tensor([]).to(device=device, non_blocking=True))
        else:
            mask = torch.tensor(valid_for_sample).unsqueeze(1).expand(-1,4)
            boxes_for_sample = torch.masked_select(relations_bbs[i],mask).view(-1,4).to(device=device, non_blocking=True)
            tgt_boxes.append(boxes_for_sample)
    
    
    tgt_labels = []
    for i in range(relations_descriptions.shape[0]):
        labels_for_sample = []
        for j in range(relations_descriptions.shape[1]):
            if relation_samples[i][j] == False:
                continue
            desc = relations_descriptions[i][j]
            exists = False
            for k in range (len(class_tokens)):
                if torch.equal(desc,class_tokens[k]):
                    exists = True
                    labels_for_sample.append(k)
                    break
            if not exists:
                labels_for_sample.append(len(class_tokens))
                class_tokens.append(desc)
        tgt_labels.append(torch.tensor(labels_for_sample).type(torch.int64).to(device=device, non_blocking=True))
    class_tokens = torch.stack(class_tokens).to(device,non_blocking=True)
    targets = [{"labels": l, "boxes": b} for l,b in zip(tgt_labels,tgt_boxes)]
    return class_tokens, targets




def train_one_epoch(model, freeze_model, object_head, bb_head, relation_head, rel_bb_head, random_rows, vgcriterion, vgrelcriterion, data, vg_dataloader, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()

    if args.vg_loss_lambda > 0:
        freeze_model.train()
        object_head.train()
        bb_head.train()
        random_rows.train()
        if args.relations > 0:
            relation_head.train()
            rel_bb_head.train()

    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)
    if args.negatives:
        neg_loss = HNLoss(alpha=args.negatives_loss_lambda)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    if args.vg_data and args.distributed:
        vg_dataloader.sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    clip_loss_m = AverageMeter()
    negatives_loss_m = AverageMeter()
    vg_loss_m = AverageMeter()
    ce_loss_m = AverageMeter()
    bbox_loss_m = AverageMeter()
    giou_loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    if args.vg_data:
        vg_iter = iter(vg_dataloader)
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        images, texts = batch
        if args.vg_data:
            try:
                vg_batch = next(vg_iter)
            except StopIteration:
                vg_iter = iter(vg_dataloader)
                vg_batch = next(vg_iter)
            if args.vg_loss_lambda > 0.0:
                if args.relations > 0:
                    if args.negatives:
                        vg_images, vg_texts, valid_objects, vg_bbs, object_descriptions, valid_relations, vg_rel_bbs, relation_descriptions, neg_texts, neg_masks =  vg_batch
                        neg_masks = neg_masks.to(device,non_blocking=True)
                        texts = torch.cat([texts, vg_texts, neg_texts])
                    else:
                        vg_images, vg_texts, valid_objects, vg_bbs, object_descriptions, valid_relations, vg_rel_bbs, relation_descriptions =  vg_batch
                        texts = torch.cat([texts, vg_texts])
                    relations_descs, relations_targets = organize_batch_classes_relations(relation_descriptions,valid_relations,vg_rel_bbs,args,device)
                else:
                    if args.negatives:
                        vg_images, vg_texts, valid_objects, vg_bbs, object_descriptions, neg_texts, neg_masks =  vg_batch
                        neg_masks = neg_masks.to(device,non_blocking=True)
                        texts = torch.cat([texts, vg_texts, neg_texts])
                    else:
                        vg_images, vg_texts, valid_objects, vg_bbs, object_descriptions =  vg_batch
                        texts = torch.cat([texts, vg_texts])
                class_tokens, targets = organize_batch_classes(object_descriptions, valid_objects, vg_bbs, args, device)
            else:
                if args.negatives:
                    vg_images, vg_texts, neg_texts, neg_masks= vg_batch
                    neg_masks = neg_masks.to(device,non_blocking=True)
                    texts = torch.cat([texts, vg_texts, neg_texts]) 
                else:
                    vg_images, vg_texts = vg_batch
                    texts = torch.cat([texts, vg_texts])
            images = torch.cat([images,vg_images])
        images = images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)

        #logging.info("data ready")

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        with autocast():
            if args.vg_data and args.vg_loss_lambda > 0:
                with torch.no_grad():
                    if args.relations > 0:
                        all_descriptions = torch.cat([class_tokens,relations_descs])
                        all_description_embeddings = freeze_model(None, all_descriptions)
                        #all_description_embeddings = torch.rand(36,512).to(device,non_blocking=True)
                        object_description_embeddings = all_description_embeddings[:class_tokens.shape[0]]
                        relation_description_embeddings = all_description_embeddings[class_tokens.shape[0]:]
                    else:
                        object_description_embeddings = freeze_model(None, class_tokens) 
                if args.relations > 0: 
                    relation_description_embeddings = random_rows(relation_description_embeddings,"relations")
                object_description_embeddings = random_rows(object_description_embeddings,"objects")
            image_features, extra_tokens, text_features, logit_scale = model(images, texts)
            vg_losses = 0
            if args.vg_data and args.vg_loss_lambda > 0.0:
                object_tokens = extra_tokens[-vg_images.shape[0]:, : args.object_tokens,:]
                label_embeddings = object_head(object_tokens)
                label_predictions = logit_scale * label_embeddings @ object_description_embeddings.t()
                bb_predictions = bb_head(object_tokens).sigmoid()
                predictions_dict = {"pred_logits" : label_predictions, "pred_boxes": bb_predictions}
                objects_loss_dict = vgcriterion(predictions_dict, targets)
                weight_dict = vgcriterion.weight_dict
                vg_losses = sum(objects_loss_dict[k] * weight_dict[k] for k in objects_loss_dict.keys() if k in weight_dict)
                if args.relations > 0:
                    relation_tokens = extra_tokens[-vg_images.shape[0]:,args.object_tokens : args.object_tokens + args.relation_tokens,:]
                    label_embeddings = relation_head(relation_tokens)
                    label_predictions = logit_scale * label_embeddings @ relation_description_embeddings.t()
                    bb_predictions = rel_bb_head(relation_tokens).sigmoid()
                    predictions_dict = {"pred_logits" : label_predictions, "pred_boxes": bb_predictions}
                    relations_loss_dict = vgrelcriterion(predictions_dict, relations_targets)
                    weight_dict = vgrelcriterion.weight_dict
                    vg_losses += sum(relations_loss_dict[k] * weight_dict[k] for k in relations_loss_dict.keys() if k in weight_dict)
                    vg_losses /= 2
            if args.negatives:
                clip_loss = loss(image_features, text_features[:-vg_images.shape[0],:], logit_scale)
                negatives_loss = neg_loss(image_features, text_features, logit_scale, neg_masks, vg_images.shape[0])
                total_loss = clip_loss + vg_losses * args.vg_loss_lambda + negatives_loss
            else:
                total_loss = loss(image_features, text_features, logit_scale) + vg_losses * args.vg_loss_lambda

        #logging.info("computation")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        #logging.info("step")
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
        
        if args.vg_loss_lambda > 0.0:
            with torch.no_grad():
                if not args.distributed:          
                    for param, param_m in zip(model.parameters(), freeze_model.parameters()):
                        param_m.data = param_m.data * args.momentum + param.data * (1. - args.momentum)
                        param_m.requires_grad = False  # not update by gradient 
                else:
                    for param, param_m in zip(model.module.parameters(), freeze_model.parameters()):
                        param_m.data = param_m.data * args.momentum + param.data * (1. - args.momentum)
                        param_m.requires_grad = False  # not update by gradient 


        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            if vg_dataloader != None:
                vg_batch_size = vg_images.shape[0]
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            if args.negatives:
                clip_loss_m.update(clip_loss.item(),batch_size)
                negatives_loss_m.update(negatives_loss.item(), torch.sum(neg_masks).item())
            if args.vg_loss_lambda > 0.0:
                vg_loss_m.update(vg_losses.item(),vg_batch_size)
                ce_loss_m.update(objects_loss_dict["loss_ce"].item(),vg_batch_size)
                bbox_loss_m.update(objects_loss_dict["loss_bbox"].item(),vg_batch_size)
                giou_loss_m.update(objects_loss_dict["loss_giou"].item(),vg_batch_size)
            logit_scale_scalar = logit_scale.item()
            if args.vg_loss_lambda > 0.0 and args.negatives:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"clip_Loss: {clip_loss_m.val:#.5g} ({clip_loss_m.avg:#.4g}) "
                    f"negatives_loss: {negatives_loss_m.val:#.5g} ({negatives_loss_m.avg:#.4g}) "
                    f"vg_Loss: {vg_loss_m.val:#.5g} ({vg_loss_m.avg:#.4g}) "
                    f"ce_Loss: {ce_loss_m.val:#.5g} ({ce_loss_m.avg:#.4g}) "
                    f"bbox_Loss: {bbox_loss_m.val:#.5g} ({bbox_loss_m.avg:#.4g}) "
                    f"giou_Loss: {giou_loss_m.val:#.5g} ({giou_loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f}"
                )
            elif args.negatives:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"clip_Loss: {clip_loss_m.val:#.5g} ({clip_loss_m.avg:#.4g}) "
                    f"negatives_loss: {negatives_loss_m.val:#.5g} ({negatives_loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f}"
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f}"
                )


            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)

    
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics

def evaluate_winoground(model, clip_processor, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    autocast = get_autocast(args.precision)
    model.eval()
    #if not is_master(args):
    #    return metrics
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)

    #check if winoground folder exists
    if not os.path.exists(os.path.join(args.logs,args.name,"winoground")):
        os.mkdir(os.path.join(args.logs,args.name,"winoground"))
    result_dict = {}
    auth_token = "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"
    winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]
    categories_clip_scores = {}
    categories_clip_scores["All Dataset"] = []
    categories_clip_scores["Ambiguously Correct"] = []
    categories_clip_scores["Visually Difficult"] = []
    categories_clip_scores["Unusual Text"] = []
    categories_clip_scores["Complex Reasoning"] = []
    categories_clip_scores["Unusual Image"] = []
    categories_clip_scores["Non Minimal"] = []
    categories_clip_scores["No Tag"] = []

    #load tag assignments
    f = open("Winoground/tag_assignments.json")
    tag_assignments = json.load(f)

    for example in tqdm(winoground):
    # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
    # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
        image0 = clip_processor(example["image_0"].convert("RGB")).unsqueeze(0)
        image1 = clip_processor(example["image_1"].convert("RGB")).unsqueeze(0)
        images = torch.cat([image0,image1],dim=0)
        caption0 = tokenize(example["caption_0"])[0].unsqueeze(0)
        caption1 = tokenize(example["caption_1"])[0].unsqueeze(0)
        captions = torch.cat([caption0,caption1],dim=0)
        images = images.to(args.device, non_blocking=True)
        captions = captions.to(args.device, non_blocking=True)
        with torch.no_grad():
            with autocast():
                image_features, _, text_features, logit_scale = model(images, captions)
                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image = logits_per_image.detach().cpu()
        clip_score_c0_i0 = logits_per_image[0,0].item()
        clip_score_c1_i0 = logits_per_image[0,1].item()
        clip_score_c0_i1 = logits_per_image[1,0].item()
        clip_score_c1_i1 = logits_per_image[1,1].item()
        example_id = str(example["id"])
        all_tags = tag_assignments[example_id]
        if len(all_tags) == 0:
            all_tags = ["No Tag"]
        all_tags.append("All Dataset")
        sample_dict = {"id" : example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1}
        for tag in all_tags:
            categories_clip_scores[tag].append(sample_dict)
        
        sample_result_dict = {"text": True if text_correct(sample_dict) else False, "image": True if image_correct(sample_dict) else False, "group": True if group_correct(sample_dict) else False}
        result_dict[example_id] = sample_result_dict

    for category in categories_clip_scores:
        category_clip_scores = categories_clip_scores[category]
        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in category_clip_scores:
            text_correct_count += 1 if text_correct(result) else 0
            image_correct_count += 1 if image_correct(result) else 0
            group_correct_count += 1 if group_correct(result) else 0

        denominator = len(category_clip_scores)
        winoground_text_score = text_correct_count/denominator
        winoground_image_score = image_correct_count/denominator
        winoground_group_score = group_correct_count/denominator

        metrics = {category + " text score": text_correct_count/denominator, 
        category + " image score": image_correct_count/denominator,
        category + " group score": group_correct_count/denominator,
        "epoch": epoch}

        if not metrics:
            return metrics

        logging.info(
        f"winoground " + category + " Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)

            with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch})
    result_dict_path = os.path.join(args.logs,args.name,"winoground","results_" + str(epoch) + ".json")
    out_file = open(result_dict_path, "w") 
    json.dump(result_dict, out_file, indent = 6)
    out_file.close()
    return metrics

def evaluate_auxiliary(model,object_head,bb_head,random_rows,batch,args,epoch):
    metrics = {}
    if not is_master(args):
        return metrics
    autocast = get_autocast(args.precision)
    inv_trans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
                                transforms.Normalize(mean = [ -0.48145466, -0.4578275, -0.40821073 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    device = torch.device(args.device)
    model.eval()
    object_head.eval()
    bb_head.eval()
    random_rows.eval()
    vg_images, vg_texts, valid_objects, vg_bbs, object_descriptions, real_texts, text_lengths = batch
    real_texts = real_texts.flatten(0,1).tolist()
    real_texts = [[c for c in s if c !=36] for s in real_texts]
    real_texts = [''.join(chr(i) for i in L) for L in real_texts]
    real_texts = [l for l in real_texts if l != ""]
    vg_object_descriptions = object_descriptions[0,:valid_objects[0],:]
    for l in range (1 , object_descriptions.shape[0]):
        vg_object_descriptions = torch.cat([vg_object_descriptions,object_descriptions[l,:valid_objects[l],:]])
    real_texts.append("random_row")
    real_texts.append("no_object")
    vg_images = vg_images.to(device=device, non_blocking=True)
    vg_bbs = vg_bbs.to(device=device, non_blocking=True)
    vg_object_descriptions = vg_object_descriptions.to(device=device, non_blocking=True)
    with torch.no_grad():
        with autocast():
            _, extra_tokens, description_embeddings, logit_scale = model(vg_images, vg_object_descriptions)
            object_tokens = extra_tokens[:, : args.object_tokens,:]
            if args.distributed:
                description_embeddings = torch.cat([description_embeddings,random_rows.module.random_row,random_rows.module.no_object_row])
            else:
                description_embeddings = torch.cat([description_embeddings,random_rows.random_row,random_rows.no_object_row])
            bb_predictions = bb_head(object_tokens).sigmoid()
            label_embeddings = object_head(object_tokens)
            label_probs = logit_scale.mean() * label_embeddings @ description_embeddings.t().softmax(dim=-1)
            label_predictions = torch.argmax(label_probs, dim = -1)
            bb_predictions = box_cxcywh_to_xyxy(bb_predictions) * 224
    bb_predictions = bb_predictions.detach().cpu()
    label_predictions = label_predictions.detach().cpu()
    vg_images = vg_images.cpu()
    imgs_folder_path = os.path.join(args.checkpoint_path,f'bb_vis-{epoch}')
    if not os.path.exists(imgs_folder_path):
        os.mkdir(imgs_folder_path)
    for i in range(vg_images.shape[0]):
        full_image_path = os.path.join(imgs_folder_path, "img_bb_" + str(i) +  ".jpg")
        img = vg_images[i]
        img = inv_trans(img)
        img = torch.clamp(img,min=0.0,max=1.0)
        img = convert_image_dtype(img,torch.uint8)
        label_predictions_list = label_predictions[i].tolist()
        object_indexes = [j for j in range(len(label_predictions_list)) if label_predictions_list[j] != len(real_texts) - 1 and label_predictions_list[j] != len(real_texts) - 2]
        bb = bb_predictions[i,object_indexes,:]
        label = label_predictions[i,object_indexes]
        labels = [real_texts[i] for i in label.tolist()]
        bb_img = draw_bounding_boxes(img,bb,labels)
        new_image = transforms.ToPILImage()(bb_img)
        new_image.save(full_image_path)
    return metrics

     


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


class VsrDataset(Dataset):
    def __init__(self, blip_processor):
        super(VsrDataset).__init__()
        self.data = []
        with open('vsr/all_vsr_validated_data.jsonl', "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                self.data.append(j_line)
        self.blip_processor = blip_processor
        self.images_folder = "../../../../datasets/vsr/images"
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_number = sample["image"]
        image_path = os.path.join(self.images_folder,image_number)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.blip_processor(image)
        caption = sample["caption"]
        label = sample["label"]
        relation = sample["relation"]
        return image, caption, label, relation


def get_vsr_loader(dataset, batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1
    )
    return dataloader




def evaluate_vsr(clip_model, clip_processor, args):
    autocast = get_autocast(args.precision)
    clip_model.eval()
    vsr_dataset = VsrDataset(blip_processor=clip_processor)
    vsr_loader = get_vsr_loader(vsr_dataset,32)
    all_results_dict = {32:[], 31:[], 30:[], 29:[], 28:[], 27:[], 26:[], 25:[], 24:[], 23:[], 22:[], 21:[], 20:[]}
    all_relations = []
    device = args.device
    for batch in tqdm(vsr_loader):
        images, captions, labels, relations = batch
        images = images.to(device)
        labels = labels.to(device)
        captions = tokenize(captions).to(device=device,non_blocking=True)
        with autocast():
            # predict
            if args.distributed and not args.horovod:
                image_features, _ = clip_model.module.encode_image(images)
                text_features = clip_model.module.encode_text(captions)
            else:
                image_features, _ = clip_model.encode_image(images)
                text_features = clip_model.encode_text(captions)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            scores = torch.diagonal(100. * image_features @ text_features.t())
        for i in range(20,33):
            results = torch.eq(scores > i,labels)
            results = results.tolist()
            all_results_dict[i]+=results
        all_relations += relations

    rel2cat = {}
    cats = []
    rel_count = 0
    with open("vsr/rel_meta_category_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            cat, rels = line.strip().split(": ")
            cat = cat.strip()
            rel_list = rels.split(",")
            rel_count+=len(rel_list)
            rel_list = [rel.strip() for rel in rel_list]
            for rel in rel_list:
                rel2cat[rel] = cat
            cats.append(cat)
    print (f"# rel: {rel_count}")
    cats = list(set(cats))
    for j in range(20,33):
        results_by_cat = {}
        results_by_meta_cat = {"Adjacency":{"corrects":0,"samples":0}, "Directional":{"corrects":0,"samples":0}, "Orientation":{"corrects":0,"samples":0},
        "Projective":{"corrects":0,"samples":0},"Proximity":{"corrects":0,"samples":0},"Topological":{"corrects":0,"samples":0},"Unallocated":{"corrects":0,"samples":0}}
        for i in range(len(all_results_dict[j])):
            result = all_results_dict[j][i]
            relation = all_relations[i]
            if relation not in results_by_cat:
                if result == True:
                    results_by_cat[relation] = {"corrects":1,"samples":1}
                else:
                    results_by_cat[relation] = {"corrects":0,"samples":1}  
            else:
                results_by_cat[relation]["samples"] +=1
                if result == True:
                    results_by_cat[relation]["corrects"] +=1
            results_by_cat[relation]["accuracy"] = results_by_cat[relation]["corrects"]/results_by_cat[relation]["samples"]
            if relation not in rel2cat:
                continue
            metacat = rel2cat[relation]
            results_by_meta_cat[metacat]["samples"] +=1
            if result == True:
                results_by_meta_cat[metacat]["corrects"] +=1
            results_by_meta_cat[metacat]["accuracy"] = results_by_meta_cat[metacat]["corrects"]/results_by_meta_cat[metacat]["samples"]
        
    return results_by_cat, results_by_meta_cat
