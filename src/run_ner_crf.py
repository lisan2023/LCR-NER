import glob
import logging
import math
import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup, BertTokenizer
from torch.optim import AdamW
from models.bert_for_ner import BertCrfForNer
from processors.utils_ner import get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from tools.finetuning_argparse import get_argparse

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, BertTokenizer),
}

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    if args.save_steps==-1 and args.logging_steps==-1:
        args.logging_steps=len(train_dataloader)
        args.save_steps = len(train_dataloader)
    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, tokenizer)
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir, safe_serialization=False)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    #eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['attention_mask'])
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = batch[4].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(args.id2label[tags[i][j]])
        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss

    info = "ALL" + "-".join([f' {key}: {value} ' for key, value in list(results.items())[3:]])
    logger.info(info)

    entity_order = ['PER', 'LOC', 'OFI', 'BOOK']
    entity_order = ['BOO', 'JOB', 'LOC', 'ORG', 'PER', 'WAR']

    for key in entity_order:
        if key in entity_info:
            info = key + "-".join([f' {k}: {v} ' for k, v in list(entity_info[key].items())[3:]])
            logger.info(info)

    logger.info("---------------------------------------")

    info = "ALL" + "-".join([f' {key}: {value} ' for key, value in list(results.items())[0:3]])
    logger.info(info)

    entity_order = ['PER', 'LOC', 'OFI', 'BOOK']
    entity_order = ['BOO', 'JOB', 'LOC', 'ORG', 'PER', 'WAR']
    for key in entity_order:
        if key in entity_info:
            info = key + "-".join([f' {k}: {v} ' for k, v in list(entity_info[key].items())[0:3]])
            logger.info(info)

    return results

def predict(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    # Ensure the per-checkpoint subdirectory exists when prefix is non-empty
    # (otherwise output_predict_file's parent dir is missing -> FileNotFoundError)
    if prefix and args.local_rank in [-1, 0]:
        target_dir = os.path.join(pred_output_dir, prefix)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []
    results2 = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    output_predict_file2 = os.path.join(pred_output_dir, prefix, "test_prediction.txt")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags  = tags.squeeze(0).cpu().numpy().tolist()

            preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
            true_labels = batch[3][0][1:-1].cpu().numpy().tolist()
            
            # Update metric
            metric.update(pred_paths=[preds], label_paths=[true_labels])
            
            label_entities = get_entities(preds, args.id2label, args.markup)
            json_d = {}
            json_d['id'] = step
            json_d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
            json_d['entities'] = label_entities
            results.append(json_d)

            tokens  = tokenizer.batch_decode(batch[0][0][1:-1], skip_special_tokens=True)
            true_labels_str = [args.id2label[x] for x in true_labels]
            pred_labels = [args.id2label[x] for x in preds]
            output_lines = []
            for x, y, z in zip(tokens, true_labels_str, pred_labels):
                if y != z:
                    output_lines.append(f"{x} {y} {z} #ERROR\n")
                else:
                    output_lines.append(f"{x} {y} {z}\n")

            results2.append("".join(output_lines))
            pbar(step)
    logger.info("\n")

    # Get evaluation results
    eval_info, entity_info = metric.result()

    # Count total processed lines
    total_lines = sum(len(record.split('\n')) for record in results2)
    logger.info("***** Total processed lines: %d *****", total_lines)

    info = "ALL" + "-".join([f' {key}: {value} ' for key, value in list(eval_info.items())[3:]])
    logger.info(info)
    entity_order = ['PER', 'LOC', 'OFI', 'BOOK']
    entity_order = ['BOO', 'JOB', 'LOC', 'ORG', 'PER', 'WAR']
    #
    for key in entity_order:
        info = key + "-".join([f' {k}: {v} ' for k, v in list(entity_info.get(key, {}).items())[3:]])
        logger.info(info)

    logger.info("---------------------------------------")

    info = "ALL" + "-".join([f' {key}: {value} ' for key, value in list(eval_info.items())[0:3]])
    logger.info(info)
    entity_order = ['PER', 'LOC', 'OFI', 'BOOK']
    entity_order = ['BOO', 'JOB', 'LOC', 'ORG', 'PER', 'WAR']
    for key in entity_order:
        info = key + "-".join([f' {k}: {v} ' for k, v in list(entity_info.get(key, {}).items())[3:]])
        logger.info(info)
    
    
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    with open(output_predict_file2, "w", encoding="utf-8") as writer:
        for record in results2:
            writer.write(record + '\n')
    if args.task_name == 'cluener':
        output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
        test_text = []
        with open(os.path.join(args.data_dir,"test.json"), 'r') as fr:
            for line in fr:
                test_text.append(json.loads(line))
        test_submit = []
        for x, y in zip(test_text, results):
            json_d = {}
            json_d['id'] = x['id']
            json_d['label'] = {}
            entities = y['entities']
            words = list(x['text'])
            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d['label']:
                        if word in json_d['label'][tag]:
                            json_d['label'][tag][word].append([start, end])
                        else:
                            json_d['label'][tag][word] = [[start, end]]
                    else:
                        json_d['label'][tag] = {}
                        json_d['label'][tag][word] = [[start, end]]
            test_submit.append(json_d)
        json_to_text(output_submit_file,test_submit)


def predict_nbest(args, model, tokenizer, prefix=""):
    """N-best prediction: decode top-N paths via CRF and dump with scores."""
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, data_type='test')
    test_sampler = SequentialSampler(test_dataset) \
        if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=1, collate_fn=collate_fn)

    nbest = args.nbest
    logger.info("***** Running n-best prediction %s (nbest=%d) *****",
                prefix, nbest)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    results = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting-NBest")
    if isinstance(model, nn.DataParallel):
        model = model.module

    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        logits = outputs[0]

        # CRF viterbi -> (nbest, 1, L) paths, (nbest, 1) joint scores
        paths, scores = model.crf.decode_nbest_with_score(
            logits, inputs['attention_mask'], nbest=nbest)
        # detach: scores carry grad (built from start/end/transition Parameters)
        paths  = paths.detach().squeeze(1).cpu().numpy().tolist()       # (nbest, L)
        scores = scores.detach().squeeze(1).cpu().numpy().tolist()      # (nbest,)

        # softmax(score - max) -> relative probability within the N-best list
        max_s = max(scores)
        exp_scores = [math.exp(s - max_s) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        input_ids  = batch[0].cpu().numpy().tolist()[0]
        input_mask = batch[1].cpu().numpy().tolist()[0]
        # Real token count from attention_mask - 2 ([CLS]/[SEP]); more robust than batch[4] semantics.
        L = max(sum(input_mask) - 2, 0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[1:1 + L]) if L > 0 else []

        # Per-sample emission matrix (used only to fill per-token scores below;
        # not serialized as a top-level field).
        emissions_np     = logits.detach().squeeze(0).cpu().numpy()                # [L_full, K]
        transitions_np   = model.crf.transitions.detach().cpu().numpy()            # [K, K]
        start_trans_np   = model.crf.start_transitions.detach().cpu().numpy()        # [K]
        # Format helper: 4 significant figures.
        sig4 = lambda x: f"{float(x):.4g}"

        # Gold tags for this sample (batch[3] = labels per custom collate_fn).
        gold_label_ids = batch[3][0].cpu().numpy().tolist()[1:1 + L]
        gold_tag_strs = [args.id2label[t] for t in gold_label_ids]
        # Gold entity spans, formatted the same way as paths[i].entities.
        gold_spans = get_entities(gold_label_ids, args.id2label, args.markup)
        gold_entity_strs = [
            "".join(tokens[s:e + 1]) + ":" + ent_type
            for ent_type, s, e in gold_spans
        ]

        # One record per sentence; nbest paths nested inside.
        path_records = []
        for rank, (path_ids, score, prob) in enumerate(
                zip(paths, scores, probs)):
            tag_ids = path_ids[1:1 + L]
            tags = [args.id2label[i] for i in tag_ids]
            spans = get_entities(tag_ids, args.id2label, args.markup)
            entity_strs = [
                "".join(tokens[s:e + 1]) + ":" + ent_type
                for ent_type, s, e in spans
            ]

            # Per-token breakdown for this path:
            #   "token:tag:emission_score:transition_score"
            # emission_score[i]   = E[i, tag_ids[i]]              (BERT's logit at chosen tag)
            # transition_score[i] = A[tag_ids[i-1], tag_ids[i]]    (incoming transition into position i)
            #   - at i == 0, use start_T[tag_ids[0]] since no predecessor exists
            tags_with_score = []
            for i, (tok, t) in enumerate(zip(tokens, tag_ids)):
                emission = emissions_np[i + 1][t]
                if i == 0:
                    trans = start_trans_np[t]                   # A[start -> y_0]
                else:
                    trans = transitions_np[tag_ids[i - 1]][t]  # A[y_{i-1} -> y_i]
                tags_with_score.append(
                    f"{tok} | {args.id2label[t]} | {sig4(emission)} | {sig4(trans)}"
                )

            # Per-path entity-level P/R/F1 vs gold (via existing SeqEntityScore).
            # SeqEntityScore treats the precision key as "acc" in its result dict.
            metric = SeqEntityScore(args.id2label, markup=args.markup)
            metric.update(label_paths=[gold_tag_strs], pred_paths=[tags])
            prf_overall, _ = metric.result()
            prf = {
                "precision": prf_overall["acc"],
                "recall":    prf_overall["recall"],
                "f1":        prf_overall["f1"],
            }

            path_records.append({
                "rank":            rank,
                "tags":            " ".join(tags),
                "score":           float(score),
                "prob":            float(prob),
                "entities":        entity_strs,
                "prf":             prf,
                "tags_with_score": tags_with_score,    # one "tok:tag:E:T" per token
            })

        # Rank within this sample's top-N that has the highest F1 (tie -> first hit).
        best_path_idx = max(range(len(path_records)),
                            key=lambda i: path_records[i]["prf"]["f1"])
        # Also expose a flat summary of per-path PRFs at the sample level.
        path_prfs = [rec["prf"] for rec in path_records]

        results.append({
            "id":            step,
            "tokens":        " ".join(tokens),
            "gold_entities": gold_entity_strs,
            "gold_tags":     " ".join(gold_tag_strs),   # full BIO tag sequence for downstream analysis
            "best_path":     best_path_idx,    # rank of path with highest F1
            "prfs":          path_prfs,        # length-nbest, mirrors paths[i].prf
            "paths":         path_records,
        })
        pbar(step)

    logger.info("\n")
    output_file = os.path.join(pred_output_dir, args.nbest_output_file)
    if args.local_rank in [-1, 0]:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "task_name":    args.task_name,
                "model_name":   args.model_name_or_path,
                "nbest":        nbest,
                "num_examples": len(results),
                "predictions":  results,
            }, f, ensure_ascii=False, indent=2)
        logger.info("N-best predictions saved to %s", output_file)


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_crf-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                    else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def main():
    args = get_argparse().parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    #args.output_dir = args.output_dir + '_{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path,num_labels=num_labels,)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        checkpoints = [args.model_name_or_path]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)

    if args.do_predict_nbest and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        prefix = args.model_name_or_path.split('/')[-1] if args.model_name_or_path.find('checkpoint') != -1 else ""
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.to(args.device)
        predict_nbest(args, model, tokenizer, prefix=prefix)


if __name__ == "__main__":
    main()
