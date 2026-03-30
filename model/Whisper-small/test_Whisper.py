#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import pickle
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import logging
from transkv import cloverpca_whisper

logging.set_verbosity_error()

from utils import compute_pruned_dim

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [feature["input_features"] for feature in features]
        audio_lengths = [len(x) for x in input_features]

        padded_features = self.processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt"
        )
        input_features_pt = padded_features["input_features"]

        max_len = input_features_pt.shape[-1]
        attention_mask = torch.zeros((len(input_features), max_len), dtype=torch.long)
        for i, length in enumerate(audio_lengths):
            attention_mask[i, :length] = 1

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": input_features_pt,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def get_whisper_test_loader(processor, data_path, batch_size, num_workers):
    """
    构建 CommonVoice English 测试集 DataLoader
    """
    language = "en"
    split = "test"
    full_path = os.path.join(data_path, f"common_voice_17_0_{language}_{split}_16khz.pkl")

    if os.path.exists(full_path):
        print(f"Loading cached {split} dataset from {full_path}...")
        with open(full_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        print(f"Preparing {split} dataset for {language}...")
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", language, split=split, cache_dir=data_path)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        def prepare_dataset(batch):
            audio = batch["audio"]
            batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["labels"] = processor.tokenizer(batch["sentence"], truncation=True).input_ids
            return batch

        dataset = dataset.map(prepare_dataset, num_proc=num_workers, remove_columns=dataset.column_names)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            pickle.dump(dataset, f)

    collate_fn = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn
    )
    return loader

def evaluate_whisper(model, test_loader, device, processor):
    """
    评估模型在给定验证集上的 WER 和 CER 性能
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_features = batch["input_features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=225,
            )

            pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)

            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(labels_for_decode, skip_special_tokens=True)

            all_preds.extend(pred_str)
            all_labels.extend(label_str)

    wer = 100 * wer_metric.compute(predictions=all_preds, references=all_labels)
    cer = 100 * cer_metric.compute(predictions=all_preds, references=all_labels)
    
    return {"wer": wer, "cer": cer}

def main():
    parser = argparse.ArgumentParser(description="Test Whisper model on CommonVoice (Optional: Pruning)")
    
    parser.add_argument("--model_name", type=str, default="discoverylabs/whisper-small-english-finetuned", help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    parser.add_argument("--data_path", type=str, default="/data/CLOVER-main/data/", help="Directory to cache dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    parser.add_argument("--pca_path", type=str, default=None, help="Path to the PCA .pt file. If None, test original model.")
    parser.add_argument("--pruned_ratio", type=float, default=0.5, help="Ratio of heads to prune")
    parser.add_argument("--whisper_pruning_component", type=str, default="encoder", choices=["encoder", "decoder", "cross"], help="Which component to prune")
    
    args = parser.parse_args()

    print(f"Loading processor and original model '{args.model_name}' to {args.device}...")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    args.pruned_dim = compute_pruned_dim(model.model, args.pruned_ratio)
    
    model.generation_config.forced_decoder_ids = None
    model.generation_config.begin_suppress_tokens = []  
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.no_timestamps = True
    model.to(args.device)

    if args.pca_path is not None and args.pca_path.strip() != "":
        print(f"PCA save path provided: {args.pca_path}")
        print(f"Pruning model ({args.whisper_pruning_component}) to head dimension: {args.pruned_dim}...")
        model = cloverpca_whisper(model, args)
        model.to(args.device)
        print("Model pruning completed.")
    else:
        print("No PCA save path provided. Testing the original unpruned model.")

    print(f"Initializing CommonVoice test loader...")
    test_loader = get_whisper_test_loader(
        processor=processor,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print("Starting evaluation...")
    results = evaluate_whisper(model, test_loader, args.device, processor)

    print("\n" + "="*30)
    print("Evaluation Results:")
    print(f"Best WER: {results['wer']:.2f}")
    print(f"Best CER: {results['cer']:.2f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()