#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:05:25 2025

@author: ellie
"""

# finetune_sbert_contrastive.py
from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset, losses
from torch.utils.data import DataLoader
import pandas as pd
from siamese_main import splitData
import torch

def fine_tune_sbert(csv_path, output_path="models/finetuned_sbert", batch_size=64, epochs=4, warmup_steps=500, margin=0.5):
    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    print("device")
    # Load your CSV data.
    df = pd.read_csv(csv_path)
    df_train, _, _ = splitData(df)
    
    # Prepare a list of InputExample objects using your duplicate question pairs.
    # Make sure your CSV has columns: 'question1', 'question2', and 'is_duplicate'
    train_examples = []
    for _, row in df_train.iterrows():
        train_examples.append(
            InputExample(texts=[row['question1'], row['question2']],
                         label=float(row['is_duplicate']))
        )
    print("got training")
    
    # Load the pre-trained SBERT model.
    model_ft = SentenceTransformer('all-MiniLM-L6-v2')
    model_ft = model_ft.to(device)
    
    # Create the SentencesDataset for fine-tuning.
    train_dataset = SentencesDataset(train_examples, model_ft)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    print("got data")
    # Set up the ContrastiveLoss with a margin.
    # The margin parameter controls how far apart non-matching pairs are pushed.
    train_loss = losses.ContrastiveLoss(model_ft, margin=margin)
    
    # Fine-tune the model with the contrastive loss.
    model_ft.fit(train_objectives=[(train_dataloader, train_loss)],
                 epochs=epochs,
                 warmup_steps=warmup_steps,
                 output_path=output_path)
    
    print(f"Fine-tuned model using contrastive loss saved at: {output_path}")

if __name__ == "__main__":
    # Adjust the csv path accordingly.
    fine_tune_sbert("data/preprocessedquestions.csv")
