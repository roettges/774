#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 12:29:32 2025

@author: ellie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import sys
import os
from datetime import datetime
import atexit
import pandas as pd
import ast

log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class SiameseDataset(Dataset):
    def __init__(self, df):
        """
        Assumes the DataFrame 'df' has precomputed embeddings stored in columns 'emb1' and 'emb2'
        and a label column 'is_duplicate'.
        """
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            # Directly pull precomputed embeddings from DataFrame.
            emb1 = row['question1_embedding']
            emb2 = row['question2_embedding']
            
            if isinstance(emb1, str):
                emb1 = ast.literal_eval(emb1)
            if isinstance(emb2, str):
                emb2 = ast.literal_eval(emb2)
            
            # Convert embeddings to torch tensors.
            emb1 = torch.from_numpy(np.array(emb1, dtype=np.float32))
            emb2 = torch.from_numpy(np.array(emb2, dtype=np.float32))

            label = torch.tensor([row['is_duplicate']], dtype=torch.float32)

            return emb1, emb2, label

        except Exception as e:
            print(f"[Data Error] Skipping index {idx}: {e}")
            # Try the next index, or loop to 0 if at end
            next_idx = (idx + 1) % len(self.df)
            return self.__getitem__(next_idx)

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        """
        embedding_dim: dimensionality of each input embedding.
        sim_feature_dim: dimension of the similarity feature (e.g., cosine similarity).
        hidden_dim: hidden layer dimension.
        """
        super().__init__()
        # The input features consist of: emb1, emb2, and their absolute difference.
        # If similarity features are used, add that as well.
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, emb1, emb2):
        diff = torch.abs(emb1 - emb2)
        concat = torch.cat((emb1, emb2, diff), dim=1)
        return self.fc(concat)

def evaluate(model, dataloader, device, dataset_name="Validation", test_mode=False):
    """
    Evaluates the model and computes metrics.
    If test_mode is False (default), computes accuracy and F1 score.
    If test_mode is True, computes additional metrics: precision, recall, ROC AUC, and confusion matrix.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            emb1, emb2, labels = batch

            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            labels = labels.to(device)

            outputs = model(emb1, emb2)
            batch_preds = outputs.squeeze().cpu().numpy().tolist()
            preds += batch_preds
            trues += labels.squeeze().cpu().numpy().tolist()

    # Convert predictions to binary using threshold of 0.5
    preds_binary = [1 if p > 0.5 else 0 for p in preds]
    
    # Common metrics
    accuracy = accuracy_score(trues, preds_binary)
    f1 = f1_score(trues, preds_binary)
    print(f"{dataset_name} Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # If test_mode is True, compute additional metrics
    if test_mode:
        precision = precision_score(trues, preds_binary)
        recall = recall_score(trues, preds_binary)
        try:
            roc_auc = roc_auc_score(trues, preds)
        except ValueError:
            roc_auc = float('nan')
        cm = confusion_matrix(trues, preds_binary)
        print(f"{dataset_name} Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"{dataset_name} ROC AUC: {roc_auc:.4f}")
        print(f"{dataset_name} Confusion Matrix:\n{cm}")
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    else:
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
    return metrics

def save_predictions(model, test_set, device, output_path, batch_size=512):
    """
    Evaluates the model on the test set, appends two new columns to the original DataFrame:
      - 'raw_prediction': continuous model output
      - 'predicted': binary prediction based on thresholding at 0.5
    Then saves the DataFrame to a CSV.
    """
    model.eval()
    raw_predictions = []
    binary_predictions = []
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in test_loader:
            emb1, emb2, _ = batch

            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            outputs = model(emb1, emb2)
            
            batch_raw = outputs.squeeze().cpu().numpy().tolist()
            batch_binary = [1 if p > 0.5 else 0 for p in batch_raw]
            
            raw_predictions.extend(batch_raw)
            binary_predictions.extend(batch_binary)
    
    test_set.df['raw_prediction'] = raw_predictions
    test_set.df['predicted'] = binary_predictions
    test_set.df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def train_siamese_gpt(df_train, df_val, df_test, device="cpu", epochs=50, batch_size=512, 
                  early_stopping_patience=5):
    # Dynamically determine the embedding dimension from the first row of the training DataFrame.
    sample_emb = df_train.iloc[0]['question1_embedding']
    if isinstance(sample_emb, str):
        sample_emb = ast.literal_eval(sample_emb)
    # If the embedding is stored as a list or numpy array:
    if isinstance(sample_emb, list):
        embedding_dim = len(sample_emb)
    elif isinstance(sample_emb, np.ndarray):
        embedding_dim = sample_emb.shape[0]
    else:
        raise ValueError("Embedding type not recognized. It should be a list or numpy array.")
    
    print(f"Determined embedding dimension: {embedding_dim}")

    # Create datasets without an encoder.
    train_set = SiameseDataset(df_train)
    val_set = SiameseDataset(df_val)
    test_set = SiameseDataset(df_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Pass the determined embedding dimension to the SiameseNet.
    model = SiameseNet(embedding_dim=embedding_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_f1 = 0.0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            emb1, emb2, labels = batch
            
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(emb1, emb2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        # Evaluate on validation set (test_mode=False)
        val_metrics = evaluate(model, val_loader, device, dataset_name="Validation", test_mode=False)
        print(f"Epoch {epoch+1} Validation Metrics: {val_metrics}")

        # Early stopping: if F1 improved, save the model and reset the counter
        if val_metrics['f1_score'] > best_f1 + 0.005:
            best_f1 = val_metrics['f1_score']
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print("Validation F1 improved. Saving best model...")
        else:
            epochs_no_improve += 1
            print(f"No improvement in F1 for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Load best model state before testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model loaded for final evaluation.")
    else:
        print("No improvement observed during training; using last epoch model.")

    # Save final model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/siamese_model_{log_time}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

    # Evaluate and print detailed metrics on test set (test_mode=True)
    test_metrics = evaluate(model, test_loader, device, dataset_name="Test", test_mode=True)

    # Save predictions to CSV for the test set
    output_csv = f"predictions_{log_time}.csv"
    save_predictions(model, test_set, device, output_csv)
    
    return test_metrics
