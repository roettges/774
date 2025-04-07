#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:46:41 2025

@author: ellie
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

class SiameseDataset(Dataset):
    def __init__(self, df, encoder, use_sim_features=True):
        self.encoder = encoder
        self.df = df.reset_index(drop=True)
        self.use_sim_features = use_sim_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q1 = row['question1']
        q2 = row['question2']
        label = torch.tensor([row['is_duplicate']], dtype=torch.float32)

        emb1 = self.encoder.encode(q1, convert_to_tensor=True)
        emb2 = self.encoder.encode(q2, convert_to_tensor=True)

        # Optional handcrafted similarity features (e.g., Jaccard, cosine, etc.)
        if self.use_sim_features and 'sim_jaccard' in self.df.columns:
            sim_feats = torch.tensor(row[['sim_jaccard', 'sim_levenshtein']], dtype=torch.float32)
        else:
            sim_feats = None
            # sim_feats = torch.zeros(2)

        return emb1, emb2, sim_feats, label

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=384, sim_feature_dim=0, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 3 + sim_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, emb1, emb2, sim_feats):
        diff = torch.abs(emb1 - emb2)
        if sim_feats is not None:
            concat = torch.cat((emb1, emb2, diff, sim_feats), dim=1)
        else:
            concat = torch.cat((emb1, emb2, diff), dim=1)
        return self.fc(concat)

def train_siamese(df_train, df_val, df_test, device="cpu", epochs=10, batch_size=32, 
                  use_sim_features=True, early_stopping_patience=3):
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    train_set = SiameseDataset(df_train, encoder, use_sim_features=use_sim_features)
    val_set = SiameseDataset(df_val, encoder, use_sim_features=use_sim_features)
    test_set = SiameseDataset(df_test, encoder, use_sim_features=use_sim_features)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = SiameseNet(sim_feature_dim=2 if use_sim_features else 0).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_f1 = 0.0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for emb1, emb2, sim_feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Move data to device
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            # If sim_feats is None, create a dummy tensor of appropriate shape
            sim_feats = sim_feats.to(device) if sim_feats is not None else None
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(emb1, emb2, sim_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        # Evaluate on validation set
        val_acc, val_f1 = evaluate(model, val_loader, device, dataset_name="Validation")
        print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

        # Early stopping: if F1 improved, save the model and reset the counter
        if val_f1 > best_f1:
            best_f1 = val_f1
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
    else:
        print("No improvement observed during training; using last epoch model.")

    # Evaluate on test set
    test_acc, test_f1 = evaluate(model, test_loader, device, dataset_name="Test")
    print(f"Test Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")

def evaluate(model, dataloader, device, dataset_name="Validation"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for emb1, emb2, sim_feats, labels in dataloader:
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            sim_feats = sim_feats.to(device) if sim_feats is not None else None
            outputs = model(emb1, emb2, sim_feats)
            preds += outputs.squeeze().cpu().numpy().tolist()
            trues += labels.squeeze().cpu().numpy().tolist()

    # Convert continuous outputs to binary predictions (threshold=0.5)
    preds_binary = [1 if p > 0.5 else 0 for p in preds]
    acc = accuracy_score(trues, preds_binary)
    f1 = f1_score(trues, preds_binary)
    print(f"{dataset_name} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return acc, f1
