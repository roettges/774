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

def train_siamese(df_train, df_val, device="cpu", epochs=5, batch_size=32, use_sim_features=True):
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    train_set = SiameseDataset(df_train, encoder, use_sim_features=use_sim_features)
    val_set = SiameseDataset(df_val, encoder, use_sim_features=use_sim_features)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = SiameseNet(sim_feature_dim=2 if use_sim_features else 0).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for emb1, emb2, sim_feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            emb1, emb2, sim_feats, labels = emb1.to(device), emb2.to(device), sim_feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(emb1, emb2, sim_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        evaluate(model, val_loader, device)

def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for emb1, emb2, sim_feats, labels in dataloader:
            emb1, emb2, sim_feats = emb1.to(device), emb2.to(device), sim_feats.to(device)
            outputs = model(emb1, emb2, sim_feats)
            preds += outputs.squeeze().cpu().numpy().tolist()
            trues += labels.squeeze().cpu().numpy().tolist()

    preds_binary = [1 if p > 0.5 else 0 for p in preds]
    acc = accuracy_score(trues, preds_binary)
    f1 = f1_score(trues, preds_binary)
    print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
