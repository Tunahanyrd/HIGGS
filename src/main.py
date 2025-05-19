#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:06:08 2025

@author: tunahan
HIGGS dataset
"""
# =============================================================================
# İmporting library
# =============================================================================
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# =============================================================================
# Loading data and definition variable 
# =============================================================================
num_workers = max(2, os.cpu_count() // 2) - 1

# https://archive.ics.uci.edu/dataset/280/higgs

df = pd.read_csv("../data/HIGGS.csv", header=None)
        
labels = df.iloc[:, 0]

weights = df.iloc[:, 1]

features = df.iloc[:, 2:]


corr = df.iloc[:, 1:].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# =============================================================================
# Creating pytorch dataset
# =============================================================================
class CustomDataset(Dataset):
    def __init__(self, features, labels, weights):
        self.X = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
        self.w = torch.tensor(weights.values, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.w[index]

batch_size = 4096
dataset = CustomDataset(features, labels, weights)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

# =============================================================================
# Creating the model and making the necessary adjustments
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    def forward(self, x):
        return F.relu(x + self.block(x))
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(27,256)
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.out = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )
    def forward(self, x, return_features = False):
        x = F.relu(self.input_layer(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        if return_features: # for meta model
            return x
        return self.out(x)
lr = 5e-4 * (batch_size / 256)
EPOCHS = 30
model = Model()
model = nn.DataParallel(model).to("cuda")
loss_fn = nn.BCEWithLogitsLoss(reduction="none")
optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                          steps_per_epoch=len(train_loader), 
                                          pct_start=0.1, anneal_strategy="cos",
                                          total_steps=EPOCHS)
criterion = nn.BCEWithLogitsLoss()
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4, save_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_auc = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def step(self, val_auc, model):
        if self.best_auc is None or val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            torch.save(model.state_dict(), self.save_path)
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"Counter: {self.counter}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
        
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y, batch_w in tqdm(dataloader, desc="Train Model:"):
        batch_X = batch_X.to("cuda")
        batch_y = batch_y.float().to("cuda")  
        batch_w = batch_w.to("cuda")

        optimizer.zero_grad()
        outputs = model(batch_X)  # logits
        losses = loss_fn(outputs, batch_y)
        loss = (losses * batch_w).mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()
        correct += (preds == batch_y.long()).sum().item()
        total += batch_y.size(0)

    avg_loss = running_loss / total
    accuracy = (correct / total) * 100
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y, batch_w in tqdm(dataloader, desc="Eval Model:"):
            batch_X = batch_X.to("cuda")
            batch_y = batch_y.float().to("cuda")
            batch_w = batch_w.to("cuda")
            outputs = model(batch_X)
            
            losses = loss_fn(outputs, batch_y)
            loss = (losses * batch_w).mean() 
            running_loss += loss.item() * batch_X.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

            correct += (preds == batch_y.long()).sum().item()
            total += batch_y.size(0)

    avg_loss = running_loss / total
    accuracy = (correct / total) * 100
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, auc

def test_model(model, dataloader):
    model.eval()
    all_preds, all_labels, all_probs, all_weights = [], [], [], []

    with torch.no_grad():
        for batch_X, batch_y, batch_w in tqdm(dataloader, desc="Test Model:"):
            batch_X = batch_X.to("cuda")
            batch_y = batch_y.to("cuda")
            
            outputs = model(batch_X)  # logits
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_weights.extend(batch_w.cpu().numpy())
    return all_preds, all_labels, all_probs, all_weights

train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_loss = float("inf")
early_stopper = EarlyStopping(patience=10)
for epoch in range(EPOCHS):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
    val_loss, val_acc, auc = evaluate(model, test_loader, criterion)
    scheduler.step()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print()
    print(f"===Epoch {epoch+1}/{EPOCHS}===")
    print(f"Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Val AUC: {auc:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    print()
    if early_stopper.step(auc, model):
        print("Early stopping in {epoch} epoch. Best model saved.")
        break
    
model.load_state_dict(torch.load("best_model 11m.pt"))
all_preds, all_labels, all_probs, all_weights = test_model(model, test_loader)

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_probs = np.concatenate(all_probs)
all_weights = np.concatenate(all_weights)
# =============================================================================
# Testing MLP model
# =============================================================================
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, auc, precision_recall_curve, average_precision_score
plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Train Loss", color='blue', linewidth=2)
plt.plot(val_losses, label="Validation Loss", color='orange', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(train_accs, label="Train Accs", color='blue', linewidth=2)
plt.plot(val_accs, label="Validation Accs", color='orange', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accs")
plt.title("Train vs Validation Accs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()        

class_names = ["Not Higgs", "Higgs"]
cm, roc_auc, roc_curve = confusion_matrix, roc_auc_score, roc_curve      

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Purples")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df = pd.DataFrame(report).transpose()
class_metrics = df.iloc[:, :4]
plt.figure(figsize=(10, len(class_metrics) * 0.6))
sns.heatmap(class_metrics, annot=True, fmt=".2f", cmap="Purples", linewidths=0.5)
plt.title("Classification Report Heatmap")
plt.ylabel("Classes")
plt.xlabel("Metrics")
plt.tight_layout()
plt.show()  

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()

precision, recall, _ = precision_recall_curve(all_labels, all_probs)
ap = average_precision_score(all_labels, all_probs)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f"AP = {ap:.2f}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid()
plt.tight_layout()
plt.show()
# =============================================================================
# Creating meta model
# =============================================================================
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def extract_features_kfold(model_class, full_dataset, original_features, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    X_features_total = []
    y_total = []

    labels_np = np.array([full_dataset[i][1].item() for i in range(len(full_dataset))])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np)):
        print(f"\nFold {fold + 1}/{n_splits}")

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=1024, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=1024, shuffle=False, num_workers=num_workers)

        model = model_class().to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(5):
            for batch_X, batch_y, _ in train_loader:
                batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda")
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        fold_features = []
        fold_labels = []

        with torch.no_grad():
            for batch_X, batch_y, _ in tqdm(val_loader, desc=f"Extract Fold {fold+1}"):
                batch_X = batch_X.to("cuda")
                features = model(batch_X, return_features=True)  # 256-dim mid layer
                fold_features.append(features.cpu().numpy())
                fold_labels.append(batch_y.cpu().numpy())

        X_dnn = np.vstack(fold_features)
        y_fold = np.vstack(fold_labels).reshape(-1)

        X_orig = original_features.iloc[val_idx].values  # 27 dim
        X_combined = np.hstack([X_dnn, X_orig])  # [256 + 27] = 283

        X_features_total.append(X_combined)
        y_total.append(y_fold)

    X_stack = np.vstack(X_features_total)
    y_stack = np.hstack(y_total)

    return X_stack, y_stack


X_stack, y_stack = extract_features_kfold(Model, dataset, features, n_splits=5)

xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.01,
                    eval_metric='logloss', verbosity=1, tree_method='hist')

xgb.fit(X_stack, y_stack)
    
xgb_preds = xgb.predict(X_stack)
xgb_probs = xgb.predict_proba(X_stack)[:, 1]

acc = accuracy_score(y_stack, xgb_preds)
auc = roc_auc_score(y_stack, xgb_probs)

print(f"\n K-Fold XGBoost Stacking Accuracy: {acc:.4f}")
print(f"K-Fold XGBoost Stacking AUC: {auc:.4f}")
    
    
    
    
    
    
    
    
    
    
