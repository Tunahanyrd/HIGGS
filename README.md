# Higgs Boson Classification with Deep Neural Network and XGBoost Stacking

This project applies deep learning and ensemble techniques to the HIGGS dataset — a large-scale binary classification benchmark from the UCI Machine Learning Repository.

We combine a residual-based neural network with XGBoost stacking to improve classification accuracy and robustness.

Pretrained MLP models are also included, trained on 1M, 5.5M, and 11M samples.

---

##  Features

- Residual DNN with PyTorch
- Custom dataset handling with sample weights
- OneCycleLR scheduler, EarlyStopping, AUC evaluation
- ROC, PRC, Confusion Matrix, and Classification Report plots
- K-Fold stacking with learned DNN features + raw input features
- Final classifier: XGBoost (only for 1M dataset due to memory limits)
- 5.5M and 11M models include only DNN output (no stacking)

---

##  Dataset

- File: [`HIGGS.csv.gz`](https://archive.ics.uci.edu/dataset/280/higgs) from the UCI Machine Learning Repository
- 11 million rows, 28 columns (1 label + 1 event weight + 27 features)
-  Note: Original weights are not scaled, so AMS metric is not evaluated in this project

---

##  Project Structure

All logic is contained in `main.py`, including:

- Data loading & preprocessing
- Residual MLP model definition
- Training loop
- Evaluation & plotting
- K-Fold feature extraction
- XGBoost stacking

---

##  Performance (Validation AUC)

| Dataset Size | DNN Only | XGBoost Stacked |
|--------------|----------|-----------------|
| 1M           | ~0.82    | ~0.868          |
| 5.5M         | ~0.868   | Not available   |
| 11M          | ~0.8658  | Not available   |

>  `5.5e6.pt` (with data parallel ver.) is the best checkpoint for DNN-only training.  
>  K-Fold stacking for 5.5M and 11M datasets could not be completed.  
>  The Linux Kernel OOM Killer has been hunting this project like a ghost in the final fold.

---

##  Requirements

```bash
pip install -r requirements.txt

```
_Python 3.9+_

---
#  Usage
```bash
python main.py
```
Adjust nrows in the pd.read_csv(...) line of main.py to switch dataset size.

Model checkpoints are saved automatically to disk after training.

> This was my first stacking attempt. Honestly, I kind of messed it up —  
> partially because of inexperience, partially because the OOM killer wouldn’t leave me alone.

---
> MIT License.
Feel free to use, fork, or cite.

> “When we had no computers, we had no programming problem either.
When we had a few computers, we had a mild programming problem.
Confronted with machines a million times as powerful,
we are faced with a gigantic programming problem.”
— Edsger W. Dijkstra
