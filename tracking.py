# tracking.py
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

def log_metrics(y_test, y_pred, label_encoder, model_name="Model", log_file="model_scores.csv"):
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_test, y_pred)
    accuracy = np.mean(y_test == y_pred)
    macro_f1 = np.mean(f1s)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, model_name, accuracy, macro_f1, weighted_f1]

    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "accuracy", "macro_f1", "weighted_f1"])
        writer.writerow(row)
    
    print(f"\n✅ Logged metrics to {log_file}")
    return precisions, recalls, f1s, supports

def plot_f1_scores(label_encoder, f1s, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.bar(label_encoder.classes_, f1s, color='skyblue')
    plt.title("F1-Score per Genre")
    plt.xlabel("Genre")
    plt.ylabel("F1-Score")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📊 F1-score plot saved to {save_path}")
    plt.show()

def save_conf_matrix(y_test, y_pred, label_encoder, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("🎯 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ Confusion matrix saved to {save_path}")
    plt.show()
