from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score,
    recall_score, f1_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def analyze_naive_bayes(model, class_names):
    print("\nNaive Bayes Feature Statistics:")
    try:
        print(f"{'Class':<12} {'Mean':<25} {'Variance':<25}")
        print("-" * 60)
        for i, name in enumerate(class_names):
            print(f"{name:<12} {str(model.theta_[i][:3].round(4)) + '...':<25} {str(model.var_[i][:3].round(4)) + '...':<25}")
        print(f"\nShowing first 3 of {model.theta_.shape[1]} features")
    except Exception as e:
        print(f"Could not analyze model: {str(e)}")

def train_decision_tree(X_train, y_train, max_depth=5):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def visualize_decision_tree(model, class_names, max_depth=2):
    plt.figure(figsize=(25, 15))
    try:
        n_features = model.n_features_in_
        feature_names = [f"pixel_{i}" for i in range(n_features)]
        plot_tree(
            model,
            filled=True,
            rounded=True,
            class_names=class_names,
            feature_names=feature_names,
            max_depth=max_depth,
            fontsize=8,
            impurity=False
        )
        plt.title(f"Decision Tree (First {max_depth} Levels)")
        plt.tight_layout()
        fig = plt.gcf()
        return fig
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        print("\nText representation of the tree:")
        print(export_text(model, max_depth=max_depth))
        return None

def train_mlp(X_train, y_train, hidden_layer_sizes=(32,), max_iter=100):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True
    )
    model.fit(X_train, y_train)
    return model

def custom_classification_report(y_true, y_pred, class_names):
    """Prints per-class metrics only (no macro avg or weighted avg)."""
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )
    print(f"{'Class':>12} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>8}")
    for i, cname in enumerate(class_names):
        print(f"{cname:>12} {precision[i]:10.2f} {recall[i]:10.2f} {f1[i]:10.2f} {support[i]:8d}")

def evaluate_model(model, X_test, y_test, class_names):
    try:
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        custom_classification_report(y_test, y_pred, class_names)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        return y_pred
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None

def print_model_results(name, y_true, y_pred, class_names):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print("=" * 30)
    print(f" Results: {name}")
    print("=" * 30)
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}\n")
    return {'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

def generate_comparison_report(models, model_names, X_test, y_test, class_names):
    results = []
    print("\n============================")
    print("    Individual Model Results")
    print("============================")
    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        custom_classification_report(y_test, y_pred, class_names)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        res = print_model_results(name, y_test, y_pred, class_names)
        results.append(res)

    print('-' * 54)
    print('|{:^16}|{:^9}|{:^10}|{:^8}|{:^7}|'.format(
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1'))
    print('-' * 54)
    for r in results:
        print('|{:^16}|{:^9.3f}|{:^10.3f}|{:^8.3f}|{:^7.3f}|'.format(
            r['Model'], r['Accuracy'], r['Precision'], r['Recall'], r['F1']))
    print('-' * 54)
    best = max(results, key=lambda x: x['Accuracy'])
    print(f"\nBest Model: {best['Model']} (Accuracy: {best['Accuracy']:.3f})\n")

def create_confusion_matrix_plots(models, model_names, X_test, y_test, class_names):
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    for idx, (model, name) in enumerate(zip(models, model_names)):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)
        disp.plot(ax=axes[idx])
        axes[idx].set_title(name)
    plt.tight_layout()
    return fig
