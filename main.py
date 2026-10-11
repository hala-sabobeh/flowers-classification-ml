import os
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import load_images, prepare_data
from models import (train_naive_bayes, analyze_naive_bayes,
                   train_decision_tree, train_mlp,
                   evaluate_model, visualize_decision_tree,
                   generate_comparison_report, create_confusion_matrix_plots)

DATA_PATH = "flowers"

def print_class_distribution(labels, class_names):
    counts = np.bincount(labels)
    total = sum(counts)
    print("\nClass Distribution:")
    for name, count in zip(class_names, counts):
        print(f"{name:>12}: {count:4d} samples ({count/total:.1%})")

def main():
    print("Flower Classification Project")
    print("=" * 40)
    print("Features: Raw pixels | Split: 80/20 | Models: NB, DT, MLP")
    print("=" * 40)
    
    try:
        # Load and prepare data
        print("\n[1/4] Loading dataset...")
        images, labels, class_names = load_images(DATA_PATH)
        print_class_distribution(labels, class_names)
        
        print("\n[2/4] Preprocessing images...")
        X_train, X_test, y_train, y_test = prepare_data(images, labels)
        print(f"Training shape: {X_train.shape}")
        print(f"Testing shape: {X_test.shape}")
        
        # Naive Bayes
        print("\n[3/4] Training Models...")
        nb_model = train_naive_bayes(X_train, y_train)
        analyze_naive_bayes(nb_model, class_names)
        evaluate_model(nb_model, X_test, y_test, class_names)
        
        # Decision Tree
        dt_model = train_decision_tree(X_train, y_train)
        evaluate_model(dt_model, X_test, y_test, class_names)
        
        
        # Neural Network
        mlp_model = train_mlp(X_train, y_train, hidden_layer_sizes=(32,), max_iter=100)
        evaluate_model(mlp_model, X_test, y_test, class_names)
        
        # Final comparison 
        print("\n[4/4] Generating Final Report...")
        generate_comparison_report(
            models=[nb_model, dt_model, mlp_model],
            model_names=["Naive Bayes", "Decision Tree", "Neural Network"],
            X_test=X_test,
            y_test=y_test,
            class_names=class_names
        )
        
        print("\nAll operations completed successfully!")
        print("Visualizations (plots) will show now...")
        
        # plot 
        dt_fig = visualize_decision_tree(dt_model, class_names, max_depth=3)
        if dt_fig:
            dt_fig.savefig("decision_tree.png")
            plt.show()  
        
        cm_fig = create_confusion_matrix_plots(
            [nb_model, dt_model, mlp_model],
            ["Naive Bayes", "Decision Tree", "Neural Network"],
            X_test, y_test, class_names
        )
        if cm_fig:
            cm_fig.savefig("model_comparison.png")
            plt.show() 
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if isinstance(e, FileNotFoundError):
            print("Please ensure:")
            print(f"1. The '{DATA_PATH}' folder exists")
            print("2. It contains subfolders: daisy, dandelion, rose, sunflower, tulip")
        print("\nDebug info:")
        print(f"Current dir: {os.getcwd()}")
        print(f"Data path: {os.listdir(DATA_PATH) if os.path.exists(DATA_PATH) else 'Not found'}")

if __name__ == "__main__":
    main()
