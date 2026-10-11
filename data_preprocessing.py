import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images(base_path, target_size=(32, 32)):  
    """Load and preprocess flower images from subfolders"""
    categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    images = []
    labels = []
    
    for label, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, target_size)
                image = image / 255.0  
                images.append(image)
                labels.append(label)
    
    return np.array(images), np.array(labels), categories

def extract_pixel_features(images):
    """Convert images to pixel feature vectors"""
    return images.reshape(images.shape[0], -1)  

def prepare_data(images, labels, test_size=0.2, random_state=42):
    """Split data into train/test sets and extract features"""
    # pixel features
    features = extract_pixel_features(images)
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    return X_train, X_test, y_train, y_test
