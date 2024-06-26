
import cv2
import numpy as np
import math
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import pickle


def rgb_to_grayscale(image):
    # Convert RGB image to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def localize_iris(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Use Hough Circle Transform to detect iris boundary
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=200, param2=20, minRadius=15, maxRadius=50)
    
    if circles is not None:
        # Convert circle parameters to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Assuming the largest circle corresponds to the iris
        (x, y, r) = circles[0]
        return x, y, r
    else:
        print("No iris found")
        return None

def crop_left_upper_iris(image, x, y, r):
    # Define the cropping region for left upper part of the iris
    x_start = x - (r*2)
    x_end = x
    y_start = y - (r*2)
    y_end = y - int(r/2)
    
    # Crop the left upper part of the iris
    left_upper_iris_roi = image[y_start:y_end, x_start:x_end]
    return left_upper_iris_roi

def crop_right_upper_iris(image, x, y, r):
    # Define the cropping region for right upper part of the iris
    x_start = x + int(r/2) + int(r/2) + int(r/2)  # Move towards the right
    x_end = x + (r*2)  # Increase the width
    y_start = y - int(r/2) - int(r/4)  # Move lower
    y_end = y + int(r/4)
    
    # Crop the right upper part of the iris
    right_upper_iris_roi = image[y_start:y_end, x_start:x_end]
    
    # Mark the cropped portion on the main image
    marked_image = image.copy()
    cv2.rectangle(marked_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    return right_upper_iris_roi#, marked_image
 


def rubber_sheet_normalization(iris_roi):
    # Perform rubber sheet normalization
    normalized_roi = cv2.equalizeHist(iris_roi)  # Example normalization, you can customize as needed
    return normalized_roi

def enhance_roi(roi):
    # Apply histogram equalization
    enhanced_roi = cv2.equalizeHist(roi)
    
    # Apply contrast stretching
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_roi = clahe.apply(enhanced_roi)
    
    # Apply adaptive histogram equalization
    enhanced_roi = cv2.equalizeHist(enhanced_roi)

    return enhanced_roi

def extract_texture_features(image):
    import numpy as np
    # Define GLCM properties
    distances = [1]  # distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles for texture comparison
    levels = 256  # 256 levels because the image is grayscale
    
    # Compute GLCM
    glcm = graycomatrix(image, distances, angles, levels=levels, symmetric=True, normed=True)
    
    # Extract texture features
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    
    return contrast, dissimilarity, homogeneity, energy, correlation

def normalize_contrast(contrast):
    # Min-max normalization
    min_val = np.min(contrast)
    max_val = np.max(contrast)
    normalized_contrast = (contrast - min_val) / (max_val - min_val)
    return normalized_contrast

def normalize_dissimilarity(dissimilarity_values):
    # Normalize dissimilarity values to range [1, 2]
    min_val = np.min(dissimilarity_values)
    max_val = np.max(dissimilarity_values)
    
    normalized_values = 1 + ((dissimilarity_values - min_val) / (max_val - min_val)) * (2 - 1)
    return normalized_values
def predict(img_path):
    iris_image = cv2.imread(img_path)
    # Convert RGB image to grayscale
    gray_image = rgb_to_grayscale(iris_image)

    # Localize the iris
    x, y, r = localize_iris(gray_image)

    if (x, y, r) is not None:
        # Crop the left upper part of the iris
        #left_upper_iris_roi = crop_left_upper_iris(gray_image, x, y, r)
        # Crop the right upper part of the iris
        #marked_image = crop_right_upper_iris(iris_image, x, y, r)
        right_upper_iris_roi = crop_right_upper_iris(gray_image, x, y, r)

        #cv2.imshow('Marked Image', marked_image)

        cv2.imwrite('roi.jpg',right_upper_iris_roi)
        #ROI enhancement
        enhanced=enhance_roi(right_upper_iris_roi)
        cv2.imwrite('enhanced_roi.jpg',enhanced)
        # Apply rubber sheet normalization
        normalized_roi = rubber_sheet_normalization(enhanced)
    contrast, dissimilarity, homogeneity, energy, correlation = extract_texture_features(normalized_roi)

    print("Contrast:", contrast[0])
    print("Dissimilarity:", dissimilarity[0])
    print("Homogeneity:", homogeneity[0])
    print("Energy:", energy[0])
    print("Correlation:", correlation[0])

    x=contrast[0]
    y=dissimilarity[0]
    z=homogeneity[0]
    l=energy[0]
    k=correlation[0]

    feature_names = ['Contrast', 'Dissimilarity', 'Homogenity', 'Energy', 'Correlation']

    test={
        'Contrast': [x],
        'Dissimilarity': [y],
        'Homogenity': [z],
        'Energy': [l],
        'Correlation': [k]
        }
    t = pd.DataFrame(test)

    with open('model.pkl', 'rb') as f:
        sc = pickle.load(f)
        y_pred=sc.predict(t)
        print(y_pred)
    #if img_path=='C:\Users\ACER\Dropbox\My PC (LAPTOP-PNE2C8NB)\Documents\baksha\S1002L01.jpg':
        #print("The Patient is CHD Positive")
        #return "The Patient is CHD Positive"
    if y_pred==1:
        print("The Patient is CHD Negative")
        return "The Patient is CHD Negative"
    else:
         print("The Patient is CHD Positive")
         return "The Patient is CHD Positive"



