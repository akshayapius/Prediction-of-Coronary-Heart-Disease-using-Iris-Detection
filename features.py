import cv2
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import graycomatrix, graycoprops

def extract_first_order_features(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute mean, standard deviation, skewness, kurtosis, and entropy
    mean_value = np.mean(gray_image)
    std_dev_value = np.std(gray_image)
    skewness_value = skew(gray_image.flatten())
    kurtosis_value = kurtosis(gray_image.flatten())
    entropy_value = entropy(np.histogramdd(gray_image.flatten(), bins=256)[0].ravel())
    
    return mean_value, std_dev_value, skewness_value, kurtosis_value, entropy_value
def extract_glcm_features(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute GLCM
    distances = [1]  # Distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for GLCM computation
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Compute GLCM properties
    autocorrelation = graycoprops(glcm, 'autocorrelation').ravel()
    cluster_prominence = graycoprops(glcm, 'cluster_prominence').ravel()
    cluster_shade = graycoprops(glcm, 'cluster_shade').ravel()
    contrast = graycoprops(glcm, 'contrast').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    difference_entropy = graycoprops(glcm, 'difference_entropy').ravel()
    difference_variance = graycoprops(glcm, 'difference_variance').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    entropy = graycoprops(glcm, 'entropy').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    asm = graycoprops(glcm, 'ASM').ravel()
    
    return autocorrelation, cluster_prominence, cluster_shade, contrast, correlation, \
           difference_entropy, difference_variance, dissimilarity, energy, entropy, \
           homogeneity, asm

# Load the image
image = cv2.imread("HH_subsampled.jpg")

# Extract first-order statistical features
mean, std_dev, skewness, kurtosis, entropy = extract_first_order_features(image)

autocorrelation, cluster_prominence, cluster_shade, contrast, correlation, difference_entropy, difference_variance, dissimilarity, energy, entropy, homogeneity, asm = extract_glcm_features(image)

# Display or print the extracted features
print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Skewness:", skewness)
print("Kurtosis:", kurtosis)
print("Entropy:", entropy)
print("Autocorrelation:", autocorrelation)
print("Cluster Prominence:", cluster_prominence)
print("Cluster Shade:", cluster_shade)
print("Contrast:", contrast)
print("Correlation:", correlation)
print("Difference Entropy:", difference_entropy)
print("Difference Variance:", difference_variance)
print("Dissimilarity:", dissimilarity)
print("Energy:", energy)
print("Entropy:", entropy)
print("Homogeneity:", homogeneity)
print("ASM:", asm)