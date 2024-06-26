from skimage.feature import greycomatrix, greycoprops

def extract_texture_features(image):
    # Define GLCM properties
    distances = [1]  # distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles for texture comparison
    levels = 256  # 256 levels because the image is grayscale
    
    # Compute GLCM
    glcm = greycomatrix(image, distances, angles, levels=levels, symmetric=True, normed=True)
    
    # Extract texture features
    contrast = greycoprops(glcm, 'contrast').ravel()
    dissimilarity = greycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = greycoprops(glcm, 'homogeneity').ravel()
    energy = greycoprops(glcm, 'energy').ravel()
    correlation = greycoprops(glcm, 'correlation').ravel()
    
    return contrast, dissimilarity, homogeneity, energy, correlation

# Extract texture features from the normalized ROI
contrast, dissimilarity, homogeneity, energy, correlation = extract_texture_features(normalized_roi)

print("Contrast:", contrast)
print("Dissimilarity:", dissimilarity)
print("Homogeneity:", homogeneity)
print("Energy:", energy)
print("Correlation:", correlation)
