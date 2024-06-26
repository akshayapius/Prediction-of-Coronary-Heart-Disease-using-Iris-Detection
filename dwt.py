import cv2
import numpy as np
import pywt

def dwt_split(image):
    # Convert image to float32 for DWT
    image = np.float32(image)
    
    # Perform 2D Discrete Wavelet Transform
    coeffs = pywt.dwt2(image, 'haar')
    
    # Get the approximation (low-pass) and detail (high-pass) coefficients
    LL, (LH, HL, HH) = coeffs
    
    # Subsampled images
    LL_subsampled = cv2.resize(LL, (0,0), fx=0.5, fy=0.5)  # Approximation
    LH_subsampled = cv2.resize(LH, (0,0), fx=0.5, fy=0.5)  # Horizontal detail
    HL_subsampled = cv2.resize(HL, (0,0), fx=0.5, fy=0.5)  # Vertical detail
    HH_subsampled = cv2.resize(HH, (0,0), fx=0.5, fy=0.5)  # Diagonal detail
    
    return LL_subsampled, LH_subsampled, HL_subsampled, HH_subsampled

# Load the image
image = cv2.imread("enhanced_roi.jpg", cv2.IMREAD_GRAYSCALE)

# Perform DWT and split into four subsampled images
LL, LH, HL, HH = dwt_split(image)

# Display or save the subsampled images
cv2.imshow('LL (Approximation)', LL)
cv2.imshow('LH (Horizontal Detail)', LH)
cv2.imshow('HL (Vertical Detail)', HL)
cv2.imshow('HH (Diagonal Detail)', HH)

# Save the subsampled images
cv2.imwrite('LL_subsampled.jpg', LL)
cv2.imwrite('LH_subsampled.jpg', LH)
cv2.imwrite('HL_subsampled.jpg', HL)
cv2.imwrite('HH_subsampled.jpg', HH)

cv2.waitKey(0)
cv2.destroyAllWindows()
