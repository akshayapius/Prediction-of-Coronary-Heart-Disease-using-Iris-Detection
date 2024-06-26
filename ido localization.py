import cv2
import numpy as np

def localize_iris_ido(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection (e.g., Canny edge detector)
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  # Adjust threshold as needed
    
    # Get the largest contour (assuming it corresponds to the iris)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Find the minimum enclosing circle for the largest contour
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    return center, radius

# Load the iris image
image = cv2.imread("S1002L01.jpg")

# Localize the iris using IDO method
center, radius = localize_iris_ido(image)

# Draw the detected iris boundary on the image
cv2.circle(image, center, radius, (0, 255, 0), 2)

# Display the image with the detected iris boundary
cv2.imshow('Detected Iris Boundary', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
