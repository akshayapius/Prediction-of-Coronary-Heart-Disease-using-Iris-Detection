import cv2
import matplotlib.pyplot as plt

# Load the iris image
iris_image = cv2.imread("color eye.jpg")

# Convert RGB image to grayscale
gray_image = cv2.cvtColor(iris_image, cv2.COLOR_BGR2GRAY)

# Display the original RGB image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(iris_image, cv2.COLOR_BGR2RGB))
plt.title('Original RGB Image')
plt.axis('off')

# Display the grayscale image
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.show()
