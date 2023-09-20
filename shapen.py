import cv2
import numpy as np
import csv

def sum_pooling(image, kernel_size):

    # Create a kernel of ones with the specified size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    
    # Use filter2D to apply the max pooling operation
    pooled_image = cv2.dilate(image, kernel, iterations=1)
    return pooled_image

# Load your image
input_image = cv2.imread('C:/Users/dzung/project/people-flow-modified/People-Flows-Modified/plot/24/034_24_hsv.jpg')

# Set the kernel size for the sum pooling operation
kernel_size = 11  # You can adjust this value

# Apply the sum pooling filter
sum_pooled_image = sum_pooling(input_image, kernel_size)

# Save the sum pooled image
cv2.imwrite('sum_pooled_image.jpg', sum_pooled_image)

# Define the CSV file path
csv_file = 'image_data.csv'

# Convert the image to a list of lists
sum_pooled_image = sum_pooled_image.tolist()

# Write the image data to a CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row (channel names)
    writer.writerow(['Red', 'Green', 'Blue'])
    
    # Write the pixel data
    for row in sum_pooled_image:
        writer.writerow(row)
