import cv2
import os
import numpy as np
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Directory containing the images
input_dir = 'C:/Users/Pablo/Desktop/Light Distribution/input'
output_dir = 'C:/Users/Pablo/Desktop/Light Distribution/output'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
for file_name in image_files:
    img_path = os.path.join(input_dir, file_name)

    # Read and convert the image to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Display the original image
    display(Image.fromarray(img))

    # Calculate histogram of light intensities
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))

    # Display the histogram
    plt.figure()
    plt.title("Light Distribution Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Pixel Count")
    plt.plot(bin_edges[0:-1], histogram)  # bin_edges is one element longer than histogram
    plt.show()

    # Calculate basic light distribution statistics
    avg_intensity = np.mean(img)
    std_dev_intensity = np.std(img)
    skewness_intensity = skew(img.flatten())
    kurtosis_intensity = kurtosis(img.flatten())

    # Print light distribution statistics
    print(f"Image: {file_name}")
    print(f"Average Intensity: {avg_intensity}")
    print(f"Standard Deviation of Intensity: {std_dev_intensity}")
    print(f"Skewness of Intensity: {skewness_intensity}")
    print(f"Kurtosis of Intensity: {kurtosis_intensity}\n")

    # Save the light distribution data to a file
    light_data_path = os.path.join(output_dir, f'light_data_{file_name}.txt')
    with open(light_data_path, 'w') as file:
        file.write(f"Image: {file_name}\n")
        file.write(f"Average Intensity: {avg_intensity}\n")
        file.write(f"Standard Deviation of Intensity: {std_dev_intensity}\n")
        file.write(f"Skewness of Intensity: {skewness_intensity}\n")
        file.write(f"Kurtosis of Intensity: {kurtosis_intensity}\n")
