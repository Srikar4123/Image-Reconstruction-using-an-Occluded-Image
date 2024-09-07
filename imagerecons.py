import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import the cv2_imshow function from google.colab.patches

# Load the occluded image and the mask
# Update with the actual path to your image
image_path = '/547924_Generate the image of a boy standing on top of a h_xl-1024-v1-0.png'
occ_image_path = '/Untitled design (1).jpg'
# Update with the actual path to your mask
mask_path = '/Untitled design (2).jpg'

image = cv2.imread(image_path)
occ_image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Check if images loaded successfully
if image is None:
    print(f"Error: Could not load image from {image_path}")
if occ_image is None:
    print(f"Error: Could not load image from {occ_image_path}")
if mask is None:
    print(f"Error: Could not load mask from {mask_path}")

# Ensure the mask is binary
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Perform inpainting if images were loaded
if image is not None and mask is not None:
    #Removed occ_image as an argument as it is not needed for inpainting
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Display the original and inpainted images
    cv2_imshow(image)
    cv2_imshow(occ_image) # Use cv2_imshow instead of cv2.imshow
    cv2_imshow(mask)   # Use cv2_imshow instead of cv2.imshow
    cv2_imshow(inpainted_image) # Use cv2_imshow instead of cv2.imshow

    # Save the reconstructed image
    cv2.imwrite('reconstructed_image.jpg', inpainted_image)

    print("Image reconstruction completed and saved as 'reconstructed_image.jpg'")

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
