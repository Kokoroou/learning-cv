from pathlib import Path
from typing import Union

import cv2
import imutils
import numpy as np
from PIL import Image


def detect_edge(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """
    Detect edges in the input image.

    :param image: Raw image.
    :return: Edge detected image.
    """

    # Convert PIL image to OpenCV image
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny edge detection
    image = cv2.Canny(image, 50, 150)

    # Threshold the image
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    return image


def draw_edge(
    original_image: Union[Image.Image, np.ndarray], edge_image: np.ndarray
) -> np.ndarray:
    """
    Draw edges on the original image.

    :param original_image: Original image.
    :param edge_image: Edge detected image.
    :return: Image with edges drawn.
    """

    # Convert PIL image to OpenCV image
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Convert image to grayscale
    edge_image = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

    # Draw edges on the original image
    image = cv2.addWeighted(original_image, 0.7, edge_image, 0.3, 0)

    return image


if __name__ == "__main__":
    data_dir = Path(
        "D:/OneDrive - Hanoi University Of Pharmacy/Tuan Anh/Workspace/Projects/learning-cv"
    )
    img_path = data_dir / "downloads/beach-001.jpg"

    img = cv2.imread(str(img_path))
    detected_img = detect_edge(img)

    img = imutils.resize(img, height=500)
    detected_img = imutils.resize(detected_img, height=500)

    # Get pixel values of the flattened edge detected image
    print(set(detected_img.flatten()))

    # detected_img = draw_edge(img, detected_img)

    cv2.imshow("Original Image", img)
    cv2.imshow("Edge Detected Image", detected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
