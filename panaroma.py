import cv2
import numpy as np
import streamlit as st
import os

# Define a function to create the panorama
def create_panorama(image1_path, image2_path):
    print("Image 1 path:", image1_path)
    print("Image 2 path:", image2_path)

    # Load the input images
    image2 = cv2.imread(image1_path)
    image1 = cv2.imread(image2_path)

    if image1 is None:
        print("Error loading image 1.")
        return None
    if image2 is None:
        print("Error loading image 2.")
        return None

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize feature detector
    detector = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    # Create a feature matcher
    bf = cv2.BFMatcher()

    # Match features between images
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image1 to image2's perspective
    result = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], image1.shape[0]))

    # Combine both images
    result[:, 0:image2.shape[1]] = image2

    return result


# Create a Streamlit app
st.title('Image Panorama App')

# Upload images
uploaded_image1 = st.file_uploader('Upload the first image', type=['jpg', 'jpeg', 'png'])
uploaded_image2 = st.file_uploader('Upload the second image', type=['jpg', 'jpeg', 'png'])


image1_path = os.path.join(os.getcwd(), "images", "my_image1.jpg")
image2_path = os.path.join(os.getcwd(), "images", "my_image2.jpg")


#if uploaded_image1 and uploaded_image2 is not None:
if uploaded_image1 and uploaded_image2:
    image1_path = 'image1.jpg'
    image2_path = 'image2.jpg'

    # Save the uploaded images to local files
    with open(image1_path, 'wb') as f:
        f.write(uploaded_image1.read())
    with open(image2_path, 'wb') as f:
        f.write(uploaded_image2.read())

    # Create the panorama
    panorama_result = create_panorama(image1_path, image2_path)

    if panorama_result is not None:
        # Display the panorama
        st.image(panorama_result, caption='Panorama', use_column_width=True)

        # Save the stitched image
        cv2.imwrite('panoramaS1_S2.jpeg', panorama_result)

st.write("Upload two images to create a panorama.")
