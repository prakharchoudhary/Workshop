"""
This script involves reading a live image feed and applying 
various CV filters to cartoonify the image feed.
"""

import cv2
import numpy as np

def outlineRect(image, rect, color):
    """used to draw a rectangle"""
    if rect is None:
        return
    X, y, w, h = [int(i) for i in rect]
    cv2.rectangle(image, (X, y), (X + w, y + h), color)

def widthHeightDividedBy(image, divisor):
    """Return an image's dimensions, divided by a value."""
    h, w = image.shape[:2]
    return (int(w / divisor), int(h / divisor))

def cartoonize_image(img, ds_factor=4, sketch_mode=False, detect_features=False, clf=None):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median filter to the grayscale image
    img_gray = cv2.medianBlur(img_gray, 7)

    # Detect edges in the image and threshold it
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

    # Detect faces
    if clf is not None:
        faces = []
        minSize = widthHeightDividedBy(img, 8)
        faceRects = clf.detectMultiScale(img,
        	scaleFactor, minNeighbors, flags, minSize
        	)

        if faceRects is not None:
            for faceRect in faceRects:
                faces.append(faceRect)

        for face in faces:
            outlineRect(img, face, (0,0,255))

    # 'mask' is the sketch of the image
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Resize the image to a smaller size for faster computation
    img_small = cv2.resize(img, None, fx=1.0 / ds_factor, fy=1.0 / ds_factor,
                           interpolation=cv2.INTER_AREA)
    num_repititions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5

    # Apply bilateral filter the image multiplies times
    for i in range(num_repititions):
        img_small = cv2.bilateralFilter(
            img_small, size, sigma_color, sigma_space)

    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor,
                            interpolation=cv2.INTER_LINEAR)
    dst = np.zeros(img_gray.shape)

    # Add the thick boundary lines to the image using 'AND' operator
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    cur_char = -1
    prev_char = -1
    faceClassifier = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    scaleFactor=1.2
    minNeighbors=2
    flags=cv2.CASCADE_SCALE_IMAGE
    state = False

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.95, fy=0.95,
                           interpolation=cv2.INTER_AREA)
        c = cv2.waitKey(1)
        if c == 27:
            break

        if c > -1 and c != prev_char:
            cur_char = c
        prev_char = c

        # displaying the image in sketch mode
        if cur_char == ord('s'):
            cv2.imshow('Cartoonize', cartoonize_image(
                frame, sketch_mode=True))
            state = True

        # displaying the cartoonified image
        elif cur_char == ord('c'):
            cv2.imshow('Cartoonize', cartoonize_image(
                frame, sketch_mode=False))
            state = False

        # encase the face in a box
        elif cur_char == ord('f'):
        	cv2.imshow('Cartoonize', cartoonize_image(
        		frame, sketch_mode=state, clf=faceClassifier))
        else:
            cv2.imshow('Cartoonize', frame)

    cap.release()
    cv2.destroyAllWindows()