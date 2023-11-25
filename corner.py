import cv2
import numpy as np
"""https://stackoverflow.com/questions/55039229/how-to-find-corners-of-an-object-using-opencv"""

def find_centroids(dst, gray):
    """return the central corner based on cornerHarris output"""
    # Threshold for an optimal value, it may vary depending on the image.
    ret, dst = cv2.threshold(dst, 0.0001 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 
                0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids[1:]),(5,5), 
          (-1,-1),criteria)
    return corners



def coord_corners (image):
    """return pixelmatrix of corners, coordenates of the coorner's shapes in the image in format of np.array and list"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    gray = cv2.GaussianBlur(gray,(11,11),0)

    dst = cv2.cornerHarris(gray, 3, 3, 0.04)

    dst = cv2.dilate(dst, None)

    corners = find_centroids(dst, gray)
    corners_list = []
    # To draw the corners   
    for corner in corners:
        image[int(corner[1]), int(corner[0])] = [0, 0, 255]
        corners_list.append([int(round(corner[1])),int(round(corner[0]))])
    int_corners = np.asarray(corners, dtype = int)
    return dst,int_corners, corners_list






