import numpy as np
import Polygon as poly
import imutils
import cv2 as cv
import random as rng
rng.seed(12345)

# Reshaping
def rescaleFrame(frame, scale):
    """This function resizes the the input image (frame) by a certain scale"""
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)


def draw_objects_in_Virtual_Env(src):
    """return a Virtual Enviroment with the objects in the Enviroment as perfect polygons"""

    #https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html
    
    
    threshold = 70
    maxX,maxY,_ = src.shape
    print("MAXX,MAXY:",maxX,"\n",maxY,"\n")
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3,3))
    max_thresh = 255
    
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    
    
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)
    # Draw contours + rotated rects + ellipses
    
    drawing = np.ones((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)*255
    
    for i, c in enumerate(contours):

        box = cv.boxPoints(minRect[i])

        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        print("\nBOX:",box)

        centroid = poly.get_poly_centroid(box,maxY,maxX)
        bgr_poly = src[int(centroid[1]),int(centroid[0])]

        color = (255,255,255)

        if(np.linalg.norm(np.array(bgr_poly)-np.array([40,30,30])) <= 50):
            color = (0,0,0)

        if(np.linalg.norm(np.array(bgr_poly)-np.array([87,69,155])) <= 50):
            print("sink")
            color = (255,0,0)

        cv.drawContours(drawing, [box], 0, color,-1)
    
    return drawing
