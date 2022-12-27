# from
#  twitch-SLAM https://www.youtube.com/watch?v=7Hlb8YX2-W8
# to
#  https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132
# better, this uses RANSAC
# # ... https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6679325/

import cv2
import numpy as np

def mask_frame(frame):
    h = frame.shape[0]
    w = frame.shape[1]
    print(w)
    poly = np.array([(0,480),(700,400),(w,500),(w,640),(620,h),(0,h)]) # cut hood
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, pts=[poly], color=[0xff])
    segment = cv2.bitwise_and(frame, mask)
    cv2.imshow('mask',mask)
    return segment

def calc_lines(frame,hough):
    pass

cap = cv2.VideoCapture("Road - 1101.mp4")
while(cap.isOpened()):
    ret,frame = cap.read()
#    cv2.imshow('frame',frame)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

    # sharpen
#    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
#    kernel = np.array([[-1,-1,-1], [-1,5,-1], [-1,-1,-1]])
#    img_sharp = cv2.filter2D(edges, -1, kernel)
#    edges = img_sharp

    seg = mask_frame(edges)
    hough = cv2.HoughLinesP(seg, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
#    print(hough)
    linesP = hough
    img_lines = np.zeros_like(frame)
    # for line in lines:
    #     for x1,x2,y1,y2 in line:
    #         cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    if ret==True:
        cv2.imshow('edges',edges)
        cv2.imshow('edges',edges)
        cv2.imshow('frame',frame)
         # show one frame at a time
#        cv2.waitKey(100)
        cv2.waitKey(0) == ord('n')
    if cv2.waitKey(1) & 0xff == ord('q'): # imshow only works with this
        cap.release()
        cap.destroyAllWindows()
        break