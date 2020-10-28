import cv2
import numpy as np
#import matplotlib.pylab as plt
def process(image):
    global image_with_lines
    print(image.shape)
    cv2.imshow('vertices',image)
    region_of_interest_vertices= [
        (5, 470),
        (5, 157),
        (633, 157),
        (633, 471)
    ]

    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        #channel_count = img.shape[2]
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    def drow_the_lines(img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        try:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
        except:
            print('lines is none')
        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img

    gblur = cv2.GaussianBlur(image, (5, 5), 0)
    #gblur = cv2.bilateralFilter(image, 5, 50, 50)
    gray_image = cv2.cvtColor(gblur, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 25
    #gblur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    #gblur = cv2.bilateralFilter(gray_image,5,25,25)
    cv2.imshow('gaussianblur',gblur)
    #median = cv2.medianBlur(gblur, 5)
    _, thres = cv2.threshold(gray_image, 165, 255, cv2.THRESH_BINARY)
    #thres1= cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,1)
    cv2.imshow('nor_thres',thres)
    #cv2.imshow('adap_thres',thres1)

    kernel1 = np.ones((3, 3), np.uint8)
    #th = cv2.morphologyEx(thres1,cv2.MORPH_OPEN,kernel1)
    #th = cv2.morphologyEx(thres1,cv2.MORPH_CLOSE,kernel1)
    th = cv2.morphologyEx(thres, cv2.MORPH_TOPHAT, kernel1)  #blur or adaptive thresholding
    cv2.imshow('th',th)
    canny_image = cv2.Canny(th, 200, 200)

    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32), )
    cv2.imshow('crop image',cropped_image)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=10,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=30,
                            maxLineGap=40)
    #print(lines)
        #if lines == None:
            ##continue
    image_with_lines = drow_the_lines(image, lines) 
    return image_with_lines
  # try:
   #     image_with_lines = drow the_lines(image, lines)
   #except:
    #   print('lines is none')

    # cv2.imshow('lines',image_with_lines)
   #try:
    #    return image_with_lines
   #except:
    #    return

cap = cv2.VideoCapture('lane_vgt.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame1',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

