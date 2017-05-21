'''
    License Plate Recognition using OpenCV by Leonid Dashko

    Also good sources of information about ALPR:
    - https://github.com/abdulfatir/pyANPD/blob/master/pyANPD.py
    - http://stackoverflow.com/questions/981378/how-to-recognize-vehicle-license-number-plate-anpr-from-an-image
'''

import cv2
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import numpy as np
from pytesseract import image_to_string
from copy import deepcopy
import time
import os


##############################################
# Subplot generator for images
def plot(figure, subplot, image, title, cmap=None):
    figure.subplot(subplot)
    figure.imshow(image, cmap)
    figure.xlabel(title)
    figure.xticks([])
    figure.yticks([])


def satisfy_ratio(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    error = 0.4
    aspect = 4.7272  # In Estonia, car plate size: 52x11 aspect 4,7272
    # Plate number ssizes: https://en.wikipedia.org/wiki/Vehicle_registration_plate#Sizes
    # Set a min and max area. All other patches are discarded
    min = 15*aspect*15  # minimum area
    max = 125*aspect*125  # maximum area
    # Get only patches that match to a respect ratio.
    rmin = aspect - aspect*error
    rmax = aspect + aspect*error

    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    return True


def verify_sizes(rectangle):
    # print candidate
    # help(cv2.minAreaRect)
    (x, y), (width, height), rect_angle = rectangle

    # Calculate angle and discard rects that has been rotated more than 15 degrees
    angle = 90 - rect_angle if (width < height) else -rect_angle
    if 15 < abs(angle) < 165:  # 180 degrees is maximum
        return False

    # We make basic validations about the regions detected based on its area and aspect ratio.
    # We only consider that a region can be a plate if the aspect ratio is approximately 520/110 = 4.727272
    # (plate width divided by plate height) with an error margin of 40 percent
    # and an area based on a minimum of 15 pixels and maximum of 125 pixels for the height of the plate.
    # These values are calculated depending on the image sizes and camera position:
    area = height * width

    if height == 0 or width == 0:
        return False
    if not satisfy_ratio(area, width, height):
        return False

    return True


def get_dominant_color(plate):
    average_color = np.mean(plate).astype(np.uint8)
    return average_color


def is_white_color_dominant(plate):
    average_color = np.mean(plate).astype(np.uint8)
    return 100 <= average_color  # white color is dominant if mean color > 100


def plot_plate_numbers(plates_images):
    ''' Plot Plate Numbers as separate images '''
    i = 0
    for plate_img in plates_images:
        cv2.imshow('plate-%s' % i, plate_img)
        cv2.resizeWindow("plate-%s" % i, 300, 40)
        cv2.imwrite('plates/plate-%s.jpg' % i, plate_img)
        i += 1


def preprocess_plate(plate_img):
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)  # make greyscale
    # gaus_threshold = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
    dilate_thresh = cv2.dilate(plate_gray, kernel, iterations=1)  # dilate
    _, thresh = cv2.threshold(dilate_thresh, 150, 255, cv2.THRESH_BINARY)  # threshold
    # cv2.imshow('Grayscale plate', plate_gray)
    # cv2.imshow('Dilate filter', dilate_thresh)
    # cv2.imshow('Threshold', thresh)
    # cv2.resizeWindow("Grayscale plate", 300, 40)
    # cv2.resizeWindow("Dilate filter", 300, 40)
    # cv2.resizeWindow("Threshold", 300, 40)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
    # cv2.imshow('plate-%s', thresh)
    # cv2.waitKey(0)

    if contours:
        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        max_cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        # Can't satisfy ratio, exit
        max_cnt_area = areas[max_index]
        if not satisfy_ratio(max_cnt_area, w, h):
            return plate_img, None

        final_plate_img = plate_img[y:y + h, x:x + w]  # crop and fetch only plate number

        # # for each contour found, draw a rectangle around it on original image
        plate_img_with_contours = plate_img.copy()
        cv2.drawContours(plate_img_with_contours, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)

        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # ax0, ax1 = axes.flatten()
        #
        # ax0.hist(plate_img.ravel(), bins=255, range=[0, 255], histtype='bar', color='tan')
        # ax0.set_title('Histogram of Initial detected plate')
        # ax1.hist(final_plate_img.ravel(), bins=255, range=[0, 255], histtype='bar', color='lime')
        # ax1.set_title('Histogram of cleaned detected plate')
        #
        # fig.tight_layout()
        # plt.show()

        # plot(plt, 321, img, "Original image")
        # plot(plt, 322, imp_blurred, "Blurred image")
        # plot(plt, 323, img_gray, "Grayscale image")
        # plot(plt, 324, img_sobel_x, "Sobel")
        # plot(plt, 325, img_threshold, "Threshold image")
        # # plot(plt, 326, morph_img_threshold, "After Morphological filter")
        # plt.tight_layout()
        # plt.show()

        # cv2.imshow('Initial detected plate', plate_img)
        # cv2.imshow('Cleaned plate number', final_plate_img)
        # cv2.imshow('Normal threshold', thresh)
        # cv2.imshow('Gaus threshhold', gaus_threshold)
        # cv2.imshow('Plate with detected contours', plate_img_with_contours)
        ## Resize windows
        # cv2.resizeWindow("Initial detected plate", 300, 40)
        # cv2.resizeWindow("Cleaned plate number", 300, 40)
        # cv2.resizeWindow("Normal threshold", 300, 40)
        # cv2.resizeWindow("Gaus threshhold", 300, 40)
        # cv2.resizeWindow("Plate with detected contours", 300, 40)
        # cv2.waitKey(0)
        # exit()

        return final_plate_img, [x, y, w, h]
    else:
        return plate_img, None


def find_contours(img):
    '''
    :param img: (numpy array)
    :return: all possible rectangles (contours)
    '''
    img_blurred = cv2.GaussianBlur(img, (5, 5), 1)  # remove noise
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)  # greyscale image
    # cv2.imshow('', img_gray)
    # cv2.waitKey(0)

    # Apply Sobel filter to find the vertical edges
    # Find vertical lines. Car plates have high density of vertical lines
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_8UC1, dx=1, dy=0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # cv2.imshow('img_sobel', img_sobel_x)

    # Apply optimal threshold by using Oslu algorithm
    retval, img_threshold = cv2.threshold(img_sobel_x, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # cv2.imshow('s', img_threshold)
    # cv2.waitKey(0)

    # TODO: Try to apply AdaptiveThresh
    # Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    # gaus_threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)
    # cv2.imshow('or', img)
    # cv2.imshow('gaus', gaus_threshold)
    # cv2.waitKey(0)

    # Define a stuctural element as rectangular of size 17x3 (we'll use it during the morphological cleaning)
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))

    # And use this structural element in a close morphological operation
    morph_img_threshold = deepcopy(img_threshold)
    cv2.morphologyEx(src=img_threshold, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    # cv2.dilate(img_threshold, kernel=np.ones((1,1), np.uint8), dst=img_threshold, iterations=1)
    # cv2.imshow('Normal Threshold', img_threshold)
    # cv2.imshow('Morphological Threshold based on rect. mask', morph_img_threshold)
    # cv2.waitKey(0)

    # Find contours that contain possible plates (in hierarchical relationship)
    contours, hierarchy = cv2.findContours(morph_img_threshold,
                                           mode=cv2.RETR_EXTERNAL,  # retrieve the external contours
                                           method=cv2.CHAIN_APPROX_NONE)  # all pixels of each contour

    plot_intermediate_steps = False
    if plot_intermediate_steps:
        plot(plt, 321, img, "Original image")
        plot(plt, 322, img_blurred, "Blurred image")
        plot(plt, 323, img_gray, "Grayscale image", cmap='gray')
        plot(plt, 324, img_sobel_x, "Sobel")
        plot(plt, 325, img_threshold, "Threshold image")
        # plot(plt, 326, morph_img_threshold, "After Morphological filter")
        plt.tight_layout()
        plt.show()

    return contours


def find_plate_numbers(origin_img, contours, mask):
    # For each contour detected, extract the bounding rectangle of minimal area and validate every contour
    # before classifying every region

    plot_all_found_contours = False
    plates, plates_images = [], []
    time_start = time.time()
    for rect_n, cnt in enumerate(contours):
        # Also good example to filter rectangles
        # http://nullege.com/codes/show/src@s@e@seawolf5-HEAD@vision@entities@binscontour.py/74/cv2.minAreaRect
        min_rectangle = cv2.minAreaRect(cnt)
        # Debug: keep track of found contours
        if plot_all_found_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(origin_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(origin_img, str(rect_n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 255))

        if verify_sizes(min_rectangle):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = deepcopy(origin_img[y:y + h, x:x + w])  # crop
            clean_plate_img, plate_rect = preprocess_plate(plate_img)

            # Rebuild coords for cleaned plate (if the plate number has been cleared more precisely)
            if plate_rect:
                x1, y1, w1, h1 = plate_rect
                x, y, w, h = x + x1, y + y1, w1, h1

            # In order to make it faster, filter by dominant color on the 2nd step
            if is_white_color_dominant(clean_plate_img):
                # Try to parse vehicle number

                # Apply Tesseract app to parse plate number
                plate_im = Image.fromarray(clean_plate_img)
                # plate_im = plate_im.filter(ImageFilter.MedianFilter())
                # plate_im.save('plates/temp2.jpg')

                t_start = time.time()
                plate_text = image_to_string(plate_im, lang='eng')
                plate_text = plate_text.replace(' ', '').upper()
                print 'Time taken to extract text from contour: %s' % (time.time() - t_start)

                # TODO: Use faster method to detect (Tesseract requires ~0.5 sec to identify text)
                # if len(plate_text):
                    # Draw rectangle around plate number
                cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
                plates.append(plate_text)
                plates_images.append(clean_plate_img)

    print "Time taken to process contours: %s" % (time.time() - time_start)
    # Debug: Plot all found contours
    if plot_all_found_contours:
        cv2.imshow('', origin_img)
        cv2.waitKey(0)
        exit()
    return plates, plates_images, mask

###########################################
# Main methods ############################
###########################################
def process_single_image(images=[], plot_plates=False):
    '''
    :param images: list (full path to images to be processed)
    '''
    if images:
        img_n = 1
        for path_to_image in images:
            t_start = time.time()
            img = cv2.imread(path_to_image)

            # Resizing of the image
            r = 400.0 / img.shape[1]
            dim = (400, int(img.shape[0] * r))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            mask = np.zeros_like(img)  # init mask

            contours = find_contours(img)
            # cv2.drawContours(img, contours, -1, (0, 255, 255))
            # cv2.waitKey(0)
            plates, plates_images, mask = find_plate_numbers(img, contours, mask)

            print "Time needed to complete: %s" % (time.time() - t_start)
            print "Plate Numbers: %s" % ", ".join(plates)

            # Apply mask to image and plot image
            img = cv2.add(img, mask)

            if plot_plates:
                plot_plate_numbers(plates_images)

            cv2.imshow('Resized Original image_%s + Detected Plate Number' % img_n, img)
            img_n += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        exit('Images are not provided!')


def process_video(path_to_video):
    cap = cv2.VideoCapture(path_to_video)  # Load video

    while True:
        ret, frame = cap.read()
        print frame
        if ret is False or (cv2.waitKey(30) & 0xff) == 27: break  # Exit if the video ended

        mask = np.zeros_like(frame)  # init mask
        contours = find_contours(frame)
        plates, plates_images, mask = find_plate_numbers(frame, contours, mask)

        print "Plate Numbers: %s" % ", ".join(plates)

        processed_frame = cv2.add(frame, mask)  # Apply the mask to image
        cv2.imshow('frame', processed_frame)
    cv2.destroyAllWindows()
    cap.release()


###########################################
# Run The Program #########################
###########################################
if __name__ == '__main__':
    test_plates = [os.path.join('test_images', im)  for im in os.listdir("test_images")]
    # images = ['real_plates/IMG_6762.JPG'] + test_plates
    images = ['test_images/3266CNT.JPG']

    process_single_image(images)

    # process_video(path_to_video='traffic.avi')
