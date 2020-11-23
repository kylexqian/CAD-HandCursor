import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import mouse

# mouse.click('right')
# print(mouse.get_position())

class HandCursor:
    def __init__(self, camera=0):
        # camera number
        self.camera = camera

        # Skin color segmentation mask (HSV)
        self.HSV_min = np.array([0,40,45],np.uint8) # 0,40,50 | 0, 20, 70
        self.HSV_max = np.array([50,255,255],np.uint8) #50,250,255 | 20, 255, 255

    def run(self, result=False, raw=False, HSV=False, contour=False):
        # get camera (default 0)
        print('Setting up camera...')
        video = cv.VideoCapture(self.camera)
        print('Camera setup')

        frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH)) # int(video.get(3))
        frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT)) # int(video.get(4))

        ### video writing ###
        # Write videos to a folder called 'Results'
        if video.isOpened() == False:
            print('Error reading video')
        if result:
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            result_video = cv.VideoWriter('Results/result_video.avi', fourcc, 20, (frame_width,frame_height))
        if raw:
            fourcc1 = cv.VideoWriter_fourcc(*'MJPG')
            raw_video = cv.VideoWriter('Results/raw.avi', fourcc1, 20, (frame_width,frame_height))
        if HSV:
            fourcc2 = cv.VideoWriter_fourcc(*'MJPG')
            HSV_video = cv.VideoWriter('Results/HSV_mask.avi', fourcc2, 20, (frame_width,frame_height))
        if contour:
            fourcc3 = cv.VideoWriter_fourcc(*'MJPG')
            contour_video = cv.VideoWriter('Results/contour.avi', fourcc3, 20, (frame_width,frame_height))

        ### main loop ###
        while True:
            ### Image Processing ###
            # Read video and flip
            ret, frame = video.read()
            frame = cv.flip(frame, 0)
            frame = cv.flip(frame, 1)

            # Show raw frame
            cv.imshow('Raw frame', frame)
            if raw: raw_video.write(frame)

            ### Skin Detection ###
            # Create HSV image
            hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # Apply skin color mask
            mask = cv.inRange(hsv_img, self.HSV_min, self.HSV_max)
            skin_img = cv.bitwise_and(hsv_img, hsv_img, mask=mask)

            # morphological operators
            kernel = np.ones((9,9)) # 7x7 kernel
            skin_img = cv.erode(skin_img,  kernel, iterations=1)
            skin_img = cv.dilate(skin_img, kernel, iterations=1)

            # blur image
            skin_img = cv.GaussianBlur(skin_img, (5,5), 0) # 5x5 kernel and 0 STD

            # skin mask result (and video)
            cv.imshow('skin mask', skin_img)
            if HSV: HSV_video.write(skin_img)

            ### Contours ###
            # get RGB and gray image, create contour frame
            RGB_image = cv.cvtColor(skin_img, cv.COLOR_HSV2BGR)
            gray_image = cv.cvtColor(RGB_image, cv.COLOR_BGR2GRAY)
            contour_frame = frame.copy()

            # find contours
            contours, _ = cv.findContours(gray_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                # max area contour = hand
                hand = max(contours, key = lambda x: cv.contourArea(x))

                # if area is large enough, draw contour
                if cv.contourArea(hand)>15000: # arbitrary 10000 pixel area
                    # draw contour on hand
                    cv.drawContours(contour_frame, [hand], 0, (255, 0, 0), 3)

                    # create hull
                    hull = cv.convexHull(hand)
                    cv.drawContours(contour_frame, [hull], 0, (0, 255, 0),3)

                    # optional video save
                    if contour: contour_video.write(contour_frame)


            # cv.imshow('RGB', RGB_image)
            cv.imshow('Contour Frame', contour_frame)


            # break if press q
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # close camera
        video.release()
        # output.release()
        cv.destroyAllWindows()

cursor = HandCursor()
cursor.run(result=False, raw=False, HSV=False, contour=False)
# cursor.run(raw=True, HSV=True)
#
