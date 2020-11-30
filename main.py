import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
# import mouse
import pyautogui as pg
# import math

# mouse.click('right')
# prinsssssssSt(mouse.get_position())

class HandCursor:
    def __init__(self, camera=0):
        # camera number
        self.camera = camera

        # Skin color segmentation mask (HSV)
        self.HSV_min = np.array([0,40,50],np.uint8) # 0,40,50 | 0, 20, 70
        self.HSV_max = np.array([50,250,255],np.uint8) #50,250,255 | 20, 255, 255

    def run(self, fps=20, result=False, raw=False, HSV=False, contour=False):
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
            result_video = cv.VideoWriter('Results/result_video.avi', fourcc, fps, (frame_width,frame_height))
        if raw:
            fourcc1 = cv.VideoWriter_fourcc(*'MJPG')
            raw_video = cv.VideoWriter('Results/raw.avi', fourcc1, fps, (frame_width,frame_height))
        if HSV:
            fourcc2 = cv.VideoWriter_fourcc(*'MJPG')
            HSV_video = cv.VideoWriter('Results/HSV_mask.avi', fourcc2, fps, (frame_width,frame_height))
        if contour:
            fourcc3 = cv.VideoWriter_fourcc(*'MJPG')
            contour_video = cv.VideoWriter('Results/contour.avi', fourcc3, fps, (frame_width,frame_height))

        ### main loop ###
        # distance calculate function
        def dist(p1, p2):
            return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

        # Cursor movement setup
        pointer = None
        prev_pointer = None
        distance_array = []
        filter_size = 5 # how large you want your average filter
        scale_factor_x = 3 # scaling x
        scale_factor_y = 3 # scaling y

        # finger counting
        count_array = []
        count_size = 2 # count array size (like average filter but with np.all)
        official_count = 1

        # state
        state = 1

        # loop
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
            skin_img = cv.dilate(skin_img, np.ones((5,5)), iterations=1)

            # blur image
            skin_img = cv.GaussianBlur(skin_img, (7,7), 0) # 7x7 kernel and 0 STD

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

                    # find defects
                    hull = cv.convexHull(hand, returnPoints=False)
                    defects = cv.convexityDefects(hand, hull)

                    # find number of fingers
                    count = 1
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(hand[s][0])
                        end = tuple(hand[e][0])
                        far = tuple(hand[f][0])
                        a = np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
                        b = np.sqrt((far[0]-start[0])**2+(far[1]-start[1])**2)
                        c = np.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
                        angle = np.arccos((b**2+c**2-a**2)/(2*b*c))*57
                        if angle <= 90:
                            count += 1
                        cv.circle(contour_frame, far, 5, [0, 0, 255], -1)
                        cv.line(contour_frame, start, end, [0, 255, 0], 2)

                    # filter to ensure no finger jumps
                    # official_count = count
                    count_array.append(count)
                    if len(count_array) > count_size:
                        count_array.pop(0)
                        ca = np.array(count_array)
                        if np.all(ca == ca[0]):
                            official_count = count
                    if len(count_array) < count_size:
                        pass

                    # optional video save
                    if contour: contour_video.write(contour_frame)

                    ### movement & states ###
                    # get pointer
                    pointer = tuple(hand[hand[:,:,1].argmin()][0])
                    cv.circle(contour_frame, pointer, 5, (0,0,255), 3)

                    # hand detected
                    if prev_pointer and dist(pointer, prev_pointer) <= 100:
                        ### states ###
                        # these are of course alterable dependent on CAD shortcuts/desired usability
                        # state 1: movement
                        if official_count == 1:
                            state = 1
                            # moving average filter for movement
                            distance = ((pointer[0]-prev_pointer[0])*scale_factor_x, (pointer[1]-prev_pointer[1])*scale_factor_y)
                            distance_array.append(distance)
                            if len(distance_array) > filter_size:
                                distance_array.pop(0)
                                da = np.array(distance_array)
                                x_val = np.sum(da,axis=0)[0]//filter_size
                                y_val = np.sum(da,axis=0)[1]//filter_size
                                pg.move(x_val, y_val)
                            if len(distance_array) < filter_size:
                                pass
                        # state 2: left click
                        if official_count == 2 and state != 2:
                            state = 2
                            cv.putText(contour_frame, "Lclick!", (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            pg.click()
                        # state 3: sketch
                        if official_count == 3 and state != 3:
                            state = 3
                            cv.putText(contour_frame, "Sketch", (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            pg.press('s')
                        # state 4: extrude
                        if official_count == 4 and state != 4:
                            state = 4
                            cv.putText(contour_frame, "Extrude", (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            pg.press('e')
                        # state 5: enter
                        if official_count == 5 and state != 5:
                            state = 5
                            cv.putText(contour_frame, "Enter", (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            pg.press('enter')

                    prev_pointer = pointer


            # cv.imshow('RGB', RGB_image)
            cv.putText(contour_frame, "finger count" + str(official_count), (0,25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv.imshow('Contour Frame', contour_frame)

            # optional video save
            if result: result_video.write(contour_frame)

            # break if press q
            if cv.waitKey(1) & 0xFF == ord('q'):
                # just incase
                pg.keyUp('ctrl')
                break

        # close camera
        video.release()
        # output.release()
        cv.destroyAllWindows()

cursor = HandCursor()
cursor.run(fps=20, result=True, raw=True, HSV=True, contour=True)
# cursor.run(raw=True, HSV=True)
#
