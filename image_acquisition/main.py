#!/usr/bin/env python3

#Library Import
import copy
import cv2
from datetime import datetime

def main():
    # --------------------------------------
    # Initialization
    # --------------------------------------
    
    #Terminal:
        # > Type cd /dev
        # > Type ls video and then press tab, if you find only result as video0, that means only webcam is present.
        # > Now repeat 1 to 2 with USB webcam plugged in. You should find video1 or video2 when you repeat the steps.
    
    # NOTE Não encontrei manual da camara, nem aplicação especifica para esta camâra. Apenas encontrei documentação relativa ao sensor CMOS
    
    #Maximum Resolutation
    WIDTH = 3840
    HEIGHT = 2160

    #Capture image  
    cap = cv2.VideoCapture("/dev/video0") 
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    # cap.set(cv2.CAP_PROP_EXPOSURE, )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('Camera Resolution: ', str(width), 'x', str(height))
   
    # --------------------------------------
    # Execution
    # --------------------------------------
    while(cap.isOpened()): # iterate video frames

        # Time
        data_hora_atuais = datetime.now()
        data_hora = data_hora_atuais.strftime("%Y%m%d_%H_%M_%S")

        # Grab a single frame of video
        result, image_rgb = cap.read() # Capture frame-by-frame
        if result is False:

            break
        # print(image_rgb.shape)
        image_gui = copy.deepcopy(image_rgb) 
               
    # --------------------------------------
    # Visualization
    # --------------------------------------
        # Resize the image 
        scale_factor = 0.25
        cap_resize = cv2.resize(image_gui, None, fx=scale_factor, fy=scale_factor)
        # print('Resized Image ' + str(cap_resize.shape))

        # Display the resulting image
        cv2.imshow('GUI',cap_resize)
      
        k = cv2.waitKey(1)

        if k == ord("s"): 
            cv2.imwrite("./Images/Image_" + str(data_hora) + ".png", image_gui)
            print("Print saved")
           
        # Hit 'q' on the keyboard to quit, (Continuos Image)
        if cv2.waitKey(1) & k == ord('q') :
            break

    
if __name__ == "__main__":
    main()
