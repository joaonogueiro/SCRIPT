import copy
import cv2
import json
import numpy as np
import math
import time

class Algoritm:
    def __init__(self):
        self.calibrationParams_path = '../Params/data_calibration.json' 
        self.image_scalePercent = 30
        self.PCB_numb = 20
        self.templateMatch_thresh = 20
        self.mandMask_offset = 700
        self.notMandMask__offset = 10


    def undistort(self, image):
        # Function responsible for removing distortion through the camera calibration parameters

        # Read the file with Extrinsic and Intrinsic parameters of the camera
        with open(self.calibrationParams_path) as f:
            data = json.load(f)
        
        h,  w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.matrix(data['mtx']), np.matrix(data['dist']), (w,h), 1, (w,h))
        
        # Undistort
        image_Undistort = cv2.undistort(image, np.matrix(data['mtx']), np.matrix(data['dist']), None, newcameramtx)
        
        # NOTE: Compensate for the angle (if necessary)
        # Rotate the image one degree clockwise
        # rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), 1.5, 1) # Angle is negative for clockwise rotation
        # image_Undistort = cv2.warpAffine(image_Undistort, rotation_matrix, (w, h))
        
        # Crop the image
        x, y, w, h = roi
        image_Undistort_crop = image_Undistort[y:y+h, x:x+w]
        
        # Return the cropped image
        return image_Undistort_crop
    

    def resizeImage(self, image):
        # Resize image (The image is too big for debug)

        width = int(image.shape[1] * self.image_scalePercent / 100)
        height = int(image.shape[0] * self.image_scalePercent / 100)
        dim = (width, height)
        image_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return image_resized
    

    def img_crop(self, img, start_y, end_y, start_x, end_x):

        # Crop image with given YX limits
        croped = img[start_y:end_y, start_x:end_x]
        return croped
    

    def template_matching(self, template, scene):

        self.template = template
        self.scene = scene

        self.method = cv2.TM_CCORR_NORMED
        h, w = template.shape[:2]
        res = cv2.matchTemplate(self.scene, self.template, self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # print(max_val)
        # if max_val >= 0.91:
        #     top_left = max_loc
        #     bottom_right = (top_left[0] + w, top_left[1] + h)
        #     return top_left, bottom_right
        # else:
        #     return None, None
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return top_left, bottom_right, max_val
    
    def descritorSIFT(self, templateImage, sceneImage, distancematch):
        self.templateImage = templateImage
        self.sceneImage = sceneImage
        self.distanceMatch = distancematch

        # Inizialitation of SIFT
        sift = cv2.SIFT_create()

        # Extract keypoints 
        kp1, des1 = sift.detectAndCompute(self.sceneImage, None)
        kp2, des2 = sift.detectAndCompute(self.templateImage, None) # Não seria necessário estar dentro do ciclo

        ## BRUTE-FORCE ##
        # Inizialitation of object Matcher (Brute-Force)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test 
        good_matches = []
        for m,n in matches:
            if m.distance < self.distanceMatch * n.distance:
                good_matches.append(m)

        # Show the Matches
        img_matches = cv2.drawMatches(self.sceneImage, kp1, self.templateImage, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return kp1, kp2, good_matches, img_matches
    
    def verniz_contour(self, image, idx):
    # Teste de melhorar a detenção dos contornos
        self.image = image

        image = copy.deepcopy(self.image)
        # Only the blue channel is selected
        blue_image = image[:,:,0]

        # Apply Gaussian filtering
        blur_image = cv2.GaussianBlur(blue_image, (5, 5), 0)
        # Apply Otsu's thresholding after Gaussian filtering
        # _ , blur_otsu = cv2.threshold(blur_image, 0, 255, (cv2.THRESH_BINARY + cv2.THRESH_OTSU))
        _ , blur_otsu = cv2.threshold(blur_image, 0, 255, (cv2.THRESH_BINARY + cv2.THRESH_OTSU))

        # Close
        kernel = np.ones((3,3),np.uint8) # Structuring element
        img_close = cv2.morphologyEx(blur_otsu, cv2.MORPH_CLOSE, kernel, iterations = 5)

        # Countour
        contours, hierarchy = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.image, contours, -1, (0,255,0), 1)

        # Remove the analysis connector
        # Apply the Component analysis function 
        analysis = cv2.connectedComponentsWithStats(img_close, 4, cv2.CV_32S) 
        (totalLabels, label_ids, values, centroid) = analysis 
    
        # Initialize a new image to 
        # store all the output components 
        output = np.zeros(img_close.shape, dtype="uint8")

        # Loop through each component 
        for i in range(1, totalLabels): 

              # Area of the component 
            area = values[i, cv2.CC_STAT_AREA]  

            if (area > 8000) and (area < 40000): 
                componentMask = (label_ids == i).astype("uint8") * 255
                output = cv2.bitwise_or(output, componentMask) 


        # Get the label ID of the clicked pixel
        # label_id = 2

        # Remove the selected area by setting its corresponding label to 0
        # output[label_ids == label_id] = 255

        # cv2.imshow(f'Output {idx}', output)

        # Debug
        # cv2.imshow(f'Blue Channel image number {idx}', blue_image)
        # cv2.imshow(f'Blur image number {idx}', blur_image)
        # cv2.imshow(f'Binarization image number {idx}', blur_otsu)
        # cv2.imshow(f'Close image number {idx}', img_close)
        # cv2.imshow(f'Result contour image number {idx}', output)
        return output
    
    def validation_masks(self, image):
        # TODO mais tarde substituir por a opção de interface com o utilizador
        self.image = image

        # Creation of masks
        mandatory_Mask = np.zeros_like(self.image[:,:,0], dtype=np.uint8)
        notMandatory_Mask = copy.deepcopy(mandatory_Mask)

        ##############################
        # Verniz Mask (mandatory_Mask)
        ##############################
        # Params
        height_mandatory = 50
        # height_mandatory = 130
        length_mandatory = 110

        start_x = 0
        end_y = mandatory_Mask.shape[0]

        # start_x = 0
        end_x = start_x + length_mandatory
        # end_y = mandatory_Mask.shape[0]
        start_y = end_y - height_mandatory

        # Criar um retângulo branco na máscara
        mandatory_Mask[start_y:end_y, start_x:end_x] = 255

        ###################################################
        # Areas can't have Varnish Mask (notMandatory_Mask)
        ###################################################
        # Horizontal
        start_x = 0 
        end_x = self.image.shape[1]
        start_y = 0
        end_y = 40
        notMandatory_Mask[start_y:end_y, start_x:end_x] = 255

        # Vertical
        start_x = self.image.shape[1] - 40
        end_x = self.image.shape[1]
        start_y = 0
        end_y = self.image.shape[0]
        notMandatory_Mask[start_y:end_y, start_x:end_x] = 255

        return mandatory_Mask, notMandatory_Mask
    
    def valitationRepres(self, image, info):
        self.image = image
        self.info = info
        self.OKmask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.NOKmask = copy.deepcopy(self.OKmask)
                    
        for idx, info in self.dict_infoPCB_Sorted.items():
            self.value = info["value"]
            self.top_left = info["topLeft_Point"]
            self.bottom_right = info["bottomRight_Point"]
            if self.value == 'OK':
                # cv2.putText(self.image, str(idx), (self.top_left[0]+150, self.top_left[1]+150), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 0), 5, cv2.LINE_AA) 
                cv2.putText(self.image, 'OK', self.top_left, cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 255, 0), 5, cv2.LINE_AA) 
                cv2.rectangle(self.image, self.top_left, self.bottom_right, (0, 255, 0), 5)
                cv2.rectangle(self.OKmask, self.top_left, self.bottom_right, 255, -1)
            elif self.value == 'NOK':
                # cv2.putText(self.image, str(idx), (self.top_left[0]+150, self.top_left[1]+150), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 0), 5, cv2.LINE_AA) 
                cv2.putText(self.image, 'NOK', self.top_left, cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 255), 5, cv2.LINE_AA) 
                cv2.rectangle(self.image, self.top_left, self.bottom_right, (0, 0, 255), 5)
                cv2.rectangle(self.NOKmask, self.top_left, self.bottom_right, 255, -1)
        # Colored Masks (b,g,r)
        OKmask_colored = cv2.merge([np.zeros_like(self.OKmask), self.OKmask, np.zeros_like(self.OKmask)])  # Green Channel
        NOKmask_colored = cv2.merge([np.zeros_like(self.NOKmask), np.zeros_like(self.NOKmask), self.NOKmask])  # Red Channel
        # Aplicar as máscaras na imagem resultante com opacidade
        alpha = 0.25  # opacity
        beta = 1 - alpha
        self.resultImage = cv2.addWeighted(self.image, beta, OKmask_colored, alpha, 0)
        self.resultImage = cv2.addWeighted(self.resultImage, 1, NOKmask_colored, alpha, 0)
        return self.resultImage



    def main(self, imageWhite, imageUV):
        # TIC
        t = time.time()

        # Receive the images
        self.imageWhite = imageWhite
        self.imageUV = imageUV

        # Undistort Images
        self.imageWhite_Undst = self.undistort(self.imageWhite)
        self.imageUV_Undst = self.undistort(self.imageUV)

        # --------------------------------------
        # Crop the ROI's
        # --------------------------------------
        # Divide the image into multiple regions of interest (MÁX 20 ROI's - maximum tray capacity)
        ### -> Scene Image <- ### 
        self.sceneWhite = copy.deepcopy(self.imageWhite_Undst)
        self.sceneWhiteGRAY = cv2.cvtColor(self.sceneWhite, cv2.COLOR_BGR2GRAY)

        ### -> Template Image <- ### 
        self.template_img = cv2.imread("static/Template/template_8.png")
        self.template_gray = cv2.cvtColor(self.template_img, cv2.COLOR_BGR2GRAY)

     
        self.dict_infoPCB = {}
        # self.thresh = 0.924
        # self.thresh = 0.95
        # self.threshVAL = self.thresh + 0.01
        # idx = 0
        # while self.thresh < self.threshVAL:
            # idx +=1
        for idx, m in enumerate(range(0, self.PCB_numb-13)):
            top_left, bottom_right, self.threshVAL = self.template_matching(self.template_gray, self.sceneWhiteGRAY)
            cv2.rectangle(self.sceneWhiteGRAY, top_left, bottom_right, (0, 0, 0), -1)
            # self.topLeft_offset = (top_left[0]-self.templateMatch_offset,top_left[1]-self.templateMatch_offset)
            # self.bottomRight_offset = (bottom_right[0]+self.templateMatch_offset, bottom_right[1]+self.templateMatch_offset)
            self.dict_infoPCB[idx] = {
                "topLeft_Point": top_left,
                "bottomRight_Point": bottom_right,
                "value": None
            }
        # cv2.imshow(f"Result {idx}", self.resizeImage(self.sceneWhiteGRAY))
        # cv2.waitKey(0)

        # Ordenando o dicionário com base nas coordenadas dos pontos
        self.dict_infoPCB_sorted = dict(sorted(self.dict_infoPCB.items(), key=lambda item: (item[1]["topLeft_Point"], item[1]["bottomRight_Point"])))
        # Reinicializando o dicionário com os novos índices
        self.dict_infoPCB_Sorted = {}
        for idx, (key, value) in enumerate(self.dict_infoPCB_sorted.items()):
            self.dict_infoPCB_Sorted[idx] = value
       

        self.templateSIFT = cv2.imread("static/Template/template_8.png")
        self.templateSIFT_GRAY = cv2.cvtColor(self.templateSIFT, cv2.COLOR_BGR2GRAY) 
        #####################################
        # Cycle that runs through all PCB's #
        #####################################
        # for idx, (top_left, bottom_right) in enumerate(self.coordenadas_ordenadas):
        for idx, info in self.dict_infoPCB_Sorted.items():
            top_left = info["topLeft_Point"]
            bottom_right = info["bottomRight_Point"]
      
            # Load PCB Image
            # -> White Ilumination <- 
            self.PCB_cropWhite = self.img_crop(self.imageWhite_Undst, top_left[1] - self.templateMatch_thresh, bottom_right[1] + self.templateMatch_thresh, top_left[0] - self.templateMatch_thresh, bottom_right[0] + self.templateMatch_thresh)
            # cv2.imwrite(f'static/templateMatching/templateMatching_{idx}.png', self.PCB_cropWhite)
            self.PCB_cropWhiteGRAY = cv2.cvtColor(self.PCB_cropWhite, cv2.COLOR_BGR2GRAY) # Gray Scale
            # -> UV Ilumination <- 
            self.PCB_cropUV = self.img_crop(self.imageUV_Undst, top_left[1] - self.templateMatch_thresh, bottom_right[1] + self.templateMatch_thresh, top_left[0] - self.templateMatch_thresh, bottom_right[0] + self.templateMatch_thresh)
            self.PCB_cropUVGRAY = cv2.cvtColor(self.PCB_cropUV, cv2.COLOR_BGR2GRAY) # Gray Scale
            
            # Gaussian bluer (Eliminates some noice)
            self.blr = cv2.GaussianBlur(self.PCB_cropWhiteGRAY, (5, 5), 0)
            # cv2.imshow(f'{idx} PCB', self.PCB_cropWhite)
        
            ##################
            #-- Using SIFT --
            ##################
            # Better crop, insulate the PCB
            kp1, kp2, good_matches, _ = self.descritorSIFT(self.templateSIFT_GRAY, self.PCB_cropWhiteGRAY, 0.75)
            ## Merge the template in Scene ##
            # Extrair os pontos correspondentes nas duas imagens
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
            # Calcular a matriz de transformação
            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

            # Obter as dimensões da imagem do templatey
            h, w = self.templateSIFT_GRAY.shape

            # Calcular os cantos do retângulo no template
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # Transformar os cantos do retângulo do template na cena
            dst = cv2.perspectiveTransform(pts, M).reshape(-1, 2)


            ##########################################################################################################
            # Compensate for the angle (Place the plate orthogonal to the axis)

            # Pontos da Bonding Box
            x1 = dst[0,0]
            y1 = dst[0,1]
            x2 = dst[1,0]
            y2 = dst[1,1]
            x3 = dst[2,0]
            y3 = dst[2,1]
            x4 = dst[3,0]
            y4 = dst[3,1]

            m = (y3 - y2)/(x3 - x2) # Declive de uma reta linear
            # Usando a função arco tangente (atan) para calcular o ângulo em radianos
            angulo_radianos = math.atan(m)
            # Convertendo o ângulo de radianos para graus
            m = math.degrees(angulo_radianos)
  
            # Rotate the image one degree clockwise
            rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), m, 1) # Angle is negative for clockwise rotation
            # self.PCB_cropWhite_aligned = cv2.warpAffine(self.PCB_cropWhite, rotation_matrix, (w, h))
            self.PCB_cropUV_aligned = cv2.warpAffine(self.PCB_cropUV, rotation_matrix, (w, h))
            
            # Desenhar um retângulo na cena correspondente ao template
            # scene_img = cv2.polylines(self.PCB_cropWhite, [np.int32(dst)], True, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.imshow(f'{idx} SIFT', scene_img)
            ##########################################################################################################

            tol = 0
            self.PCB_cropedUV = self.img_crop(self.PCB_cropUV_aligned, round(min(y1, y4))-tol, round(min(y2, y3))+tol, round(min(x1, x2))-tol, round(min(x3, x4))+tol) #NOTE alterei o round(max(y2, y3))+tol)) para round(min(y2, y3))+tol))
            # cv2.imshow(f'{idx} SIFT', self.PCB_cropedUV)


            # --------------------------------------
            # Cassification/ Validation
            # --------------------------------------
            self.PCB_vernizMask = self.verniz_contour(self.PCB_cropedUV, idx)
            self.mandatory_Mask, self.notMandatory_Mask = self.validation_masks(self.PCB_cropedUV)

            # Using the logical AND operator to compare the masks
            self.result_mandatory = cv2.bitwise_and(self.PCB_vernizMask, self.mandatory_Mask)
            # cv2.imshow(f'{idx_img} Result Mandatory Mask', result_mandatory)
            self.result_notMandatory = cv2.bitwise_and(self.PCB_vernizMask, self.notMandatory_Mask)
            # cv2.imshow(f'{idx_img} Result Not Mandatory Mask', result_notMandatory)


            if (abs(cv2.countNonZero(self.mandatory_Mask) - cv2.countNonZero(self.result_mandatory)) <= self.mandMask_offset): # and (cv2.countNonZero(result_notMandatory) <= tol_notMand):
                # coordinates[str(idx_img)]['Value'] = "OK"
                self.dict_infoPCB_Sorted[idx]['value'] = "OK"
            else:
                # coordinates[str(idx_img)]['Value'] = "NOK"
                self.dict_infoPCB_Sorted[idx]['value'] = "NOK"

        # --------------------------------------
        # Visualization
        # --------------------------------------
        self.represent_Image = copy.deepcopy(self.imageWhite_Undst)
        self.final_result = self.valitationRepres(self.represent_Image, self.dict_infoPCB_Sorted)
        self.final_result_resize = self.resizeImage(self.final_result)
  
        # TOC
        print(f'Tempo gasto no algoritmo: {time.time() - t}')
        return self.final_result_resize
   
        # # --------------------------------------
        # # Termination (NOTE DEBUG)
        # # --------------------------------------
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     exit(0)   


# NOTE Debug:
# algoritm = Algoritm()
# imageWhite = cv2.imread('static/Image1.png')
# imageUV = cv2.imread('static/Image2.png')
# final_result_resize = algoritm.main(imageWhite, imageUV)
# cv2.imshow('Final Result', final_result_resize)
# cv2.waitKey()