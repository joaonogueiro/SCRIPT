#!/usr/bin/env python3

# Site: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2 as cv
import glob
import json
import pprint

#Dimensões do Tabuleiro de Xadrez
cbcol = 6
cbrow = 9
cbw = 25

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, cbw, 0.001)

# preparar os pontos do objeto, como (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Vetores para armazenar os pontos de objeto e pontos de imagem de todas as imagens.
objpoints = [] # ponto 3d no espaço do mundo real
imgpoints = [] # ponto 2d no plano da imagem.

# images = glob.glob('./Imagens_test/*.jpg')
images = glob.glob('./Images_Calibration/*.png')

i=0
for fname in images:
    print(fname)
    img = cv.imread(fname)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Encontra os cantos do tabuleiro de xadrez
    ret, corners = cv.findChessboardCorners(gray, (cbcol,cbrow), None)

    #Se os cantos forem encontrados, adiciona os pontos de objeto e pontos de imagem (após refiná-los)
    print(ret)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        
        #Desenha e mostra os cantos
        cv.drawChessboardCorners(img, (cbcol, cbrow), corners2, ret)
        cv.imwrite('./Resultado/RESULTADO' + str(i) + '.jpg', img)
        # cv.imshow('Debug', img)
        # cv.waitKey(150)
        i += 1
cv.destroyAllWindows()

params = {"mtx": [], "dist": [], "rvecs": [], "tvecs": []}

#Parametros de calibração da câmera
_ , mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Converter arrays em listas para guardar no ficheiro JSON
params['mtx'] = mtx.tolist()
params['dist'] = dist.tolist()
params['rvecs'] = [rvec.tolist() for rvec in rvecs]  
params['tvecs'] = [tvec.tolist() for tvec in tvecs]  

#Resultados
print("\nCamera matrix :" + str(mtx))
print("\nDistortion coefficients : " + str(dist))
print("\nRotation vector : " + str(rvecs))
print("\nTranslation vector : " + str(tvecs))

# Salvar os dados em um arquivo JSON
with open('../Params/data_calibration.json', 'w') as json_file:
    json.dump(params, json_file)


# import numpy as np
# import cv2 as cv
# import glob

# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.

# images = glob.glob('*.jpg')

# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, (7,6), None)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)
#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (7,6), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(500)
# cv.destroyAllWindows()