#------------------------------------------------------------------------------------------------------------------------
# TALLE 4: Paula Andrea Castro y Michael Hernando Contreras
#------------------------------------------------------------------------------------------------------------------------

# Importación de librerias y y definición de la case
import os
import sys
import numpy as np
from hough import Hough
from orientation_methods import gradient_map
import cv2
import random

class lines:

    # Constructor que recibe el valor de N u lo valida
    def __init__(self):
        self.N = int(input('Ingrese numero par mayor a 200:'))
        NPar=self.N%2
        if NPar==0 and self.N>200:
            self.N=self.N
        else:
            print ("Error! el numero ingresado no es un numero par o no es mayor a 100")

    # Método que contrulle una imagen en fondo color cian y un cuadrilatero en color magenta
    def generate(self):
        imagen = np.ones((self.N, self.N, 3), dtype=np.uint8)
        imagen[:, :, :] = (255, 255, 0)

        x = random.randint(self.N / 2, self.N)

        # Dibujo "cuadrilatero"
        cv2.rectangle(imagen, (x, 100), (100, x), (255, 0, 255), -1)

        cv2.imshow('imagen', imagen)
        cv2.imwrite('example2.png', imagen)
        cv2.waitKey(0)

    #Método que detecta las lineas de una figura
    def DetectCorners(self):
        Standard = 1
        Direct = 2
        path = sys.argv[1]
        image_name = sys.argv[2]
        path_file = os.path.join(path, image_name)
        image = cv2.imread(path_file)

        method = Standard
        high_thresh = 300
        bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)

        hough = Hough(bw_edges)
        if method == Standard:
            accumulator = hough.standard_transform()
        elif method == Direct:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            theta, _ = gradient_map(image_gray)
            accumulator = hough.direct_transform(theta)
        else:
            sys.exit()

        acc_thresh = 50
        N_peaks = 5
        nhood = [25, 9]
        peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        _, cols = image.shape[:2]
        image_draw = np.copy(image)

        for peak in peaks:
            rho = peak[0]
            theta_ = hough.theta[peak[1]]

            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hough.center_x
            y0 = b * rho + hough.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

            if np.abs(theta_) < 80:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
            elif np.abs(theta_) > 100:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
            else:
                if theta_ > 0:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
                else:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)

        cv2.imshow("frame", bw_edges)
        cv2.imshow("lines", image_draw)
        cv2.waitKey(0)

if __name__ == '__main__':
    clase = lines()
    clase.generate()
    clase.DetectCorners()

