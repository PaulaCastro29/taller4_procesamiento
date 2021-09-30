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
import poly_point_insect as bot

class lines:

    # Constructor que recibe el valor de N u lo valida
    def __init__(self):
        self.N = int(input('Ingrese numero par mayor a 200:'))
        NPar=self.N%2
        if NPar==0 and self.N>200:
            self.N=self.N
        else:
            print ("Error! el numero ingresado no es un numero par o no es mayor a 200")

    # Método que contrulle una imagen en fondo color cian y un cuadrilatero en color magenta
    def generate(self):
        imagen = np.ones((self.N, self.N, 3), dtype=np.uint8)
        imagen[:, :, :] = (255, 255, 0)

        x = random.randint(self.N / 2, self.N)

        # Dibujo rectangulo aleatorio
        cv2.rectangle(imagen, (x, 100), (100, x), (255, 0, 255), -1)

        #Para probar los metodos con un poligono quitar comentario de las siguientes lineas
        '''
        pts = np.array([[25, 70], [25, 160],
                        [110, 200], [200, 160],
                        [200, 70], [110, 20]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(imagen, [pts], True, (255, 0, 255))
        '''
        cv2.imshow('imagen generada', imagen)
        cv2.imwrite('example2.png', imagen)

    #Método que detecta las lineas de una figura y sus intersecciones
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
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [2550, 0, 255], thickness=2)
            elif np.abs(theta_) > 100:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
            else:
                if theta_ > 0:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
                else:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)

        cv2.imshow("frame", bw_edges)
        cv2.imwrite("lines.png", image_draw)

        #El codigo que se emplea de aqui en adelante permite encontrar los puntos intersectos usando la funcion bot tomado de: https://github.com/ideasman42/isect_segments-bentley_ottmann/blob/master/poly_point_isect.py
        gray = cv2.cvtColor(image_draw, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        rho = 1
        theta = np.pi / 180
        threshold = 15
        min_line_length = 50
        max_line_gap = 20
        line_image = np.copy(image_draw) * 0

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
        points = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 5)

        lines_edges = cv2.addWeighted(image_draw, 0.8, image_draw, 0, 0)
        intersections = bot.isect_segments(points)
        print('Los puntos de interseccion de las lineas estan en:',intersections)

        for inter in intersections:
            a, b = inter
            for i in range(10):
                for j in range(10):
                    lines_edges[int(b) + i, int(a) + j] = [0, 255, 255]

        cv2.imshow('Intersecciones', lines_edges)
        cv2.waitKey(0)


if __name__ == '__main__':
    clase = lines()
    clase.generate()
    clase.DetectCorners()