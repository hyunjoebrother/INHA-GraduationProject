import cv2
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY

src = cv2.imread("../result_sample/origin360_2_disp_Joint3D60.png" ,cv2.IMREAD_COLOR)
dst = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)



cv2.imwrite("origin360_2_disp_Joint3D60_gray.png", dst)