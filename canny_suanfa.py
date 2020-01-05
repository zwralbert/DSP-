# python opencv 的图像边缘处理
# 导入我们需要用的库有三个个库个分别是cv2、numpy和matplotlib。
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 定义sobel算子，x和y方向
sobel_suanzi_x = np.array( [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] )
sobel_suanzi_y = np.array( [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] )

# 定义双边阈值大小
TL = 50
TH = 100


# canny算法边缘函数
# 先利用cv库直接高斯滤波、灰度，在进行梯度计算，用Sobel算子，然后进行非最大抑制计算，最后双边阈值的判断和取值
def canny_edge(img):
    blur = cv.GaussianBlur( img, (3, 3), 0 )  # 高斯滤波
    img = cv.cvtColor( blur, cv.COLOR_RGB2GRAY )  # 灰度处理
    new_sobel_x = np.zeros( img.shape, dtype="uint8" )
    new_sobel_y = np.zeros( img.shape, dtype="float" )
    r, c = img.shape

    # 利用Sobel算子求想x,y梯度,将img图像切片成无数个3*3的大小矩阵
    for i in range( 1, r - 2 ):
        for j in range( 1, c - 2 ):
            sobel_x = (np.dot( np.array( [1, 1, 1] ), (sobel_suanzi_y * img[i:i + 3, j:j + 3]) )).dot(
                np.array( [[1], [1], [1]] ) )
            sobel_y = (np.dot( np.array( [1, 1, 1] ), (sobel_suanzi_x * img[i:i + 3, j:j + 3]) )).dot(
                np.array( [[1], [1], [1]] ) )
            # print(img)
            # 求梯度方向
            if sobel_x[0] == 0:
                new_sobel_y[i - 1, j - 1] = 90
                continue
            else:
                temp = (np.arctan( sobel_y[0] / sobel_x[0] ))

            # 梯度方向信息
            if sobel_x[0] * sobel_y[0] > 0:
                if sobel_x[0] > 0:
                    new_sobel_y[i - 1, j - 1] = np.abs( temp )
                else:
                    new_sobel_y[i - 1, j - 1] = (np.abs( temp ) - 180)
            if sobel_x[0] * sobel_y[0] < 0:
                if sobel_x[0] > 0:
                    new_sobel_y[i - 1, j - 1] = (-1) * np.abs( temp )
                else:
                    new_sobel_y[i - 1, j - 1] = 180 - np.abs( temp )

            new_sobel_x[i - 1, j - 1] = abs( sobel_x ) + abs( sobel_y )  # x,y梯度方向相加得到sobel算子图

    cv.imshow( "sobel_suanzi", new_sobel_x )  # 获取sobel边缘图像

    # 利用梯度方向划分进行梯度强度的非最大抑制的区分
    # 梯度向量不在我们的方向是，直接归纳在最接近它的方向上
    for i in range( 1, r - 2 ):
        for j in range( 1, c - 2 ):

            if (((new_sobel_y[i, j] >= -22.5) and (new_sobel_y[i, j] < 22.5)) or
                    ((new_sobel_y[i, j] <= -157.5) and (new_sobel_y[i, j] >= -180)) or
                    ((new_sobel_y[i, j] >= 157.5) and (new_sobel_y[i, j] < 180))):
                new_sobel_y[i, j] = 0.0

            elif (((new_sobel_y[i, j] >= 112.5) and (new_sobel_y[i, j] < 157.5)) or
                  ((new_sobel_y[i, j] <= -22.5) and (new_sobel_y[i, j] >= -67.5))):
                new_sobel_y[i, j] = -45.0

            elif (((new_sobel_y[i, j] >= 22.5) and (new_sobel_y[i, j] < 67.5)) or
                  ((new_sobel_y[i, j] <= -112.5) and (new_sobel_y[i, j] >= -157.5))):
                new_sobel_y[i, j] = 45.0

            elif (((new_sobel_y[i, j] >= 67.5) and (new_sobel_y[i, j] < 112.5)) or
                  ((new_sobel_y[i, j] <= -67.5) and (new_sobel_y[i, j] >= -112.5))):
                new_sobel_y[i, j] = 90.0

    canny_max = np.zeros( new_sobel_x.shape )  # 非极大值抑制图像矩阵

    # 1）沿着四个梯度方向，寻找像素点局部最大值，比较它前面和后面的梯度值。
    # 2)如果当前像素的梯度强度与另外两个像素相比最大，则该像素点保留为边缘点，否则该像素点将被抑制。
    for i in range( 1, canny_max.shape[0] - 1 ):
        for j in range( 1, canny_max.shape[1] - 1 ):
            if (new_sobel_y[i, j] == 0.0) and (new_sobel_x[i, j] == np.max(
                    [new_sobel_x[i, j], new_sobel_x[i + 1, j], new_sobel_x[i - 1, j]] )):
                canny_max[i, j] = new_sobel_x[i, j]

            if (new_sobel_y[i, j] == -45.0) and new_sobel_x[i, j] == np.max(
                    [new_sobel_x[i, j], new_sobel_x[i - 1, j - 1], new_sobel_x[i + 1, j + 1]] ):
                canny_max[i, j] = new_sobel_x[i, j]

            if (new_sobel_y[i, j] == 45.0) and new_sobel_x[i, j] == np.max(
                    [new_sobel_x[i, j], new_sobel_x[i - 1, j + 1], new_sobel_x[i + 1, j - 1]] ):
                canny_max[i, j] = new_sobel_x[i, j]

            if (new_sobel_y[i, j] == 90.0) and new_sobel_x[i, j] == np.max(
                    [new_sobel_x[i, j], new_sobel_x[i, j + 1], new_sobel_x[i, j - 1]] ):
                canny_max[i, j] = new_sobel_x[i, j]

    canny_double( canny_max )  # 双边处理函数嵌入


def canny_double(canny_max):
    canny_double = np.zeros( canny_max.shape )  # 定义双阈值图像

    # 根据阈值进行判断将当前像素的梯度强度与沿正负梯度方向上的两个像素进行比较。
    # 如果边缘像素的梯度值高于高阈值，则将其标记为强边缘像素；保留。
    # 如果边缘像素的梯度值小于高阈值并且大于低阈值，则将其标记为弱边缘像素；保留。
    # 如果边缘像素的梯度值小于低阈值，则会被抑制。
    for i in range( 1, canny_double.shape[0] - 1 ):
        for j in range( 1, canny_double.shape[1] - 1 ):
            if canny_max[i, j] > TH:
                canny_double[i, j] = 1
            elif canny_max[i, j] < TL:
                canny_double[i, j] = 0
            elif ((canny_max[i + 1, j] < TH) or (canny_max[i - 1, j] < TH) or (canny_max[i, j + 1] < TH) or
                  (canny_max[i, j - 1] < TH) or (canny_max[i - 1, j - 1] < TH) or (canny_max[i - 1, j + 1] < TH) or
                  (canny_max[i + 1, j + 1] < TH) or (canny_max[i + 1, j - 1] < TH)):
                canny_double[i, j] = 1

    cv.imshow( "canny_suanfa", canny_double )  # 获取canny边缘图像


# 主函数
# 图像的输入输出
# 调用前面写好的图像边缘检测函数
if __name__ == '__main__':
    src = cv.imread( "D:/Users/zwr/PycharmProjects/python-opencv/picture and video/m3.jpg" )  # 输入图片存储地址
    res = cv.resize( src, (400, 340), cv.INTER_CUBIC )  # 对图像大小进行压缩成340:400大小
    canny_edge( res )
    cv.imshow( "original", res )  # 原始图像
    cv.waitKey( 0 )  # 按下键盘任意键关闭输出图像窗口
    cv.destroyAllWindows()
