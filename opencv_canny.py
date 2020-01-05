# python opencv 的图像边缘处理
# 导入我们需要用的库有两个库个分别是cv2和numpy
import cv2 as cv
import numpy as np


# 圖像處理梯度处理三种算子直接调用不同的API调用
# Sobel算子是一阶导数的边缘检测算子，在算法实现过程中，先分别对x,y梯度方向运算，然后合成
# 通过3×3模板作为核与图像中的每个像素点做卷积和运算，然后选取合适的阈值以提取边缘。
def sobel_image(image):
    grad_x = cv.Sobel( image, cv.CV_32F, 1, 0 )
    grad_y = cv.Sobel( image, cv.CV_32F, 0, 1 )
    gradx = cv.convertScaleAbs( grad_x )
    grady = cv.convertScaleAbs( grad_y )
    gradxy = cv.addWeighted( gradx, 0.5, grady, 0.5, 0 )
    cv.imshow( "opencv_sobel", gradxy )


# canny边缘检测 先高斯模糊滤波，然后Sobel变换，最后Canny边缘检测
def canny_edge(image):
    blur = cv.GaussianBlur( image, (3, 3), 0 )  # 斯模糊滤波
    gray = cv.cvtColor( blur, cv.COLOR_RGB2GRAY )  # 灰度处理
    xgrad = cv.Sobel( gray, cv.CV_16SC1, 1, 0 )  # x梯度变换
    ygrad = cv.Sobel( gray, cv.CV_16SC1, 0, 1 )  # y梯度变换
    canny_edge = cv.Canny( xgrad, ygrad, 50, 100 )  # 将sobel梯度带入canny函数中
    cv.imshow( "opencv_canny", canny_edge )


# 打开摄像头获取video图像，然后做canny边缘算法检测边缘。
def video_demo():
    capture = cv.VideoCapture( 0 )  # 打开摄像头，0代表的是设备id，如果有多个摄像头，可以设置其他数值
    while True:
        ret, frame = capture.read()  # 读取摄像头,它能返回两个参数，第一个参数是bool型的ret，其值为True或False，代表有没有读到图片；第二个参数是frame，是当前截取一帧的图片
        frame = cv.flip( frame, 1 )  # 翻转 0:上下颠倒 大于0水平颠倒   小于180旋转
        cv.imshow( "video", frame )
        blur = cv.GaussianBlur( frame, (3, 3), 0 )
        gray = cv.cvtColor( blur, cv.COLOR_RGB2GRAY )
        xgrad = cv.Sobel( gray, cv.CV_16SC1, 1, 0 )
        ygrad = cv.Sobel( gray, cv.CV_16SC1, 0, 1 )
        edge_output = cv.Canny( xgrad, ygrad, 50, 100 )
        cv.imshow( "canny", edge_output )
        fps = capture.get( cv.CAP_PROP_FPS )
        print( fps )
        if cv.waitKey( 10 ) & 0xFF == ord( 'q' ):  # 键盘输入q退出窗口，不按q点击关闭会一直关不掉 也可以设置成其他键。
            break


# 主函数
# 图像的输入输出
# 调用前面写好的图像边缘检测函数
if __name__ == '__main__':
    src = cv.imread( "D:/Users/zwr/PycharmProjects/python-opencv/picture and video/m3.jpg" )  # 输入图片存储地址
    res = cv.resize( src, (400, 340), cv.INTER_CUBIC )  # 对图像大小进行压缩成340:400大小
    cv.imshow( "original", res )  # 输出原图
    # video_demo()    #输出canny算子处理的边缘调用摄像头，需要调用去掉前面的#注释
    canny_edge( res )  # 输出canny算子处理的边缘图像，需要调用去掉前面的#注释
    sobel_image( res )  # 输出sobel算子处理的边缘图像，需要调用去掉前面的#注释
    cv.waitKey( 0 )  # 按下键盘任意键关闭输出图像窗口
    cv.destroyAllWindows()
