import cv2
import numpy as np
import os
import tqdm


# 暗通道最小值滤波半径r
# 这个半径对于去雾效果是有影响的。一定情况下，半径越大去雾的效果越不明显，建议的范围一般是5-25之间，一般选择5,7,9等就会取得不错的效果。
# w的影响自然也是很大的。
# 这个值是我们设置的保留雾的程度（c++代码中w是去除雾的程度，一般设置为0.95就可以了）。这个基本不用修改。
# 导向滤波中均值滤波半径。
# 这个半径建议取值不小于求暗通道时最小值滤波半径的4倍。因为前面最小值后暗通道时一块一块的，为了使得透射率图更加精细，这个r不能过小
# （很容易理解，如果这个r和和最小值滤波的一样的话，那么在进行滤波的时候包含的块信息就很少，还是容易出现一块一块的形状）。

def zmMinFilterGray(src, r=5):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)
    # cv2.imshow('20190708_Dark',Dark_Channel)    # 查看暗通道
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return V1, A


def deHaze(m, r=50, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
    return Y

def ImageHazeRemoval(image):
    img = deHaze(image / 255.0) * 255
    return img

if __name__ == '__main__':
    image_list = os.listdir('D:/dataset/RTTS/JPEGImages')
    new_image_path = 'D:/dataset/RTTS/images'
    for image in tqdm.tqdm(image_list):
        m = deHaze(cv2.imread(os.path.join('D:/dataset/RTTS/JPEGImages', image)) / 255.0) * 255
        cv2.imwrite(os.path.join(new_image_path, image), m)