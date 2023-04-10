import math
from math import *
import torch
import numpy as np

## pow 효과 : 특징 값을 극대화 --> 차이가 커짐


def face_height(pts):
    return pts[8][0]

def face_width(pts):
    return pts[16][1] - pts[0][1]

def tn(pts):
    return torch.tensor([[0, pts[8][1]]])

def getAngle3P(p1 , p2  ,p3 , angle_vec = False):
    """
        세점 사이의 끼인 각도 구하기
    """

    x, y = 1, 0
    rad = torch.atan2(p3[y] - p1[y], p3[x] - p1[x]) - torch.atan2(p2[y] - p1[y], p2[x] - p1[x])
    # deg = rad * (180 / np.pi)
    # if angle_vec:
    #     deg = 360-abs(deg)
    return rad

def getAngle3P_beta(p1, p2, p3):
    x, y = 1, 0
    # v1 = torch.tensor([p1[x]-p2[x], p1[y]-p2[y]])
    # v2 = torch.tensor([p3[x]-p2[x], p3[y]-p2[y]])
    v1 = p1-p2
    v2 = p3-p2

    v1_dot_v2 = torch.dot(v1, v2)
    angle = torch.acos(v1_dot_v2 / (torch.norm(v1) * torch.norm(v2))) # if you really want the angle
    angle = min(angle, math.pi - angle)

    return angle

def getAngle2P(p1, p2):
    x, y = 1, 0
    dx = abs(p1[x]-p2[x])
    dy = abs(p1[y]-p2[y])
    angle = torch.atan2(dy, dx)

    return angle

def getSignedAngle2P(p1, p2):
    x, y = 1, 0
    angle = atan2(p2[y], p2[x]) - atan2(p1[y], p1[x])
    return angle

def dist(pts, a, b):
    return (pts[a]-pts[b]).pow(2).sum().sqrt()

def calculate_f8_l(pts):
    return getAngle2P(pts[0], pts[8])

def calculate_f9_l(pts):
    return getAngle2P(pts[1], pts[8])

def calculate_f10_l(pts):
    return getAngle2P(pts[2], pts[8])

def calculate_f11_l(pts):
    return getAngle2P(pts[3], pts[8])

def calculate_f12_l(pts):
    return getAngle2P(pts[4], pts[8])

def calculate_f13_l(pts):
    return getAngle2P(pts[5], pts[8])

def calculate_f14_l(pts):
    return getAngle2P(pts[6], pts[8])

def calculate_f15_l(pts):
    return getAngle2P(pts[7], pts[8])

def calculate_f8_r(pts):
    return getAngle2P(pts[16], pts[8])

def calculate_f9_r(pts):
    return getAngle2P(pts[15], pts[8])

def calculate_f10_r(pts):
    return getAngle2P(pts[14], pts[8])

def calculate_f11_r(pts):
    return getAngle2P(pts[13], pts[8])

def calculate_f12_r(pts):
    return getAngle2P(pts[12], pts[8])

def calculate_f13_r(pts):
    return getAngle2P(pts[11], pts[8])

def calculate_f14_r(pts):
    return getAngle2P(pts[10], pts[8])

def calculate_f15_r(pts):
    return getAngle2P(pts[9], pts[8])

### eye pos

def calculate_e9(pts):
    # 230221 : pow에 따른 차이 없음
    return ((pts[42][1] - pts[39][1]).pow(2))/face_width(pts).pow(2)

def calculate_e_f1(pts):
    # 230221 수정
    # (pts[38][0]).pow(2)/face_height(pts).pow(2) + (pts[43][0]).pow(2)/face_height(pts).pow(2) -> (pts[38][0])/face_height(pts) + (pts[43][0])/face_height(pts)
    # 눈 위치가 비대칭인 경우 극대화 되는 것을 막기 위해 pow를 사용하지 않음
    return (pts[38][0])/face_height(pts) + (pts[43][0])/face_height(pts)


### eye len

def calculate_e4(pts):
    # 230221 수정
    # 2*face_width(pts).pow(2) -> face_width(pts).pow(2)
    # 230221 pow 시 미세하게 더 커짐
    return ((pts[39][1] - pts[36][1]).pow(2) + (pts[45][1] - pts[42][1]).pow(2))\
           / face_width(pts).pow(2)


### eye shape

def dist_p_to_ab(p, a, b):
    x, y = 1, 0
    area = abs((a[x]-p[x]) * (b[y]-p[y]) - (a[y]-p[y]) * (b[x]-p[x]))
    AB = ((a[x]-b[x]).pow(2) + (a[y] - b[y]).pow(2)) ** 0.5
    return area / AB

def calculate_e2(pts):
    return (dist_p_to_ab(pts[37], pts[36], pts[39]) + dist_p_to_ab(pts[38], pts[36], pts[39])\
           + dist_p_to_ab(pts[43], pts[42], pts[45]) + dist_p_to_ab(pts[44], pts[42], pts[45])) \
           / face_height(pts)

def calculate_e3(pts):
    return (dist_p_to_ab(pts[40], pts[36], pts[39]) + dist_p_to_ab(pts[41], pts[36], pts[39])\
           + dist_p_to_ab(pts[46], pts[42], pts[45]) + dist_p_to_ab(pts[47], pts[42], pts[45])) \
           / face_height(pts)

### eb pos

def calculate_eb_e1(pts):
    # 230221 수정
    # 2*(face_height(pts).pow(2) -> (face_height(pts).pow(2)
    return ((pts[19][0]-pts[37][0]).pow(2) + (pts[24][0]-pts[44][0]).pow(2)) / (face_height(pts).pow(2))

def calculate_eb_eb(pts):
    ## 230221 새롭게 추가된 특징, 눈썹 사이 거리, 순위에 큰 영향을 미치지는 않을 듯
    return ((pts[22][1] - pts[21][1]).pow(2))/face_width(pts).pow(2)


### eb len
def calculate_eb1(pts):
    # 230221 수정
    # 2*face_width(pts).pow(2) -> face_width(pts).pow(2)
    # 230221 pow시 미세하게 더 커짐, 안하면 거의 변동 없음 (무조건 pow)
    return ((pts[21][1] - pts[17][1]).pow(2)+(pts[22][1] - pts[26][1]).pow(2))/(face_width(pts).pow(2))

### eb_shape

def calculate_eb7_l(pts):
    return getAngle3P(pts[17], pts[18], pts[19])

def calculate_eb8_l(pts):
    return getAngle3P(pts[18], pts[19], pts[20])

def calculate_eb9_l(pts):
    return getAngle3P(pts[19], pts[20], pts[21])

def calculate_eb7_r(pts):
    return getAngle3P(pts[22], pts[23], pts[24])

def calculate_eb8_r(pts):
    return getAngle3P(pts[23], pts[24], pts[25])

def calculate_eb9_r(pts):
    return getAngle3P(pts[24], pts[25], pts[26])



### lip pos
def calculate_l_f1(pts):
    return (pts[57][0] - pts[8][0]).pow(2) / face_height(pts).pow(2)

### lip len
def calculate_l1(pts):
    # 230221 수정
    # 2*face_width(pts).pow(2) -> face_width(pts).pow(2)
    # 230221 pow에 따른 변화 없음
    return ((pts[48][1] - pts[54][1]).pow(2) + (pts[60][1] - pts[64][1]).pow(2))/(face_width(pts).pow(2))

### lip shape
def calculate_l3(pts):
    # 230221 수정
    # (5*face_height(pts).pow(2) -> face_height(pts)
    # 세로 길이에 대한 실으로 바꾸고 pow(2).sum() 제거
    # 230221 pow 시 차이가 극대화 되며 이상한 값으로 훈련됨
    return ((pts[49][0] - pts[60][0])+(pts[50][0] - pts[61][0])+ (pts[51][0] - pts[62][0])
            +(pts[52][0] - pts[63][0])+(pts[53][0] - pts[64][0])) / (face_height(pts))

def calculate_l4(pts):
    # 230221 수정
    # (5*face_height(pts).pow(2) -> face_height(pts)
    # 세로 길이에 대한 실으로 바꾸고 pow(2).sum() 제거
    # 230221 pow 시 차이가 극대화 되며 이상한 값으로 훈련됨
    return ((pts[59][0] - pts[60][0])+(pts[58][0] - pts[67][0])+(pts[57][0] - pts[66][0])
            +(pts[56][0] - pts[65][0])+(pts[55][0] - pts[64][0])) / (face_height(pts))

### nose pos
def calculate_n_l_f1(pts):
    return (pts[51][0] - pts[33][0]).pow(2)/face_width(pts).pow(2)

### nose len
def calculate_n2(pts):
    return (pts[35][1] - pts[31][1]).pow(2)/face_width(pts).pow(2)

### nose shape
def calculate_n1(pts):
    # 230221 수정
    # 모든 콧대 랜드마크와 콧볼과 관련된 수식으로 변경
    # (pts[33][0] - pts[27][0]).pow(2)/face_height(pts).pow(2)
    #          -> ((pts[30][0] - pts[27][0]).pow(2)+(pts[30][0] - pts[28][0]).pow(2)
    #             +(pts[30][0] - pts[29][0]).pow(2))/face_height(pts).pow(2)
    return ((pts[30][0] - pts[27][0]).pow(2)+(pts[30][0] - pts[28][0]).pow(2)
            +(pts[30][0] - pts[29][0]).pow(2))/face_height(pts).pow(2)
