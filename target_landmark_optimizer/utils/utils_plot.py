
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import copy
import numpy as np
import os

def draw_each_component_landmark(img, pts, active_range, point_color = (0,0,255)):

    result = copy.deepcopy(img)
    circle_size = 3
    font_size = 1.5
    font_size_small = 1
    font_size_very_small = 0.8
    thickness = 2
    thickness_small = 1
    min, max = active_range


    for i, point in enumerate(pts):
       if i >= min and i <= max:
            cv2.circle(result, (round(point[1]), round(point[0])), 5, point_color, -1)
            # cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size_very_small,
            #             point_color, thickness_small)
       else:
            cv2.circle(result, (round(point[1]), round(point[0])), 5, (255, 255, 255), -1)

    return result

# def draw_each_component_landmark_with_num(img, pts):
#
#     result = copy.deepcopy(img)
#     circle_size = 3
#     font_size = 1.5
#     font_size_small = 1
#     font_size_very_small = 0.8
#     thickness = 2
#     thickness_small = 1
#
#     for i, point in enumerate(pts):
#         if (i < 17):
#             cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 255), -1)  # 얼굴형
#             cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), thickness )
#
#         if (16 < i < 22):
#             cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 255), -1)  # 왼쪽 눈썹
#             cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), thickness )
#
#
#         if (21 < i < 27):
#             cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 255), -1)  # 오른쪽 눈썹
#             cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), thickness )
#
#
#         if (26 < i < 36):
#             cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 255), -1)  # 코
#             cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size_small, (0, 0, 255), thickness_small )
#
#
#         if (35 < i < 42):
#             cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 255), -1)  # 왼쪽 눈
#             cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), thickness )
#
#
#         if (41 < i < 48):
#             cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 255), -1)  # 오른쪽 눈
#             cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), thickness )
#
#
#         if (47 < i < 60):
#             cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 255), -1)  # 입
#             if i<55:
#                 cv2.putText(result, str(i), (round(point[1]), round(point[0])-4), cv2.FONT_HERSHEY_PLAIN, font_size_small, (0, 0, 255), thickness_small)
#             else:
#                 cv2.putText(result, str(i), (round(point[1]), round(point[0])+4), cv2.FONT_HERSHEY_PLAIN, font_size_small, (0, 0, 255), thickness_small)
#
#         if (59 < i < 68):
#             cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 255), -1)  # 입
#             if i<64:
#                 cv2.putText(result, str(i), (round(point[1]), round(point[0])-2), cv2.FONT_HERSHEY_PLAIN, font_size_very_small, (0, 0, 255), thickness_small)
#             else:
#                 cv2.putText(result, str(i), (round(point[1]-2), round(point[0])+10), cv2.FONT_HERSHEY_PLAIN, font_size_very_small, (0, 0, 255), thickness_small)
#
#
#     return result

def draw_each_component_landmark_with_num(img, pts):

    result = copy.deepcopy(img)
    circle_size = 3
    font_size = 1.5
    font_size_small = 1
    font_size_very_small = 0.8
    thickness = 2
    thickness_small = 1

    for i, point in enumerate(pts):
        if (i < 17):
            cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 200, 0), -1)  # 얼굴형
            cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 200, 0), thickness )

        if (16 < i < 22):
            cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 200), -1)  # 왼쪽 눈썹
            cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 200), thickness )


        if (21 < i < 27):
            cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (200, 0, 0), -1)  # 오른쪽 눈썹
            cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (200, 0, 0), thickness )


        if (26 < i < 36):
            cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (150, 0, 150), -1)  # 코
            cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size_small, (150, 0, 150), thickness )


        if (35 < i < 42):
            cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (0, 0, 150), -1)  # 왼쪽 눈
            cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 150), thickness )


        if (41 < i < 48):
            cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (150, 0, 0), -1)  # 오른쪽 눈
            cv2.putText(result, str(i), (round(point[1]), round(point[0])), cv2.FONT_HERSHEY_PLAIN, font_size, (150, 0, 0), thickness )


        if (47 < i < 60):
            cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (200, 200, 0), -1)  # 입
            if i<55:
                cv2.putText(result, str(i), (round(point[1]), round(point[0])-4), cv2.FONT_HERSHEY_PLAIN, font_size_small, (200, 200, 0), thickness)
            else:
                cv2.putText(result, str(i), (round(point[1]), round(point[0])+4), cv2.FONT_HERSHEY_PLAIN, font_size_small, (200, 200, 0), thickness)

        if (59 < i < 68):
            cv2.circle(result, (round(point[1]), round(point[0])), circle_size, (200, 200, 0), -1)  # 입
            if i<64:
                cv2.putText(result, str(i), (round(point[1]), round(point[0])-2), cv2.FONT_HERSHEY_PLAIN, font_size_very_small, (200, 200, 0), thickness)
            else:
                cv2.putText(result, str(i), (round(point[1]-2), round(point[0])+10), cv2.FONT_HERSHEY_PLAIN, font_size_very_small, (200, 200, 0), thickness)


    return result


def draw_landmark(img, pts, color='b', marker='o', size = 2, cvtColor_flag=True, title=''):
    result = copy.deepcopy(img)
    if cvtColor_flag:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    for i, point in enumerate(pts):
        plt.plot(round(point[1]), round(point[0]), marker=marker, color=color, markersize=size)
        # cv2.circle(result, (round(point[1]), round(point[0])), 3, marker=marker, color=color, size=-1)

    plt.suptitle(title)
    plt.imshow(result)
    plt.show()


def draw_landmark_save(img, init_pts, active_pts, fixed_pts, save_path = '', cvtColor_flag=False):
    result = copy.deepcopy(img)
    if cvtColor_flag:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    result = copy.deepcopy(result)

    for i, point in enumerate(init_pts):
        cv2.circle(result, (round(point[1]), round(point[0])), 5, (120, 120, 120), -1)

    for i, point in enumerate(fixed_pts):
        cv2.circle(result, (round(point[1]), round(point[0])), 5, (255, 0, 0), -1)

    for i, point in enumerate(active_pts):
        cv2.circle(result, (round(point[1]), round(point[0])), 5, (0, 0, 255), -1)


    cv2.imwrite(save_path, result)


def draw_multiple_landmark(img, pts_order, color_order, marker_order, size = 2):
    result = copy.deepcopy(img)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    for i, pts in enumerate(pts_order):
        for point in pts:
            plt.plot(round(point[1]), round(point[0]), marker=marker_order[i], color=color_order[i], markersize=size)

    result_temp = copy.deepcopy(result)

    plt.imshow(result)
    plt.show()

    return result_temp

def draw_selected_landmark_line(img, pts, size=2, cvtColor_flag=False, point_color = (0, 0, 255), title='', show=False, close=False):
    result = copy.deepcopy(img)
    if cvtColor_flag:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    x_y_pts = xy_to_yx(pts)
    # x_y_pts = np.array(x_y_pts)
    # x_y_pts = np.int32([x_y_pts])


    result = cv2.polylines(result, np.int32([x_y_pts]), close, point_color, size) # face

    result_temp = copy.deepcopy(result)

    if show :
        plt.suptitle(title)
        plt.imshow(result)
        plt.show()

    return result_temp

def draw_all_landmark_line(img, pts, size=2, cvtColor_flag=False, color = (0, 0, 255), title='', show=False):
    result = copy.deepcopy(img)
    if cvtColor_flag:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    x_y_pts = xy_to_yx(pts)
    # x_y_pts = np.array(x_y_pts)
    # x_y_pts = np.int32([x_y_pts])

    '''
    얼굴 (0,200,0)
    왼쪽 눈썹 (0,0,200)
    오른쪽 눈썹 (200,0,0)
    코 (150,0,150)
    왼쪽 눈 (0,0,150)
    오른쪽 눈 (150,0,0)
    입 (200,200,0)
    '''
    result = cv2.polylines(result, np.int32([x_y_pts[0:17]]), False, color, size) # face
    result = cv2.polylines(result, np.int32([x_y_pts[17:22]]), False, color, size) # l eb
    result = cv2.polylines(result, np.int32([x_y_pts[22:27]]), False, color, size) # r eb
    result = cv2.polylines(result, np.int32([x_y_pts[27:31]]), False, color, size) # nose
    result = cv2.polylines(result, np.int32([x_y_pts[31:36]]), False, color, size) # nose
    result = cv2.polylines(result, np.int32([x_y_pts[36:42]]), True, color, size) # l e
    result = cv2.polylines(result, np.int32([x_y_pts[42:48]]), True, color, size) # r e
    result = cv2.polylines(result, np.int32([x_y_pts[48:60]]), True, color, size) # mouth
    result = cv2.polylines(result, np.int32([x_y_pts[60:68]]), True, color, size)  # mouth

    # result = cv2.polylines(result, np.int32([x_y_pts[0:17]]), False, (0,200,0), size) # face
    # result = cv2.polylines(result, np.int32([x_y_pts[17:22]]), False, (0,0,200), size) # l eb
    # result = cv2.polylines(result, np.int32([x_y_pts[22:27]]), False, (200,0,0), size) # r eb
    # result = cv2.polylines(result, np.int32([x_y_pts[27:31]]), False, (150,0,150), size) # nose
    # result = cv2.polylines(result, np.int32([x_y_pts[31:36]]), False, (150,0,150), size) # nose
    # result = cv2.polylines(result, np.int32([x_y_pts[36:42]]), True, (0,0,150), size) # l e
    # result = cv2.polylines(result, np.int32([x_y_pts[42:48]]), True, (150,0,0), size) # r e
    # result = cv2.polylines(result, np.int32([x_y_pts[48:60]]), True, (200,200,0), size) # mouth
    # result = cv2.polylines(result, np.int32([x_y_pts[60:68]]), True, (200,200,0), size)  # mouth

    result_temp = copy.deepcopy(result)

    if show :
        plt.suptitle(title)
        plt.imshow(result)
        plt.show()

    return result_temp


def xy_to_yx(xy_pts):
    yx_pts = []
    for x, y in xy_pts:
        yx_pts.append([y,x])
    return np.array(yx_pts)

def resize_img_landmark(img, pts, scale):
    ## 주의 : 정사각형 이미지에만 적용 가능
    h, w, c = img.shape
    rescale_pts = pts * (scale/h)
    rescale_img = cv2.resize(img, (scale, scale))

    return rescale_pts, rescale_img














