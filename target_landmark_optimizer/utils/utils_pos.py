import numpy as np

def calculate_trans_y_x(source_component_pos, target_component_pos):
    trans_y = source_component_pos[0] - target_component_pos[0]
    trans_x = source_component_pos[1] - target_component_pos[1]

    return trans_y, trans_x

def translate_xy(y, x, target_yx):
    trans_yx = target_yx.copy()

    trans_yx[:,0] = target_yx[:,0] + y
    trans_yx[:,1] = target_yx[:,1] + x

    return trans_yx


def translate_centroid_position(standard_pos, trans_target_pos):

    trans_y, trans_x = calculate_trans_y_x(standard_pos[4], trans_target_pos[4]) # 코 기준 정렬
    trans_pos = translate_xy(trans_y, trans_x, trans_target_pos)

    return trans_pos

def translate_landmark(trans_y, trans_x, trans_target_landmark):
    trans_landmark = trans_target_landmark.copy()
    trans_landmark[:,0] = trans_target_landmark[:,0] + trans_y
    trans_landmark[:,1] = trans_target_landmark[:,1] + trans_x

    return trans_landmark

def create_pos_dictionary(pts):
    pos_dict = {}
    pos_dict["l_eye"] = (pts[36:42]).mean(axis=0)
    pos_dict["r_eye"] = (pts[42:48]).mean(axis=0)
    pos_dict["l_eyebrow"] = (pts[17:22]).mean(axis=0)
    pos_dict["r_eyebrow"] = (pts[22:27]).mean(axis=0)
    pos_dict["nose"] = (pts[27:36]).mean(axis=0)
    pos_dict["lip"] = (pts[48:68]).mean(axis=0)

    return pos_dict

def dict_to_array(dict):
    array = []
    for key in dict.keys():
        array.append(dict[key])

    return np.array(array)