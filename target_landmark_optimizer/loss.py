from utils.utils_xfof import *


def create_trainable_range_pts(templete_pts, init_pts, trainable_start = 0, trainable_end = 16):

    trainable_pts = templete_pts.clone().detach()
    trainable_pts[trainable_start: trainable_end + 1] = init_pts[trainable_start: trainable_end + 1]  # 특정 구간을 고정 하기 위함

    return trainable_pts

def create_trainable_idx_pts(templete_pts, init_pts, trainable_idx=[]):

    trainable_pts = templete_pts.clone()
    for i in trainable_idx:
        trainable_pts[i] = init_pts[i]

    return trainable_pts

def inter_pos_loss(init_pts, orig_pts, train_range = [-1, -1], inter_w = 1e-5):
    ## shape 유지하면서 위치 이동할 때 필요한 loss
    internal_loss =0

    for i in range(train_range[0], train_range[1]+1):

        pre_idx = i-1
        post_idx = i+1

        if i == train_range[0] :
            pre_idx = train_range[1] - 1

        if i == (train_range[1]):
            post_idx = train_range[0]

        pred_curvature = (init_pts[post_idx] - init_pts[pre_idx])
        temp_curvature = (orig_pts[post_idx] - orig_pts[pre_idx])

        ratio_diff = (temp_curvature - pred_curvature).pow(2)

        internal_loss += ratio_diff.sum()

    return inter_w * internal_loss




def inter_len_loss(init_pts, orig_pts, train_range = [-1, -1], inter_w = 1e-10):

    # 모든 점들의 중심점과 거리를 유지하는 loss
    ## shape 유지하면서 위치 이동할 때 필요한 loss
    internal_loss =0

    for i in range(train_range[0], train_range[1]+1):
        pre_idx = i-1
        post_idx = i+1

        if i == train_range[0] :
            pre_idx = train_range[1]
        if i == train_range[1] :
            post_idx = train_range[0]


        init_ratio = divide((init_pts[i][1]-init_pts[post_idx][1]),(init_pts[i][1]-init_pts[pre_idx][1]))
        orig_ratio = divide((orig_pts[i][1]-orig_pts[post_idx][1]),(orig_pts[i][1]-orig_pts[pre_idx][1]))
        dist_diff = (orig_ratio - init_ratio).pow(2)

        internal_loss += dist_diff.sum()

    return inter_w * internal_loss

# def inter_len_loss(init_pts, orig_pts, train_range = [-1, -1], inter_w = 1e-10):
#     ## shape 유지하면서 위치 이동할 때 필요한 loss
#     internal_loss =0
#
#     for i in range(train_range[0], train_range[1]+1):
#
#         pre_idx = i-1
#         post_idx = i+1
#
#         if i == train_range[0] :
#             pre_idx = train_range[1] - 1
#
#         if i == (train_range[1]):
#             post_idx = train_range[0]
#
#         pred_curvature = (init_pts[post_idx] - init_pts[pre_idx])
#         temp_curvature = (orig_pts[post_idx] - orig_pts[pre_idx])
#
#         ratio_diff = (temp_curvature - pred_curvature).pow(2)
#
#         internal_loss += ratio_diff.sum()
#
#     return inter_w * internal_loss


# def inter_len_loss(init_pts, orig_pts, train_range = [-1, -1], inter_w = 1e-10):
#     ## shape 유지하면서 위치 이동할 때 필요한 loss
#     internal_loss =0
#
#     for i in range(train_range[0], train_range[1]+1):
#
#         pre_idx = i-1
#         post_idx = i+1
#
#         if i == train_range[0] :
#             pre_idx = train_range[1]
#
#         if i == (train_range[1]):
#             post_idx = train_range[0]
#
#         pred_angle = getSignedAngle2P(init_pts[i], init_pts[post_idx])
#         temp_angle = getSignedAngle2P(orig_pts[i], orig_pts[post_idx])
#
#         ratio_diff = (temp_angle - pred_angle) ** 2
#
#         internal_loss += ratio_diff
#
#     return inter_w * internal_loss
#
# def inter_len_loss(init_pts, orig_pts,
#                    train_range = [0, 16],
#                    inter_w = 1e-10,
#                    end_pt_idx = -1):
#
#     start, end = train_range
#
#     internal_loss = 0
#
#     for i in range(start, end+1):
#
#         if i == start :
#             loss = (orig_pts[start] - init_pts[i+1]).pow(2)
#         elif i == end :
#             if end_pt_idx > 0:
#                 loss = (init_pts[i] - orig_pts[end_pt_idx]).pow(2)
#             else:
#                 continue
#         else:
#             loss = (init_pts[i]- init_pts[i + 1]).pow(2)
#
#         internal_loss += loss.sum()
#
#     return inter_w * internal_loss

def divide(numerator, denominator):
    return (numerator + (1e-5)) / (denominator+ (1e-5))
#

def inter_width_shape_loss(init_pts, orig_pts,
                           target_pts,
                           train_range = [-1, -1],
                           inter_w = 1e-10):

    start, end = train_range

    internal_loss = 0

    for i in range(start, end+1):

        if i == start:
            init_diff = divide((init_pts[i][1]-init_pts[i+1][1]),(init_pts[i][1]-orig_pts[end][1]))
            target_diff = divide((target_pts[i][1] - target_pts[i+1][1]) , (target_pts[i][1] - target_pts[end][1]))
        elif i == end:
            init_diff = divide((init_pts[i][1]-orig_pts[start][1]),(init_pts[i][1]-init_pts[i-1][1]))
            target_diff = divide((target_pts[i][1] - target_pts[start][1]) , (target_pts[i][1] - target_pts[i-1][1]))
        else:
            init_diff = divide((init_pts[i][1]-init_pts[i+1][1]),(init_pts[i][1]-init_pts[i-1][1]))
            target_diff = divide((target_pts[i][1] - target_pts[i+1][1]) , (target_pts[i][1] - target_pts[i-1][1]))

        loss = (target_diff - init_diff) ** 2
        internal_loss += loss

    return inter_w * internal_loss


def inter_height_shape_loss(init_pts, orig_pts,
                           target_pts,
                           train_range = [-1, -1],
                           inter_w = 1e-10):

    start, end = train_range

    internal_loss = 0

    for i in range(start, end+1):

        if i == start:
            init_diff = divide((init_pts[i][0]-init_pts[i+1][0]),(init_pts[i][0]-orig_pts[end][0]))
            target_diff = divide((target_pts[i][0] - target_pts[i+1][0]) , (target_pts[i][0] - target_pts[end][0]))
        elif i == end:
            init_diff = divide((init_pts[i][0]-orig_pts[start][0]),(init_pts[i][0]-init_pts[i-1][0]))
            target_diff = divide((target_pts[i][0] - target_pts[start][0]) , (target_pts[i][0] - target_pts[i-1][0]))
        else:
            init_diff = divide((init_pts[i][0]-init_pts[i+1][0]),(init_pts[i][0]-init_pts[i-1][0]))
            target_diff = divide((target_pts[i][0] - target_pts[i+1][0]) , (target_pts[i][0] - target_pts[i-1][0]))

        loss = (target_diff - init_diff) ** 2
        internal_loss += loss

    return inter_w * internal_loss




def exter_loss(xfof_func, target_pts, input_pts, style_pts, style_w=1):

    target_xfof_pred = xfof_func(target_pts)
    style_xfof = xfof_func(style_pts)
    input_xfof = xfof_func(input_pts)
    target_xfof_gt = style_w*(style_xfof) + (1-style_w)*(input_xfof)
    print(input_xfof, style_xfof, style_w, target_xfof_gt)

    # try :
    #     external_loss = (style_w*(target_xfof_pred - style_xfof) + (1-style_w)*(target_xfof_pred - input_xfof)).pow(2).sum()
    # except:
    #     print("is angle")
    #     external_loss = style_w*(target_xfof_pred - style_xfof) + (1-style_w)*(target_xfof_pred - input_xfof)

    try :
        external_loss = (target_xfof_pred-target_xfof_gt).pow(2).sum()
    except:
        print("is angle")
        external_loss = (target_xfof_pred-target_xfof_gt)

    return external_loss



def total_exter_loss(trainable_pts, input_pts, style_pts, xfof_w, exter_w, style_w):

    total_external_loss = 0

    for xfof in xfof_w.keys():
        if xfof_w[xfof]>0:
            loss = exter_loss(eval("calculate_"+xfof), trainable_pts, input_pts, style_pts, style_w) # 함수 이름을 변수로 부를 수 있음..! 대박
            total_external_loss += exter_w * xfof_w[xfof] * loss

    return total_external_loss



def exter_angle_loss(init_pts, target_pts):
    externel_angle_loss = 0

    for move in range(0, 8):
        pred_dist = getAngle2P(init_pts, move, 8)
        target_dist = getAngle2P(target_pts, move, 8)
        externel_angle_loss += (pred_dist - target_dist)

    for move in range(9, 17):
        pred_dist = getAngle2P(init_pts, move, 8, True)
        target_dist = getAngle2P(target_pts, move, 8, True)
        externel_angle_loss += (pred_dist - target_dist)

    return externel_angle_loss