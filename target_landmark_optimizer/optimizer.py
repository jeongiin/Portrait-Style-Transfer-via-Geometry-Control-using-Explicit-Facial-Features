from utils.utils_xfof import *

## xFoF 갯수, xFoF 가중치는 사용자화
# Shape Optimizer

from utils.utils_plot import *

from loss import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def component_optimizer(input_img, style_img, input_pts, style_pts,
                        pred_pts, init_pts,
                        learning_rate,
                        optimizer,
                        save_path,
                        external_w = {"face_shape": 0.00001,
                                      "eye_pos": 100,
                                      "eye_len": 10000,
                                      "eye_shape": 100000,
                                      "eyebrow_pos": 100000,
                                      "eyebrow_len": 100,
                                      "eyebrow_shape":0.01,
                                      "lip_pos": 100000,
                                      "lip_len": 1000000,
                                      "lip_shape":10000,
                                      "nose_pos":10000,
                                      "nose_len":10000,
                                      "nose_shape":10000},
                        xfof_class_style_w={},
                        xfof_face_shape_w = {"f7": 0,
                                                      "f8_l":1,"f9_l":1,"f10_l":1,"f11_l":1,
                                                      "f12_l":1,"f13_l":1,"f14_l":1,"f15_l":1,
                                                      "f8_r": 1, "f9_r": 1, "f10_r": 1, "f11_r": 1,
                                                      "f12_r": 1, "f13_r": 1, "f14_r": 1, "f15_r": 1,
                                                      "f16":0, "f17":0, "f18":0, "f19":0,
                                                      "f20":0, "f21":0, "f22":0},
                        xfof_eye_pos_w = {"e9": 1, "e_f1": 1},
                        xfof_eye_len_w = {"e4": 1},
                        xfof_eye_shape_w = {"e2": 1, "e3": 1, "e6": 0, "e12":0},
                        xfof_eyebrow_pos_w = {"eb_e1": 1, "eb_eb":1},
                        xfof_eyebrow_len_w = {"eb1": 1},
                        xfof_eyebrow_shape_w = {"eb2_l": 0, "eb3_l": 0, "eb4_l": 0, "eb5_l": 0,
                                                         "eb2_r": 0, "eb3_r": 0, "eb4_r": 0, "eb5_r": 0,
                                                         "eb7_l": 1, "eb8_l": 1, "eb9_l": 1,
                                                         "eb7_r": 1, "eb8_r": 1, "eb9_r": 1,
                                                         "eb10_l": 0, "eb11_l":0, "eb12_l":0,
                                                         "eb10_r":0, "eb11_r":0, "eb12_r":0},
                        xfof_lip_pos_w = {"l_f1": 1},
                        xfof_lip_len_w = {"l1": 1},
                        xfof_lip_shape_w = {"l2": 0, "l3": 1, "l4": 1, "l5": 0, "l6": 0},
                        xfof_nose_pos_w = {"n_f1": 0, "n_l_f1": 1},
                        xfof_nose_len_w = {"n2": 1},
                        xfof_nose_shape_w = {"n1": 1},

                        ):


    fixed_pts = init_pts.clone().detach()

    face_shape_epoch = int(500 * xfof_class_style_w["face_shape"])
    eye_pos_epoch = int(500 * xfof_class_style_w["eye_pos"])
    eye_len_epoch = int(500 * xfof_class_style_w["eye_len"])
    eye_shape_epoch = int(500 * xfof_class_style_w["eye_shape"])
    lip_pos_epoch = int(500 * xfof_class_style_w["lip_pos"])
    lip_len_epoch = int(500 * xfof_class_style_w["lip_len"])
    lip_shape_epoch = int(500 * xfof_class_style_w["lip_shape"])
    eyebrow_pos_epoch = int(500 * xfof_class_style_w["eyebrow_pos"])
    eyebrow_len_epoch = int(500 * xfof_class_style_w["eyebrow_len"])
    eyebrow_shape_epoch = int(500 * xfof_class_style_w["eyebrow_shape"])
    nose_pos_epoch = int(500 * xfof_class_style_w["nose_pos"])
    nose_len_epoch = int(500 * xfof_class_style_w["nose_len"])
    nose_shape_epoch = int(500 * xfof_class_style_w["nose_shape"])


    for face_shape_iter in range(face_shape_epoch):
        # 손실을 계산하고 출력
        # trainable_pts = create_trainable_range_pts(input_pts, init_pts, 0, 16)
        trainable_pts = create_trainable_idx_pts(fixed_pts, init_pts, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16])
        internal_high_loss = inter_height_shape_loss(trainable_pts, fixed_pts, fixed_pts, train_range=[0, 16], inter_w=1) # 1e-10
        external_loss = total_exter_loss(trainable_pts, input_pts, style_pts, xfof_face_shape_w, external_w["face_shape"], xfof_class_style_w["face_shape"]) # exter_w : 0.0001 , 0.000001, 3
        loss_face_shape = external_loss + internal_high_loss

        # 역전파 단계 전에, Optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인)
        # 갱신할 Variable들에 대한 모든 변화도를 0으로 만듦
        optimizer.zero_grad()

        # 역전파 단계: 모델의 매개변수에 대한 손실의 변화도를 계산
        loss_face_shape.backward(retain_graph=True)

        # Optimizer의 step 함수를 호출하면 매개변수가 갱신
        optimizer.step()

    # draw_landmark_save(templete_img, init_pts.detach().numpy(), init_pts.detach().numpy()[0:17], [], save_path=save_path+'face_line.jpg')
    # draw_landmark(templete_img, init_pts.detach().numpy(), color='b', marker='o', size=2, title='apply face_line xfof')
    fixed_pts = init_pts.clone().detach()


    for eye_pos_iter in range(eye_pos_epoch):
        trainable_pts = create_trainable_range_pts(fixed_pts, init_pts, 36, 47)
        internal_l_loss = inter_pos_loss(trainable_pts, fixed_pts, train_range=[36, 41], inter_w = 1) # inter_w = 1e-5
        internal_r_loss = inter_pos_loss(trainable_pts, fixed_pts, train_range=[42, 47], inter_w = 1) # inter_w = 1e-5
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_eye_pos_w, external_w["eye_pos"], xfof_class_style_w["eye_pos"]) # 230222 : 10000 # 1000부터 미동 있으나 너무 변동이 큼
        loss_eye_pos = external_loss + internal_l_loss + internal_r_loss

        optimizer.zero_grad()
        loss_eye_pos.backward(retain_graph=True)
        optimizer.step()

    # draw_landmark_save(templete_img, init_pts.detach().numpy(), init_pts.detach().numpy()[36:48], [], save_path=save_path+'eye_pos.jpg')
    # draw_landmark(templete_img, init_pts.detach().numpy(), title='apply eye pos xfof')
    fixed_pts = init_pts.clone().detach()

    for eye_len_iter in range(eye_len_epoch):
        trainable_pts = create_trainable_idx_pts(fixed_pts, init_pts, [36,37,38,40,41,43,44,45,46,47])
        internal_l_loss = inter_len_loss(trainable_pts, fixed_pts, train_range=[36, 41], inter_w=1) # model -> style w = 1e-10 / curvature
        internal_r_loss = inter_len_loss(trainable_pts, fixed_pts, train_range=[42, 47], inter_w=1)
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_eye_len_w, external_w["eye_len"], xfof_class_style_w["eye_len"])
        loss_eye_len = external_loss + internal_l_loss + internal_r_loss

        optimizer.zero_grad()
        loss_eye_len.backward(retain_graph=True)
        optimizer.step()



    # draw_landmark_save(templete_img, init_pts.detach().numpy(), init_pts.detach().numpy()[36:48], [], save_path=save_path+'eye_len.jpg')
    # draw_landmark(templete_img, init_pts.detach().numpy(), title='apply eye len xfof')
    fixed_pts = init_pts.clone().detach()


    for eye_shape_iter in range(eye_shape_epoch):
        trainable_pts = create_trainable_idx_pts(fixed_pts, init_pts, [37, 38, 40, 41, 43, 44, 46, 47])
        # trainable_pts = create_trainable_idx_pts(fixed_pts, init_pts, [36, 37, 38, 40, 41, 43, 44, 45, 46, 47])
        internal_l_loss = inter_width_shape_loss(trainable_pts, fixed_pts, fixed_pts, train_range=[36, 41], inter_w=(1-xfof_class_style_w["eye_shape"]) * 100) # 1000
        internal_r_loss = inter_width_shape_loss(trainable_pts, fixed_pts, fixed_pts, train_range=[42, 47], inter_w=(1-xfof_class_style_w["eye_shape"]) * 100) # 1000

        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_eye_shape_w, external_w["eye_shape"], xfof_class_style_w["eye_shape"]) # 2302222 : 1000000 # 잘나온거 : 100000
        external_loss += inter_width_shape_loss(trainable_pts, fixed_pts, style_pts, train_range=[36, 41], inter_w=xfof_class_style_w["eye_shape"] * 100) # 1000
        external_loss += inter_width_shape_loss(trainable_pts, fixed_pts, style_pts, train_range=[42, 47], inter_w=xfof_class_style_w["eye_shape"] * 100) # 1000

        loss_eye_shape = external_loss + internal_l_loss + internal_r_loss

        print(eye_shape_iter, loss_eye_shape)

        optimizer.zero_grad()
        loss_eye_shape.backward(retain_graph=True)
        optimizer.step()

    fixed_pts = init_pts.clone().detach()


    for eb_pos_iter in range(eyebrow_pos_epoch):
        trainable_pts = create_trainable_range_pts(fixed_pts, init_pts, 17, 26)
        internal_l_loss = inter_pos_loss(trainable_pts, fixed_pts, train_range=[17, 21], inter_w=1) # 1e-5
        internal_r_loss = inter_pos_loss(trainable_pts, fixed_pts, train_range=[22, 26], inter_w=1)
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_eyebrow_pos_w, exter_w=external_w["eyebrow_pos"], style_w=xfof_class_style_w["eyebrow_pos"]) # 230223 : 10000000
        loss_eb_pos = external_loss + internal_l_loss + internal_r_loss
        print(eb_pos_iter, loss_eb_pos)

        optimizer.zero_grad()
        loss_eb_pos.backward(retain_graph=True)
        optimizer.step()


    fixed_pts = init_pts.clone().detach()

    for eb_len_iter in range(eyebrow_len_epoch):
        trainable_pts = create_trainable_range_pts(fixed_pts, init_pts, 17, 26)
        internal_l_loss = inter_len_loss(trainable_pts, fixed_pts, train_range=[17, 21], inter_w=1) # model -> style w = 1e-10 / curvature
        internal_r_loss = inter_len_loss(trainable_pts, fixed_pts, train_range=[22, 26], inter_w=1)
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_eyebrow_len_w, exter_w=external_w["eyebrow_len"], style_w=xfof_class_style_w["eyebrow_len"])
        loss_eb_len = external_loss + internal_l_loss + internal_r_loss
        print(eb_len_iter, loss_eb_len)

        optimizer.zero_grad()
        loss_eb_len.backward(retain_graph=True)
        optimizer.step()


    fixed_pts = init_pts.clone().detach()

    for eb_shape_iter in range(eyebrow_shape_epoch):
        trainable_pts = create_trainable_idx_pts(fixed_pts, init_pts, [17, 18, 19, 20, 23, 24, 25, 26])
        # trainable_pts = create_trainable_range_pts(fixed_pts, init_pts, 17, 26)
        internal_l_loss = inter_width_shape_loss(trainable_pts, fixed_pts, fixed_pts, train_range=[17, 21], inter_w=1) # 1e-5
        internal_r_loss = inter_width_shape_loss(trainable_pts, fixed_pts, fixed_pts, train_range=[22, 26], inter_w=1)
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_eyebrow_shape_w, exter_w=external_w["eyebrow_shape"], style_w=xfof_class_style_w["eyebrow_shape"]) # 230222 : 100
        loss_eb_shape = external_loss + internal_l_loss + internal_r_loss
        print(eb_shape_iter, loss_eb_shape)

        optimizer.zero_grad()
        loss_eb_shape.backward(retain_graph=True)
        optimizer.step()


    fixed_pts = init_pts.clone().detach()

    for lip_pos_iter in range(lip_pos_epoch):
        trainable_pts = create_trainable_range_pts(fixed_pts, init_pts, 48, 67)
        internal_loss = inter_pos_loss(trainable_pts, fixed_pts, train_range=[48, 67], inter_w=1)
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_lip_pos_w, exter_w=external_w["lip_pos"], style_w=xfof_class_style_w["lip_pos"])
        loss_lip_loss = external_loss + internal_loss

        optimizer.zero_grad()
        loss_lip_loss.backward(retain_graph=True)
        optimizer.step()


    fixed_pts = init_pts.clone().detach()

    for lip_len_iter in range(lip_len_epoch):
        trainable_pts = create_trainable_range_pts(fixed_pts, init_pts, 48, 67)
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_lip_len_w, exter_w=external_w["lip_len"], style_w=xfof_class_style_w["lip_len"])
        internal_in_loss = inter_len_loss(trainable_pts, fixed_pts, train_range=[48, 59], inter_w=1) # style -> model 1e-10
        internal_out_loss = inter_len_loss(trainable_pts, fixed_pts, train_range=[60, 67], inter_w=1) # style -> model 1e-10

        loss_lip_len = external_loss + internal_in_loss + internal_out_loss

        optimizer.zero_grad()
        loss_lip_len.backward(retain_graph=True)
        optimizer.step()

    fixed_pts = init_pts.clone().detach()

    for lip_shape_iter in range(lip_shape_epoch):
        trainable_pts = create_trainable_idx_pts(fixed_pts, init_pts, [49,50,51,52,53,
                                                                       55,56,57,58,59])
        internal_wid_in_loss = inter_width_shape_loss(trainable_pts, fixed_pts, fixed_pts, train_range=[48, 59], inter_w=1)
        internal_wid_out_loss = inter_width_shape_loss(trainable_pts, fixed_pts, fixed_pts, train_range=[60, 67],inter_w=1)
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_lip_shape_w, exter_w=external_w["lip_shape"], style_w=xfof_class_style_w["lip_shape"])
        loss_lip_shape = external_loss + internal_wid_in_loss + internal_wid_out_loss

        optimizer.zero_grad()
        loss_lip_shape.backward(retain_graph=True)
        optimizer.step()

    fixed_pts = init_pts.clone().detach()

    for nose_pos_iter in range(nose_pos_epoch):
        trainable_pts = create_trainable_range_pts(fixed_pts, init_pts, 27, 35)
        internal_loss = inter_pos_loss(trainable_pts, fixed_pts, train_range=[27, 35], inter_w=1)
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_nose_pos_w, exter_w=external_w["nose_pos"], style_w=xfof_class_style_w["nose_pos"])
        loss_nose_loss = external_loss + internal_loss

        optimizer.zero_grad()
        loss_nose_loss.backward(retain_graph=True)
        optimizer.step()

    fixed_pts = init_pts.clone().detach()

    for nose_len_iter in range(nose_len_epoch):
        trainable_pts = create_trainable_range_pts(fixed_pts, init_pts, 31, 35)
        internal_loss = inter_len_loss(trainable_pts, fixed_pts, train_range=[31, 35], inter_w=1)
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_nose_len_w, exter_w=external_w["nose_len"], style_w=xfof_class_style_w["nose_len"])
        loss_nose_len = external_loss + internal_loss

        optimizer.zero_grad()
        loss_nose_len.backward(retain_graph=True)
        optimizer.step()


    fixed_pts = init_pts.clone().detach()

    for nose_shape_iter in range(nose_shape_epoch):
        # trainable_pts = create_trainable_idx_pts(fixed_pts, init_pts, [27,28,29,30,32,34])
        trainable_pts = create_trainable_idx_pts(fixed_pts, init_pts, [27,28,29])
        internal_height_loss = inter_height_shape_loss(trainable_pts, fixed_pts, style_pts, train_range=[27, 30], inter_w=1) # 1000
        internal_width_loss = inter_width_shape_loss(trainable_pts, fixed_pts, style_pts, train_range=[31, 35], inter_w=1) # 1000
        external_loss = total_exter_loss(trainable_pts, fixed_pts, style_pts, xfof_nose_shape_w, exter_w=external_w["nose_shape"], style_w=xfof_class_style_w["nose_shape"])
        loss_nose_shape = external_loss + internal_height_loss + internal_width_loss

        optimizer.zero_grad()
        loss_nose_shape.backward(retain_graph=True)
        optimizer.step()

    fixed_pts = init_pts.clone().detach()




    return fixed_pts

