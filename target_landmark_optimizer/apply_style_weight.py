from torch.autograd import Variable
from optimizer import *
from utils.utils_save import *
from utils.utils_plot import *

# input images 경로
input_base_path = 'F:/xdst_orig/Caricature-Your-Face/ji_data/230321_sample/model/'
input_img_path = input_base_path + 'imgs/'

# style images 경로 (형태 예제)
geo_style_base_path = 'F:/xdst_orig/Caricature-Your-Face/ji_data/230321_sample/style/'
geo_style_img_path = geo_style_base_path + 'imgs/'

# input 및 style 의 landmark 경로
# input_land_path = input_base_path + 'landmarks/'
# geo_style_land_path = geo_style_base_path + 'landmarks/'
input_land_path = 'F:/xdst_orig/Caricature-Your-Face/preprocess/json_to_pts/230322/'
geo_style_land_path = 'F:/xdst_orig/Caricature-Your-Face/preprocess/json_to_pts/230322/'

# 저장 경로
save_base_path = 'F:/xdst_orig/Caricature-Your-Face/ji_data/230321_sample/outputs/230323/'

# opt 1. 폴더의 모든 이미지로 테스트
# input_img_list = os.listdir(templete_base_path+'imgs/')
# style_img_list = os.listdir(target_base_path+'imgs/')
# input_img_list.sort()
# style_img_list.sort()

# opt 2. 폴더의 단일 이미지로 테스트
input_img_list  =  ['AF1084.jpg'] # AM429 # AF891
style_img_list = ['see-you-in-my-19th-life_870_resize.jpg'] # mystical_78_resize # no-longer-a-heroine_164_resize

def apply_all_style_weight_stepwise(xfof_class_style_w, option_num=1):


    for input_fname in input_img_list:

        save_path = save_base_path + input_fname.rstrip('.jpg') + '/option'

        while True:
            if os.path.isdir(save_path+str(option_num)):
                print("Already File : ", save_path)
                option_num += 1
                continue

            elif option_num>100:
                print("Infinity Loop : Unvailed File Name")
            else:
                save_path = save_path+str(option_num) + '/'
                break

        os.makedirs(save_path, exist_ok=True)

        input_img_file_path = input_img_path + input_fname
        input_pts_file_path = input_land_path + input_fname.rstrip('.jpg') + '.txt'


        # image load (that user want to apply weight)
        input_img = cv2.imread(input_img_file_path)  # style을 templete으로 설정 # option
        # pts load
        input_pts = np.loadtxt(input_pts_file_path, delimiter=',')
        # tensor로 변경
        input_pts = torch.from_numpy(input_pts).float()

        # Optimizer
        learning_rate = 0.3  # 0.3

        init_pts = Variable(input_pts.clone(), requires_grad=True)
        pred_pts = [{'params': init_pts, 'lr': learning_rate}]
        optimizer = torch.optim.Adam(pred_pts, lr=learning_rate)
        # optimizer = torch.optim.SGD(pred_pts, lr=learning_rate, momentum=0.9)

        for style_fname in style_img_list:

            style_img_file_path = geo_style_img_path + style_fname
            style_pts_file_path = geo_style_land_path + style_fname.rstrip('.jpg') + '.txt'
            target_name = input_fname.rstrip('.jpg') + '_' + style_fname.rstrip('.jpg')

            # image load (that user want to apply weight)
            style_img = cv2.imread(style_img_file_path)  # style을 templete으로 설정 # option

            # pts load
            style_pts = np.loadtxt(style_pts_file_path, delimiter=',')

            # model, style 의 img/landmark scale 맞추기
            style_pts, style_img = resize_img_landmark(style_img, style_pts, 512)

            # tensor로 변경
            # 전체 tensor를 사용, 구조에서 나누고 전체적 훈련
            style_pts = torch.from_numpy(style_pts).float()


            reshaped_pts = component_optimizer(optimizer = optimizer,
                                               input_img=input_img,
                                               style_img=style_img,
                                               input_pts=input_pts,
                                               style_pts=style_pts,
                                               pred_pts=pred_pts,
                                               init_pts=init_pts,
                                               learning_rate=learning_rate,
                                               save_path=save_path,
                                               xfof_class_style_w=xfof_class_style_w)

            print("==========  "+input_fname+"_"+style_fname+"   ============")

            input_img = np.zeros((512, 512, 3), dtype="uint8") + 255

            reshaped_input_land_img = draw_all_landmark_line(input_img, reshaped_pts.detach().numpy(), title='result landmark')
            reshaped_templete_land_img = draw_each_component_landmark(reshaped_input_land_img, reshaped_pts.detach().numpy(), [0, 67])

            save_pts(reshaped_pts.detach().numpy(), target_name + '.txt', save_path)
            save_cv_img(reshaped_templete_land_img, target_name + '.png', save_path)
            save_options(xfof_class_style_w, option_num, save_path)

apply_all_style_weight_stepwise({"face_shape": 1,
                      "eye_pos": 1, "eye_len": 1, "eye_shape": 1,
                      "eyebrow_pos": 1, "eyebrow_len": 1, "eyebrow_shape": 1,
                      "lip_pos": 1, "lip_len": 1, "lip_shape": 1,
                      "nose_pos": 1, "nose_len": 1, "nose_shape": 1})

