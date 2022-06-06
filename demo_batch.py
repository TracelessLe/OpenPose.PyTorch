import cv2
import os
import copy
import glob
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand


def load_model(model_type='coco', use_hand=False):
    if model_type == 'body25':
        model_path = './model/pose_iter_584000.caffemodel.pt'
    else:
        model_path = './model/body_pose_model.pth'
    body_estimation = Body(model_path, model_type)
    if use_hand:
        hand_estimation = Hand('model/hand_pose_model.pth')
    else:
        hand_estimation = None
    return body_estimation, hand_estimation


def inference(oriImg, model_type, body_estimation, hand_estimation, output_path='.'):
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset, model_type)

    if hand_estimation is not None:
        # detect hand
        hands_list = util.handDetect(candidate, subset, oriImg)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(
                peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(
                peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
    img_basename = os.path.basename(test_image_path)
    result_path = output_path+'/'+'result_' + \
        img_basename.split('.')[0]+'_'+model_type+'.png'
    cv2.imwrite(result_path, canvas)


if __name__ == "__main__":
    model_type = 'coco'  #  'body25'  # 
    body_estimation, hand_estimation = load_model(
        model_type=model_type, use_hand=True)
    
    # imgs_path_list = ['demo.jpg']
    input_imgs_dir = 'images'
    imgs_path_list = glob.glob(input_imgs_dir+'/*.*g')

    output_path = 'test_results'
    os.makedirs(output_path, exist_ok=True)
    
    for i, test_image_path in enumerate(imgs_path_list):
        print(f'processing: {i+1}/{len(imgs_path_list)}, {test_image_path}')
        oriImg = cv2.imread(test_image_path)  # B,G,R order
        inference(oriImg, model_type, body_estimation, hand_estimation, output_path)
