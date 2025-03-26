import os
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import cv2
import math, cmath
from copy import deepcopy
import random
from sklearn.model_selection import train_test_split
import psutil


def readline(line):
    ''' read one line of txt file
    Args:
        line: one line of txt file. Format: angle data0 data1 ... dataX
        By default, each line has 1 + 500 data.
    Returns:
        angle: angle of this line
        data: sonar data
    '''
    line = line.split()
    line = list(map(float, line))
    angle = line[0]
    data = line[1:]
    return angle, data


def read_txt(path, angle_range=400):
    ''' read sonar data from txt file
    Args:
        path: path of txt file
        angle_range: range of angle (1 gradian = 0.9 degree), default: 400
    Returns:
        sonar_data: 2d array, shape: (angle_range, 500)
        start_angle: start angle of sonar data
        end_angle: end angle of sonar data
    '''
    sonar_data = np.zeros((angle_range, 500))
    with open(path, 'r') as f:
        lines = f.readlines()
        lines.sort(key = lambda x : float(x.split(" ")[0]))
        
        start_angle = float(lines[0].split(' ')[0])
        end_angle = float(lines[-1].split(' ')[0])
        for line in lines:
            angle, data = readline(line)
            if len(data) == 500:
                sonar_data[int(angle)] = data
                # print(angle)
                # print(sonar_data[int(angle)])
            # print(sonar_data.shape)
    return sonar_data, int(start_angle), int(end_angle)

def int2str(num, length=6):
    ''' convert int to str, and add 0 before the number
    Args:
        num: int number
        length: length of str, default: 6
    Returns:
        str_num: str number
    '''
    str_num = str(num)
    while len(str_num) < length:
        str_num = '0' + str_num
    return str_num

def object_polar2cart(sonar_data, obj, rotate=True):
    ''' erase other objects; transform from polar to cartesian; slice the object with rectangle
    Args:
        sonar_data: 3d array, shape: (channels, angle_range, 500)
        obj: [ymin, ymax, xmin, xmax]
        rotate: whether rotate the sonar data to make the object at around 0 degree
    Returns:
        polar_data: 3d array, like sonar picture and the sonar is at center of pic, shape: (channels, 1000, 1000)
        [xmin, ymin, xmax, ymax]: the object's position in polar_data (maybe after rotation)
    '''
    # if sonar_data.ndim == 2:
    #     sonar_data = sonar_data[:, np.newaxis]
    empty_data = np.zeros_like(sonar_data)
    y_mean = 0
    if rotate:
        y_mean = int((obj[0] + obj[1]) / 2)
    empty_data[obj[0]:obj[1], obj[2]:obj[3]] = sonar_data[obj[0]:obj[1], obj[2]:obj[3]]

    # sonar should be at center of pic
    polar_data = cv2.warpPolar(empty_data, (1000, 1000), (500, 500), 500,
                               cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # center rotate y_mean degrees
    M = cv2.getRotationMatrix2D((500, 500), y_mean * 0.9, 1)
    polar_data = cv2.warpAffine(polar_data, M, (1000, 1000))

    left_up = cmath.rect(obj[2], math.radians((obj[0] - y_mean) * 0.9))
    left_down = cmath.rect(obj[2], math.radians((obj[1] - y_mean) * 0.9))
    right_up = cmath.rect(obj[3], math.radians((obj[0] - y_mean) * 0.9))
    right_down = cmath.rect(obj[3], math.radians((obj[1] - y_mean) * 0.9))

    max_r = cmath.rect(obj[3], math.radians(0)).real

    x = [left_up.real+500, left_down.real+500, right_up.real+500, right_down.real+500, max_r+500]
    xmin = min(x)
    xmax = max(x)
    xmin = math.floor(xmin)
    xmax = math.ceil(xmax)
    xmax = xmax + 5
    xmin = xmin - 5
    
    y = [left_up.imag+500, left_down.imag+500, right_up.imag+500, right_down.imag+500]
    ymin = min(y)
    ymax = max(y)
    ymin = math.floor(ymin)
    ymax = math.ceil(ymax)
    ymax = ymax + 5
    ymin = ymin - 5

    polar_data = polar_data[ymin:ymax, xmin:xmax]
    return polar_data, [xmin, ymin, xmax, ymax]


def norm_distance(sonar_data):
    ''' normalize sonar data based on distance. The center distance is the anchor.
        n = (pixel_x - anchor_x) / total_x * 
        pixel_strength = pixel_strength * (1 + n)
    Args:
        sonar_data: 2d array, shape: (400, 500);
                    The 250 column is the center of sonar data, and they don't have change.

    Returns:
        norm_data: 2d array, shape: (400, 500)
    '''

class SonarData():
    ''' SonarData class
    '''
    def __init__(self, channels=3,concate_len=2):
        self.save_file_idx = 0

        self.past_objects = []
        self.past_sonar_datas = []

        self.past_data_paths = []
        self.past_label_paths = []
        self.channels = channels
        self.concate_len=concate_len
        self.bias_x=0
        self.bias_y=0

    def __checklen__(self):
        ''' check the length of past_objects and past_sonar_datas
        '''
        assert len(self.past_objects) == len(self.past_sonar_datas), 'past_objects and past_sonar_datas have different length'
        while len(self.past_objects) > self.channels:
            self.past_objects.pop(0)
            self.past_sonar_datas.pop(0)
            self.past_data_paths.pop(0)
            self.past_label_paths.pop(0)
            
    def __concatlen__(self):
        assert len(self.past_objects) == len(self.past_sonar_datas), 'past_objects and past_sonar_datas have different length'
        while len(self.past_objects) > self.concate_len:
            self.past_objects.pop(0)
            self.past_sonar_datas.pop(0)
            self.past_data_paths.pop(0)
            self.past_label_paths.pop(0)
    
    def __clean__(self):
        ''' clean the data
        '''
        self.past_objects = []
        self.past_sonar_datas = []
        self.past_data_paths = []
        self.past_label_paths = []

    def __skip_scanning__(image, skip, start_angle,end_angle, start, step):
        while start<start_angle:
            start+=step
        for i in range(start, end_angle, step):
            for j in range(skip):
                image[i + j, :] = 0
        return image

    def __process_one_pic_1ch__(self, sonar_data_path, label_file_path, pic_save_dir_path, label_save_dir_path, sonar_str):
        ''' process one pic, and it will be split into many pics contains different objects.
            This function only works on 1 channel (do not consider time axis)
        Label file format:
            The label file is default format.
            For human object: human id | state_str | xmin | ymin | xmax | ymax
            For noise object: state_str | xmin | ymin | xmax | ymax
            Note: some non-target human's id is -1
        Args:
            sonar_data_path: path of sonar data
            label_file_path: path of label file
            pic_save_dir_path: path to save processed data
            label_save_dir_path: path to save label file
        The saved pics will be *pic_save_dir_path*/*pic_name*/*idx*-state.png
        The saved label file will be *label_save_dir_path*/*pic_name*.txt
        Saved label file format: human id | state_str | pos | sonar_str | sonar_data_path | label_file_path
        '''
        sonar_data, start_angle, end_angle = read_txt(sonar_data_path)

        human_id_list, state_list, obj_list = read_default_label(label_file_path)

        for human_id, state, obj in zip(human_id_list, state_list, obj_list):
            polar_data, pos = object_polar2cart(sonar_data, obj)
            save_data_path = os.path.join(pic_save_dir_path, int2str(self.save_file_idx) + '.png')
            save_label_path = os.path.join(label_save_dir_path, int2str(self.save_file_idx) + '.txt')

            with open(save_label_path, 'w') as f:
                f.write(str(human_id) + '\n' + str(state) + '\n' + str(obj[0]) + ',' + str(obj[1]) + ',' + str(obj[2]) + ',' + str(obj[3]) + '\n' +
                        sonar_str + '\n' + sonar_data_path + '\n' + label_file_path + '\n')
            cv2.imwrite(save_data_path, polar_data)
            self.save_file_idx += 1
            
    def __process_one_pic_1ch_skip__(self, sonar_data_path, label_file_path, pic_save_dir_path, label_save_dir_path, sonar_str):

        sonar_data, start_angle, end_angle = read_txt(sonar_data_path)

        human_id_list, state_list, obj_list = read_default_label(label_file_path)
        skip_data=self.__skip_scanning__(sonar_data,)
        for human_id, state, obj in zip(human_id_list, state_list, obj_list):
            polar_data, pos = object_polar2cart(sonar_data, obj)
            save_data_path = os.path.join(pic_save_dir_path, int2str(self.save_file_idx) + '.png')
            save_label_path = os.path.join(label_save_dir_path, int2str(self.save_file_idx) + '.txt')

            with open(save_label_path, 'w') as f:
                f.write(str(human_id) + '\n' + str(state) + '\n' + str(obj[0]) + ',' + str(obj[1]) + ',' + str(obj[2]) + ',' + str(obj[3]) + '\n' +
                        sonar_str + '\n' + sonar_data_path + '\n' + label_file_path + '\n')
            cv2.imwrite(save_data_path, polar_data)
            self.save_file_idx += 1
        

    def __process_one_pic_multi_ch__(self, sonar_data_path, label_file_path, pic_save_dir_path, label_save_dir_path, sonar_str):
        ''' process one pic, and it will be split into many pics contains different objects.
            This function only works on multi channel (consider time axis)
            Use IoU to get same object instead of human id.
        Label file format:
            The label file is default format.
            For human object: human id | state_str | xmin | ymin | xmax | ymax
            For noise object: state_str | xmin | ymin | xmax | ymax
            Note: some non-target human's id is -1
        Args:
            sonar_data_path: path of sonar data
            label_file_path: path of label file
            pic_save_dir_path: path to save processed data
            label_save_dir_path: path to save label file
        The saved pics will be *pic_save_dir_path*/*pic_name*/*idx*-state.png
        The saved label file will be *label_save_dir_path*/*pic_name*.txt
        Saved label file format: human id | state_str | pos | sonar_str | sonar_data_path | label_file_path
        '''
        sonar_data, start_angle, end_angle = read_txt(sonar_data_path)
        human_id_list, state_list, obj_list = read_default_label(label_file_path)
        self.past_objects.append(obj_list)
        self.past_sonar_datas.append(sonar_data)
        self.past_data_paths.append(sonar_data_path)
        self.past_label_paths.append(label_file_path)
        self.__checklen__()
        # Use IoU to get same object positions from past frames with the current frame
        candidate_regions, candidate_datas = self.compare_objects()
        if candidate_regions is None and candidate_datas is None:
            return
        for human_id, state, obj, data in zip(human_id_list, state_list, candidate_regions, candidate_datas):
            polar_data, pos = object_polar2cart(data, obj)
            save_data_path = os.path.join(pic_save_dir_path, int2str(self.save_file_idx) + '.png')
            save_label_path = os.path.join(label_save_dir_path, int2str(self.save_file_idx) + '.txt')
            with open(save_label_path, 'w') as f:
                f.write(str(human_id) + '\n' + str(state) + '\n' + str(obj[0]) + ',' + str(obj[1]) + ',' + str(obj[2]) + ',' + str(obj[3]) + '\n' +
                        sonar_str + '\n' + ', '.join(self.past_data_paths) + '\n' + ', '.join(self.past_label_paths) + '\n')
            # cv2 stores image in BGR format
            cv2.imwrite(save_data_path, polar_data)
            self.save_file_idx += 1
        
    def __process_one_concat_multi_ch__(self, sonar_data_path, label_file_path, pic_save_dir_path, label_save_dir_path, sonar_str):
        sonar_data, start_angle, end_angle = read_txt(sonar_data_path)
        human_id_list, state_list, obj_list = read_default_label(label_file_path)
        self.past_objects.append(obj_list)
        self.past_sonar_datas.append(sonar_data)
        self.past_data_paths.append(sonar_data_path)
        self.past_label_paths.append(label_file_path)
        self.__concatlen__()
        # Use IoU to get same object positions from past frames with the current frame
        candidate_regions, candidate_datas = self.compare_objects(concat=True)
        if candidate_regions is None and candidate_datas is None:
            return
        for human_id, state, obj, data in zip(human_id_list, state_list, candidate_regions, candidate_datas):
            polar_data, pos = object_polar2cart(data, obj)
            save_data_path = os.path.join(pic_save_dir_path, int2str(self.save_file_idx) + '.png')
            save_label_path = os.path.join(label_save_dir_path, int2str(self.save_file_idx) + '.txt')
            with open(save_label_path, 'w') as f:
                f.write(str(human_id) + '\n' + str(state) + '\n' + str(obj[0]) + ',' + str(obj[1]) + ',' + str(obj[2]) + ',' + str(obj[3]) + '\n' +
                        sonar_str + '\n' + ', '.join(self.past_data_paths) + '\n' + ', '.join(self.past_label_paths) + '\n')
            # cv2 stores image in BGR format
            cv2.imwrite(save_data_path, polar_data)
            self.save_file_idx += 1
            
    def __process_one_bias_multi_ch__(self, sonar_data_path, label_file_path, pic_save_dir_path, label_save_dir_path, sonar_str):
        sonar_data, start_angle, end_angle = read_txt(sonar_data_path)
        human_id_list, state_list, obj_list = read_default_label(label_file_path)
        self.past_objects.append(obj_list)
        self.past_sonar_datas.append(sonar_data)
        self.past_data_paths.append(sonar_data_path)
        self.past_label_paths.append(label_file_path)
        self.__checklen__()
            
        # Use IoU to get same object positions from past frames with the current frame
        candidate_regions, candidate_datas = self.compare_objects(bias=True)
        if candidate_regions is None and candidate_datas is None:
            return
        for human_id, state, obj, data in zip(human_id_list, state_list, candidate_regions, candidate_datas):
            polar_data, pos = object_polar2cart(data, obj)
            save_data_path = os.path.join(pic_save_dir_path, int2str(self.save_file_idx) + '.png')
            save_label_path = os.path.join(label_save_dir_path, int2str(self.save_file_idx) + '.txt')
            with open(save_label_path, 'w') as f:
                f.write(str(human_id) + '\n' + str(state) + '\n' + str(obj[0]) + ',' + str(obj[1]) + ',' + str(obj[2]) + ',' + str(obj[3]) + '\n' +
                        sonar_str + '\n' + ', '.join(self.past_data_paths) + '\n' + ', '.join(self.past_label_paths) + '\n')
            # cv2 stores image in BGR format
            cv2.imwrite(save_data_path, polar_data)
            self.save_file_idx += 1

    def compare_objects(self,bias=False,concat=False):
        empty_data = np.zeros_like(self.past_sonar_datas[-1])
        if not concat:
            while len(self.past_objects) < self.channels:
                self.past_objects.insert(0, self.past_objects[-1])
                self.past_sonar_datas.insert(0, empty_data)
                self.past_data_paths.insert(0, 'empty')
                self.past_label_paths.insert(0, 'empty')
        else:
            if len(self.past_objects) < self.concate_len:
                while len(self.past_objects) < self.channels:
                    self.past_objects.insert(0, self.past_objects[-1])
                    self.past_sonar_datas.insert(0, empty_data)
                    self.past_data_paths.insert(0, 'empty')
                    self.past_label_paths.insert(0, 'empty')
            else:
                split_seed=random.random()*(0.8-0.2)+0.2
                image_center=self.past_sonar_datas[0]
                obj_center_list=[]
                if random.random()>0.5:
                    direction = 1
                else:
                    direction = -1
                #print(self.past_objects)
                if len(self.past_objects[0])!=len(self.past_objects[1]):
                    return None,None
                for index in range(len(self.past_objects[0])):
                    y_min_1=self.past_objects[1][index][0]
                    y_max_1=self.past_objects[1][index][1]
                    y_min_2=self.past_objects[0][index][0]
                    y_max_2=self.past_objects[0][index][1]
                    x_min_1=self.past_objects[1][index][2]
                    x_max_1=self.past_objects[1][index][3]
                    x_min_2=self.past_objects[0][index][2]
                    x_max_2=self.past_objects[0][index][3]
                    
                    if direction==1:
                        end_angle=int((y_max_1-y_min_1)*split_seed+y_min_1)
                        start_angle=y_min_1
                    else:
                        end_angle=int(y_max_1-(y_max_1-y_min_1)*split_seed)
                        start_angle=y_max_1
                    image1=self.past_sonar_datas[-1]
                    image_center[start_angle:end_angle,x_min_1:x_max_1]=image1[start_angle:end_angle,x_min_1:x_max_1]
                    obj_center=[min(y_min_1,y_min_2),max(y_max_1,y_max_2),min(x_min_1,x_min_2),max(x_max_1,x_max_2)]
                    obj_center_list.append(obj_center)
                self.past_sonar_datas.insert(1,image_center)
                self.past_objects.insert(1,obj_center_list)
                self.past_data_paths.insert(1, 'empty')
                self.past_label_paths.insert(1, 'empty')
        total_frames = len(self.past_objects)
        # TODO: candidate_data对于每个object都是不同的，所以最后并不是只有一个candidate_data
        # 但是candidate_data的中间结果不能直接输出，需要想个办法解决
        # 所有object的基准都是以current frame为主的，也就是说candidate region不会超过current frame的范围
        # 但是再往前的frame计算是否是同一个obj是基于past objects的，所以要在代码逻辑上注意
        # 第一次loop之后，len(candidata_datas) == len(candidate_regions) == len(self.past_objects[-1])

        iou_threshold=0.3
        last_objects = deepcopy(self.past_objects[-1])
        detect_flags = [True for _ in range(len(last_objects))]
        candidate_datas = [self.past_sonar_datas[-1] for _ in range(len(last_objects))]
        candidate_regions = deepcopy(last_objects)
        if len(last_objects) == 0:
            return None, None
        for i in range(total_frames - 1, 0, -1):
            # former frame
            objects_old = self.past_objects[i - 1]
            for idx, region in enumerate(candidate_regions):
                # compare last_objects and previous objects (objects_old). combine regions
                # then last_objects => previous objects
                last_obj = last_objects[idx]
                last_candidate_data = candidate_datas[idx]
                if last_obj is None:
                    assert detect_flags[idx] == False
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                            last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    continue
                if len(objects_old) == 0:
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                            last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = None
                    detect_flags[idx] = False
                    continue
                for object_old in objects_old:
                    if calculate_iou_on_small(last_obj, object_old) > iou_threshold:
                        # combine the same object in different frames
                        x_seed=random.random()
                        y_seed=random.random()
        
                        if x_seed<0.5:
                            self.bias_x=-1
                        else:
                            self.bias_x=1
                        if y_seed<0.5:
                            self.bias_y=-1
                        else:
                            self.bias_y=1
                            
                        if not bias or i != total_frames-1:
                            temp_data = self.get_clean_data(self.past_sonar_datas[i - 1], object_old)
                        else:
                            temp_data,obj_new = self.extend_data(self.past_sonar_datas[i - 1], object_old)
                        if bias and i == total_frames-1:
                            candidate_region, candidate_data = self.combine_objects(region, obj_new,
                                            last_candidate_data, temp_data)
                        else:
                            candidate_region, candidate_data = self.combine_objects(region, object_old,
                                            last_candidate_data, temp_data)
                        candidate_regions[idx] = candidate_region
                        candidate_datas[idx] = candidate_data
                        last_objects[idx] = object_old
                        break
                    else:
                        candidate_region, candidate_data = self.combine_objects(region, region, 
                                            last_candidate_data, empty_data)
                        candidate_regions[idx] = candidate_region
                        candidate_datas[idx] = candidate_data
                        last_objects[idx] = None
                        detect_flags[idx] = False
        
        return candidate_regions, candidate_datas
       
    
    def get_clean_data(self, sonar_data, obj):
        empty_data = np.zeros_like(sonar_data)
        empty_data[obj[0]:obj[1], obj[2]:obj[3]] = sonar_data[obj[0]:obj[1], obj[2]:obj[3]]
        return empty_data

    def combine_objects(self, obj1, obj2, sonar_data_new, sonar_data_old):
        ymin = min(obj1[0], obj2[0])
        ymax = max(obj1[1], obj2[1])
        xmin = min(obj1[2], obj2[2])
        xmax = max(obj1[3], obj2[3])
        if sonar_data_old.ndim == 2:
            sonar_data_old = np.expand_dims(sonar_data_old, axis=2)
        if sonar_data_new.ndim == 2:
            sonar_data_new = np.expand_dims(sonar_data_new, axis=2)
        
        combined_sonar_data = np.concatenate((sonar_data_old, sonar_data_new), axis=2)
        return [ymin, ymax, xmin, xmax], combined_sonar_data
        
    def extend_data(self, sonar_data, obj):
        y_center=(obj[0]+obj[1])/2
        x_center=(obj[2]+obj[3])/2
        offset_y=y_center-obj[0]
        offset_x=x_center-obj[2]
        y_max=obj[1]
        y_min=obj[0]
        x_max=obj[3]
        x_min=obj[2]
        if self.bias_x==1:
            x_max=int(obj[3]+offset_x)
            x_min=obj[2]
        else:
            x_min=int(obj[2]-offset_x)
            x_max=obj[3]
            
        if self.bias_y==1:
            y_max=int(obj[1]+offset_y)
            y_min=obj[0]
        else:
            y_max=obj[1]
            y_min=int(obj[0]-offset_y)
        empty_data = np.zeros_like(sonar_data)
        empty_data[obj[0]:obj[1], obj[2]:obj[3]] = sonar_data[obj[0]:obj[1], obj[2]:obj[3]]
        extend_obj=[y_min,y_max,x_min,x_max]
        return empty_data,extend_obj
        
def calculate_iou_on_small(obj1, obj2):
    ymin = max(obj1[0], obj2[0])
    ymax = min(obj1[1], obj2[1])
    xmin = max(obj1[2], obj2[2])
    xmax = min(obj1[3], obj2[3])
    if ymin >= ymax or xmin >= xmax:
        return 0
    inter = (ymax - ymin) * (xmax - xmin)
    obj1_area = (obj1[1] - obj1[0]) * (obj1[3] - obj1[2])
    obj2_area = (obj2[1] - obj2[0]) * (obj2[3] - obj2[2])
    iou = inter / min(obj1_area, obj2_area)
    return iou

def read_default_label(file_path):
    '''
    Args:
        file_path: path of label file
    Returns:
        human_ids: list of human id
        states: list of state
        objs: list of object, each object is a list of [ymin, ymax, xmin, xmax]
    '''
    human_ids = []
    states = []
    objs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split()
            # human object
            if len(arr) == 6:
                human_id = arr[0]
                state = arr[1]
                xmin = int(float(arr[2]))
                ymin = int(float(arr[3]))
                xmax = int(float(arr[4]))
                ymax = int(float(arr[5]))
            
            # noise object
            elif len(arr) == 5:
                human_id = -2
                state = "noise"
                xmin = int(float(arr[1]))
                ymin = int(float(arr[2]))
                xmax = int(float(arr[3]))
                ymax = int(float(arr[4]))

            else:
                raise ValueError('label file format error: {}'.format(label_file_path))

            obj = [ymin, ymax, xmin, xmax]
            human_ids.append(human_id)
            states.append(state)
            objs.append(obj)
    return human_ids, states, objs

def read_yolo_label(file_path, W=500, H=400):
    ''' Read one txt file in yolo format.
        human_id | state_str | x_center | y_center | width | height
    Args:
        file_path: path of label file
        W: width of sonar data, default 500 unit (20 m)
        H: height of sonar data, default 400 gradian
    Returns:
        obj_list: list of object, each object is a list of [ymin, ymax, xmin, xmax]
        states: list of state, each state is a string
    '''
    obj_list = []
    states = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split()
            human_id = arr[0]
            state = arr[1]
            x_center = float(arr[2])
            y_center = float(arr[3])
            width = float(arr[4])
            height = float(arr[5])

            xmin = int((x_center - width / 2) * W)
            xmax = int((x_center + width / 2) * W)
            ymin = int((y_center - height / 2) * H)
            ymax = int((y_center + height / 2) * H)

            obj = [ymin, ymax, xmin, xmax]
            obj_list.append(obj)
            states.append(state)
    return obj_list, states

def split_dataset_prepare_data(aug_p,raw_data_path,label_path,save_path,train_scenario,test_scenario,val_scenario):
    data_save_path = os.path.join(save_path, 'data')
    #print(data_save_path)
    label_save_path = os.path.join(save_path, 'label')
    data_save_train_path=os.path.join(data_save_path, 'train_new')
    data_save_test_path=os.path.join(data_save_path, 'test_new')
    data_save_val_path=os.path.join(data_save_path, 'val_new')
    label_save_train_path=os.path.join(label_save_path, 'train_new')
    label_save_test_path=os.path.join(label_save_path, 'test_new')
    label_save_val_path=os.path.join(label_save_path, 'val_new')
    save_file_idx = 0
    sonar_data = SonarData(channels=3)
    for scenario in tqdm(train_scenario):
        scenario = str(scenario)
        data_scenario_path = os.path.join(raw_data_path, scenario)
        label_scenario_path = os.path.join(label_path, scenario)
        if not os.path.exists(data_scenario_path):
            print("raw data path {} not exists".format(data_scenario_path))
            continue
        sonars = os.listdir(label_scenario_path)
        for sonar in sonars:
            data_dir_path = os.path.join(data_scenario_path, sonar)
            label_dir_path = os.path.join(label_scenario_path, sonar)
            assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)
            print(sonar, scenario)
            files = os.listdir(label_dir_path)
            if len(files) == 0:
                continue
            if aug_p=="random":
                random.shuffle(files)
            else:
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=True)
            sonar_data.__clean__()
            for file in files:
                data_file_path = os.path.join(data_dir_path, file)
                label_file_path = os.path.join(label_dir_path, file)
                assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                if aug_p=="none" or aug_p=="random":
                    save_file_idx = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar)
                elif aug_p=="concat":
                    save_file_idx = sonar_data.__process_one_concat_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar)
                elif aug_p=="bias":
                    save_file_idx = sonar_data.__process_one_bias_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar)
                else:
                    print("type error")
        
    mem=psutil.virtual_memory()
    print(mem)
                
    for scenario in tqdm(test_scenario):
        scenario = str(scenario)
        data_scenario_path = os.path.join(raw_data_path, scenario)
        label_scenario_path = os.path.join(label_path, scenario)
        if not os.path.exists(data_scenario_path):
            print("raw data path {} not exists".format(data_scenario_path))
            continue
        sonars = os.listdir(label_scenario_path)
        for sonar in sonars:
            data_dir_path = os.path.join(data_scenario_path, sonar)
            label_dir_path = os.path.join(label_scenario_path, sonar)
            assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)
            print(sonar, scenario)
            files = os.listdir(label_dir_path)
            if len(files) == 0:
                continue
            if aug_p=="random":
                random.shuffle(files)
            else:
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=True)
            sonar_data.__clean__()
            for file in files:
                data_file_path = os.path.join(data_dir_path, file)
                label_file_path = os.path.join(label_dir_path, file)
                assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                if aug_p=="none" or aug_p=="random":
                    save_file_idx = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar)
                elif aug_p=="concat":
                    save_file_idx = sonar_data.__process_one_concat_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar)
                elif aug_p=="bias":
                    save_file_idx = sonar_data.__process_one_bias_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar)
                else:
                    print("type error")
    mem=psutil.virtual_memory()
    print(mem)
    if aug_p=="none":
        for scenario in tqdm(val_scenario):
            scenario = str(scenario)
            data_scenario_path = os.path.join(raw_data_path, scenario)
            label_scenario_path = os.path.join(label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            sonars = os.listdir(label_scenario_path)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)
                print(sonar, scenario)
                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=True)
                sonar_data.__clean__()
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_val_path, label_save_val_path, sonar)
    mem=psutil.virtual_memory()
    print(mem)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="make label")
    parser.add_argument("--raw_data_path",
                        required=True,
                        type=str, help="raw data path")
    parser.add_argument("--save_path",
                        required=True,
                        type=str, help="process data and label save path")
    parser.add_argument("--label_path",
                        required=True,
                        type=str, help="label root path")
    parser.add_argument("--channel", type=int, default=1,
                        help="channel of sonar data")
    parser.add_argument("--reverse", action="store_true",
                        help="sort files in reverse order (used in hengfa)")
    parser.add_argument("--aug",help="data augmentation", type=str)
    args = parser.parse_args()
    #print(args.save_path)
    data_save_path = os.path.join(args.save_path, 'data')
    #print(data_save_path)
    label_save_path = os.path.join(args.save_path, 'label')
    data_save_train_path=os.path.join(data_save_path, 'train')
    data_save_test_path=os.path.join(data_save_path, 'test.txt')
    label_save_train_path=os.path.join(label_save_path, 'train')
    label_save_test_path=os.path.join(label_save_path, 'test.txt')
    #if not os.path.exists(args.save_path):
    #    os.mkdir(args.save_path)
    #if not os.path.exists(data_save_path):
    #    os.mkdir(data_save_path)
    #if not os.path.exists(label_save_path):
    #    os.mkdir(label_save_path)

    print("Start to prepare data {}".format(args.raw_data_path+" "+args.aug))
    # If the label exists, the data exists
    scenarios = os.listdir(args.label_path)
    index=np.zeros(len(scenarios))
    for i in range(len(index)):
        index[i]=i
    x_index_train,x_index_test,y_scenarios_train,y_scenarios_test=train_test_split(index,scenarios,test_size=0.33,random_state=0.42)
    save_file_idx = 0
    if args.channel == 1:
        sonar_data = SonarData(channels=1)
        for scenario in tqdm(scenarios):
            scenario = str(scenario)
            data_scenario_path = os.path.join(args.raw_data_path, scenario)
            label_scenario_path = os.path.join(args.label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            
            sonars = os.listdir(label_scenario_path)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=args.reverse)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=args.reverse)
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_pic_1ch__(data_file_path, label_file_path, data_save_path, label_save_path, sonar)
    
    elif args.aug=="random":
        sonar_data = SonarData(channels=args.channel)
        for scenario in tqdm(y_scenarios_train):
            scenario = str(scenario)
            data_scenario_path = os.path.join(args.raw_data_path, scenario)
            label_scenario_path = os.path.join(args.label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            sonars = os.listdir(label_scenario_path)
            #print(data_scenario_path,label_scenario_path)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if args.aug=="random":
                    random.shuffle(files)
                sonar_data.__clean__()
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar)
        for scenario in tqdm(y_scenarios_test):
            scenario = str(scenario)
            data_scenario_path = os.path.join(args.raw_data_path, scenario)
            label_scenario_path = os.path.join(args.label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            sonars = os.listdir(label_scenario_path)
            #print(data_scenario_path,label_scenario_path)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if args.aug=="random":
                    random.shuffle(files)
                sonar_data.__clean__()
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar)
            
    elif args.aug=="concat":
        sonar_data = SonarData(channels=args.channel)
        for scenario in tqdm(y_scenarios_train):
            scenario = str(scenario)
            data_scenario_path = os.path.join(args.raw_data_path, scenario)
            label_scenario_path = os.path.join(args.label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            sonars = os.listdir(label_scenario_path)
            #print(sonars)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=args.reverse)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=args.reverse)
                sonar_data.__clean__()
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_concat_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar)
        
        for scenario in tqdm(y_scenarios_test):
            scenario = str(scenario)
            data_scenario_path = os.path.join(args.raw_data_path, scenario)
            label_scenario_path = os.path.join(args.label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            sonars = os.listdir(label_scenario_path)
            #print(sonars)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=args.reverse)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=args.reverse)
                sonar_data.__clean__()
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_concat_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar)
                    
                    
                    
    elif args.aug=="bias":
        sonar_data = SonarData(channels=args.channel)
        for scenario in tqdm(y_scenarios_train):
            scenario = str(scenario)
            data_scenario_path = os.path.join(args.raw_data_path, scenario)
            label_scenario_path = os.path.join(args.label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            sonars = os.listdir(label_scenario_path)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=args.reverse)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=args.reverse)
                sonar_data.__clean__()
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_bias_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar)
                    
        for scenario in tqdm(y_scenarios_test):
            scenario = str(scenario)
            data_scenario_path = os.path.join(args.raw_data_path, scenario)
            label_scenario_path = os.path.join(args.label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            sonars = os.listdir(label_scenario_path)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=args.reverse)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=args.reverse)
                sonar_data.__clean__()
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_bias_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar)
                    
    else:
        sonar_data = SonarData(channels=args.channel)
        for scenario in tqdm(y_scenarios_train):
            scenario = str(scenario)
            data_scenario_path = os.path.join(args.raw_data_path, scenario)
            label_scenario_path = os.path.join(args.label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            sonars = os.listdir(label_scenario_path)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=args.reverse)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=args.reverse)
                sonar_data.__clean__()
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar)
                    
        for scenario in tqdm(y_scenarios_test):
            scenario = str(scenario)
            data_scenario_path = os.path.join(args.raw_data_path, scenario)
            label_scenario_path = os.path.join(args.label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            sonars = os.listdir(label_scenario_path)
            for sonar in sonars:
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]), reverse=args.reverse)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=args.reverse)
                sonar_data.__clean__()
                for file in files:
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    save_file_idx = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar)
                    
    print("Data preparation finished, saved in {}".format(args.save_path))
