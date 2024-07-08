# -*- coding: utf-8 -*-
"""Pose_Track_Dataloder_New.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IIQGd_BKWwbiYEQyZ1FuVkz0K2qV8Rhp
"""

import torch
import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import bisect
class PoseTrackDataset(Dataset):
  def __init__(self, main_folder, json_folder, max_frames, interp_shape):
      self.main_folder = main_folder
      self.json_folder = json_folder
      self.subdirectories = sorted(next(os.walk(main_folder))[1])
      self.valid_subdirectories = [
          subdir for subdir in self.subdirectories
          if os.path.exists(os.path.join(self.json_folder, f"{subdir}.json"))
      ]
      self.max_frames = max_frames
      self.interp_shape = interp_shape
      # with open('/content/drive/MyDrive/PoseTrack2/d1/dict.json', 'r') as json_file:
      #   load_dict = json.load(json_file)
      # self.loaded_dict = {int(k): v for k, v in load_dict.items()}

  def __len__(self):
      # last_key = list(self.loaded_dict.keys())[-1]
      # another_last_key = list(self.loaded_dict[last_key])[-1]
      # return len(self.loaded_dict[last_key][another_last_key]) + last_key
      return len(self.valid_subdirectories)

  def make_palindrome(self, tensor, required_length):
      current_length = tensor.shape[0]
      if current_length < required_length:
          additional_frames_needed = required_length - current_length
          # Reverse the tensor along the first dimension
          mirrored_part = torch.flip(tensor, [0])
          # Repeat the mirrored part if more frames are needed
          while mirrored_part.shape[0] < additional_frames_needed:
              mirrored_part = torch.cat((mirrored_part, torch.flip(tensor, [0])), dim=0)
          mirrored_part = mirrored_part[:additional_frames_needed]
          tensor = torch.cat((tensor, mirrored_part), dim=0)
      return tensor

  def load_video(self, subdir_path, frame_tuples):
    # print(frame_tuples)
    images = sorted([img for img in os.listdir(subdir_path) if img.endswith(".jpg")])
    image_arrays = []
    for img in images:
        img_path = os.path.join(subdir_path, img)
        img_array = cv2.imread(img_path)
        image_arrays.append(img_array)

    image_arrays_np = np.array(image_arrays)
    video = torch.from_numpy(image_arrays_np).permute(0, 3, 1, 2).float()[:, [2, 1, 0], :, :]
    T, C, H, W = video.shape
    # print(f"Total frames = {T}")
    # videos_lst = []
    video = F.interpolate(video, size=self.interp_shape, mode="bilinear", align_corners=True)
    video = video.reshape(T, 3, self.interp_shape[0], self.interp_shape[1])
    for start_frame in frame_tuples:
      end_frame = start_frame + self.max_frames - 1
      subclip = video[start_frame:end_frame+1]
      # print(subclip.shape)
      if subclip.shape[0] <= self.max_frames:
        subclip = self.make_palindrome(subclip, self.max_frames)
      else:
        print("Some error")
        return None
    return subclip,W,H
    #   videos_lst.append(subclip)
    # return torch.stack(videos_lst, dim=0),W,H

  def load_anno(self, json_path, img_path):
    def create_keypoints_tensor(annotation):
      keypoints = annotation['keypoints']
      processed_keypoints = []
      visibility = []
      frame_no = annotation['image_id'] % 1000
      for i in range(0, len(keypoints), 3):
          x = keypoints[i]
          y = keypoints[i + 1]
          vis = keypoints[i + 2]
          processed_keypoints.append([x, y])
          visibility.append(vis)
      return torch.tensor(processed_keypoints).unsqueeze(0), torch.tensor(visibility).unsqueeze(0)

    def best_starting_frame(person_frames):
      subclip_frames = []
      count_values = []
      init_queries_lst = []
      max_frame = person_frames[-1]
      for i in range(max_frame):
        start_frame = i
        end_frame = start_frame + self.max_frames -1
        count = 0
        for frame in person_frames:
          if start_frame <= frame <= end_frame:
            if count == 0:
              init_query_frame = frame
            count += 1
        if count >= self.max_frames/2:
          subclip_frames.append(start_frame)
          count_values.append(count)
          init_queries_lst.append(init_query_frame)

      return subclip_frames,count_values,init_queries_lst

    def extract_frame_number(file_name):
      base_name = os.path.basename(file_name)  # Get the base name of the file (e.g., '000142.jpg')
      frame_number = os.path.splitext(base_name)[0]  # Remove the extension (e.g., '000142')
      return int(frame_number)

    with open(json_path, 'r') as file:
      data = json.load(file)
      persons = {}
      frames = {}
      visibility = {}
      for i in data['annotations']:
          frame_num = i['image_id'] % 1000
          if i['person_id'] in persons:
              new_annot, vis = create_keypoints_tensor(i)
              persons[i['person_id']] = torch.cat((persons[i['person_id']], new_annot), dim=0)
              frames[i['person_id']].append(frame_num)
              visibility[i['person_id']] = torch.cat((visibility[i['person_id']], vis), dim=0)
          else:
              persons[i['person_id']], visibility[i['person_id']] = create_keypoints_tensor(i)
              frames[i['person_id']] = [frame_num]
    # print(frames)
    queries_lst = None
    trajs_e_lst = None
    vis_lst = None
    total_starts = []
    persons_and_frames = []

    # initial_frame = list_values[0]
    # person = list_values[1]
    # init_frame = list_values[2]
    # init_frame_idx = list_values[3]

    # Extracting T (Max Video Length)
    files = os.listdir(img_path)
    jpg_files = [f for f in files if f.endswith('.jpg')]
    frame_numbers = [extract_frame_number(os.path.join(img_path, f)) for f in jpg_files]
    T = max(frame_numbers)
    # frame_lst = frames[person]
    # frame_to_index = {frame: k for k, frame in enumerate(frame_lst)}
    # subclip_frames,count_values,init_queries_lst = best_starting_frame(frame_lst)
    # if person is None:
    #   default_queries = torch.zeros((17, 3))
    #   default_trajectories = torch.zeros((self.max_frames, 17, 2))
    #   default_visibility = torch.zeros((self.max_frames, 17))
    #   total_starts = [0]
    #   return default_queries, default_trajectories, default_visibility, total_starts,default_visibility
    # else:
    #   end_frame = initial_frame + self.max_frames - 1
    #   num_times = T -initial_frame +1
    #   if num_times > self.max_frames:
    #     num_times = self.max_frames
    #   trajs_e = torch.zeros((num_times, 17, 2))
    #   visib = torch.zeros((num_times, 17))
    #   for k in range(num_times):
    #     frame_number = initial_frame + k
    #     if frame_number in frame_to_index:
    #       trajs_e[k] = persons[person][frame_to_index[frame_number]]
    #       visib[k] = visibility[person][frame_to_index[frame_number]]
    #   if trajs_e.shape[0] != self.max_frames:
    #     req_frames = self.max_frames - trajs_e.shape[0]
    #     trajs_e = self.make_palindrome(trajs_e, self.max_frames)
    #     visib = self.make_palindrome(visib, self.max_frames)

    #   input_frame = persons[person][init_frame_idx]
    #   frame_tensor = torch.full((17, 1), init_frame - initial_frame)
    #   queries = torch.cat((frame_tensor, input_frame), dim=1)
    #   visib_frame = visib[init_frame - initial_frame]
    #   valids = visib_frame.unsqueeze(0).repeat(30, 1)
    #   return queries, trajs_e, visib, [initial_frame],valids
    # print(f"Hey frames{frames}")
    for i in frames:
      frame_lst = frames[i]
      subclip_frames,count_values,init_queries_lst = best_starting_frame(frame_lst)
      frame_to_index = {frame: k for k, frame in enumerate(frame_lst)}

      if len(subclip_frames) > 0:
        total_starts += subclip_frames
        person = i
        for num_subclips in range(len(subclip_frames)):
          initial_frame = subclip_frames[num_subclips]
          end_frame = initial_frame + self.max_frames - 1
          num_times = T -initial_frame +1
          if num_times > self.max_frames:
            num_times = self.max_frames
          trajs_e = torch.zeros((num_times, 17, 2))
          visib = torch.zeros((num_times, 17))
          for k in range(num_times):
            frame_number = initial_frame + k
            if frame_number in frame_to_index:
              trajs_e[k] = persons[person][frame_to_index[frame_number]]
              visib[k] = visibility[i][frame_to_index[frame_number]]
          if trajs_e.shape[0] != self.max_frames:
            req_frames = self.max_frames - trajs_e.shape[0]
            trajs_e = self.make_palindrome(trajs_e, self.max_frames)
            visib = self.make_palindrome(visib, self.max_frames)

          init_frame = init_queries_lst[num_subclips]

          init_frame_idx = frame_to_index[init_frame]
          input_frame = persons[person][init_frame_idx]
          frame_tensor = torch.full((17, 1), init_frame - initial_frame)
          queries = torch.cat((frame_tensor, input_frame), dim=1).unsqueeze(0)
          persons_and_frames.append((initial_frame,person,init_frame,init_frame_idx))
          trajs_e = trajs_e.unsqueeze(0)
          visib = visib.unsqueeze(0)
          if queries_lst is None:
            queries_lst = queries
            trajs_e_lst = trajs_e
            vis_lst = visib
          else:
            queries_lst = torch.cat((queries_lst, queries), dim=0)
            trajs_e_lst = torch.cat((trajs_e_lst, trajs_e), dim=0)
            vis_lst = torch.cat((vis_lst, visib), dim=0)

    if len(total_starts) == 0:
      default_queries = torch.zeros((1,17, 3))
      default_trajectories = torch.zeros((1,self.max_frames, 17, 2))
      default_visibility = torch.zeros((1,self.max_frames, 17))
      total_starts = [0]
      return default_queries, default_trajectories, default_visibility, total_starts, [(0,None,0,0)]


    return queries_lst, trajs_e_lst, vis_lst, total_starts, persons_and_frames

  def __getitem__(self, idx):

    # def find_greatest_leq(sorted_list, query):
      # Find the index where 'query' should be inserted to maintain sorted order
      # index = bisect.bisect_right(sorted_list, query)
      # # The greatest value less than or equal to 'query' will be the element at index-1
      # if index > 0:
      #     return sorted_list[index - 1]
      # else:
      #     return None

    subdir = self.valid_subdirectories[idx]
    # if idx >= len(self):
    #   print("Not those many values present")
    #   return None
    # list_values = list(self.loaded_dict.keys())
    # leq_idx = find_greatest_leq(list_values, idx)
    # subdir = list(self.loaded_dict[leq_idx].keys())[0]
    # start_frame_idx = idx - leq_idx
    # if start_frame_idx < 0:
    #   print("Some error in start frame idx")
    #   return None
    # list_values = self.loaded_dict[leq_idx][subdir][start_frame_idx]
    img_path = os.path.join(self.main_folder, subdir)
    anno_path = os.path.join(self.json_folder, f"{subdir}.json")
    # queries, trajs_e, vis,total_starts,valids = self.load_anno(anno_path, img_path,list_values)
    queries_lst, trajs_e_lst, vis_lst, total_starts,persons_and_frames = self.load_anno(anno_path, img_path)
    # queries_lst, trajs_e_lst, vis_lst, total_starts, persons_and_frames = self.load_anno(anno_path, img_path)
    # video,W,H = self.load_video(img_path, total_starts)
    # queries = queries.clone()
    # # print(video.shape)
    # # print(queries.shape)
    # queries[:,1:] *= queries.new_tensor(
    #     [
    #         (self.interp_shape[1] - 1) / (W - 1),
    #         (self.interp_shape[0] - 1) / (H - 1),
    #     ]
    # )
    # # Adjust tracks
    # trajs_e = trajs_e.clone()
    # trajs_e *= trajs_e.new_tensor(
    #     [
    #         (self.interp_shape[1] - 1) / (W - 1),
    #         (self.interp_shape[0] - 1) / (H - 1),
    #     ]
    # )

    # return video, queries, trajs_e, vis,valids
    # return queries, trajs_e, vis,valids
    return subdir,total_starts,persons_and_frames

val_folder = '/content/drive/MyDrive/PoseTrack2/d1/images/val'
val_json_folder = '/content/drive/MyDrive/PoseTrack2/d1/PoseTrack21/posetrack_data/val'
val_dataset = PoseTrackDataset(val_folder, val_json_folder, 30, (384,512))

total_len = 0
for i in range(len(val_dataset)):
  total_len += len(val_dataset[i])

print(f"Hey total length of video is {total_len}")

def reduce_length_stride_3(lst):
  # print(f"Before: {lst}")
  new_lst = []
  for i in range(0,len(lst),3):
    new_lst.append(lst[i])
  # print(f"After: {new_lst}")
  return new_lst

total_dict = {}
idx = 0
for i in range(len(val_dataset)):
  temp_dict = {}
  subdir,total_starts,persons_and_frames = val_dataset[i]
  persons_and_frames = reduce_length_stride_3(persons_and_frames)
  temp_dict[subdir] = persons_and_frames
  total_dict[idx] = temp_dict
  idx+=len(persons_and_frames)

with open('/content/drive/MyDrive/PoseTrack2/d1/val.json', 'w') as json_file:
  json.dump(total_dict, json_file)

train_folder = '/content/drive/MyDrive/PoseTrack2/d1/images/train'
train_json_folder = '/content/drive/MyDrive/PoseTrack2/d1/PoseTrack21/posetrack_data/train'
train_dataset = PoseTrackDataset(train_folder, train_json_folder, 30, (384,512))



len(train_dataset)



def reduce_length_stride_3(lst):
  # print(f"Before: {lst}")
  new_lst = []
  for i in range(0,len(lst),3):
    new_lst.append(lst[i])
  # print(f"After: {new_lst}")
  return new_lst

total_dict = {}
idx = 0
for i in range(len(train_dataset)):
  temp_dict = {}
  subdir,total_starts,persons_and_frames = train_dataset[i]
  persons_and_frames = reduce_length_stride_3(persons_and_frames)
  temp_dict[subdir] = persons_and_frames
  total_dict[idx] = temp_dict
  idx+=len(persons_and_frames)

a = {1:3,2:4,3:5}

list(total_dict.keys())[-1]

import bisect

def find_greatest_leq(sorted_list, query):
    # Find the index where 'query' should be inserted to maintain sorted order
    index = bisect.bisect_right(sorted_list, query)
    # The greatest value less than or equal to 'query' will be the element at index-1
    if index > 0:
        return sorted_list[index - 1]
    else:
        return None  # or some other value to indicate no valid result

# Example usage
sorted_list = [0, 43, 56, 62]

queries = [45, 57, 0, 62, 100, -10]

for query in queries:
    result = find_greatest_leq(sorted_list, query)
    print(f"The greatest value <= {query} is {result}")

with open('/content/drive/MyDrive/PoseTrack2/d1/dict.json', 'w') as json_file:
    json.dump(total_dict, json_file)

loaded_dict_with_int_keys = {int(k): v for k, v in loaded_dict.items()}

# Now, you can access the dictionary with integer keys
print(list(loaded_dict_with_int_keys.keys())[0])







# with open('/content/drive/MyDrive/PoseTrack2/d1/dict.json', 'w') as json_file:
#     json.dump(total_dict, json_file)

# !git clone https://github.com/facebookresearch/co-tracker
# %cd co-tracker
# !pip install -e .
# !pip install opencv-python einops timm matplotlib moviepy flow_vis
# !mkdir checkpoints
# %cd checkpoints
# !wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth

# %cd /content/co-tracker
# import os
# import torch

# from base64 import b64encode
# from cotracker.utils.visualizer import Visualizer, read_video_from_path
# from IPython.display import HTML

# def make_video(pred_tracks, pred_visibility, video, save_path, num):
#   vis = Visualizer(
#       save_dir=save_path,
#       linewidth=1,
#       mode='cool',
#       # tracks_leave_trace=-1
#   )
#   vis.visualize(
#       video=video,
#       tracks=pred_tracks,
#       visibility=pred_visibility,
#       filename=f'queries_{num}')

# for i in range(videos_lst.shape[0]):
#   video = videos_lst[i]
#   start_idx  = int(queries_lst[i][0][0])
#   video_start_idx = int(total_starts[i])
#   pred_tracks = trajs_e_lst[i]
#   pred_visibility = vis_lst[i]
#   make_video(pred_tracks[None], pred_visibility[None], video[None], '/content/drive/MyDrive/videos/videos1', f"_{video_start_idx}_{start_idx+video_start_idx}_{i}")





