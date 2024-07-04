# -*- coding: utf-8 -*-
"""Training_Co-Tracker2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1POa4ESflvISsdX_B2DsRS8evG-MKV0pI
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/facebookresearch/co-tracker
# %cd co-tracker
!pip install -e .
!pip install opencv-python einops timm matplotlib moviepy flow_vis
!mkdir checkpoints
# %cd checkpoints
!wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/co-tracker
import os
import torch

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML

from cotracker.predictor import CoTrackerPredictor

model = CoTrackerPredictor(
    checkpoint=os.path.join(
        './checkpoints/cotracker2.pth'
    )
)

def make_video(pred_tracks, pred_visibility, video, vis_writer, num, idx):
  video = video[0][idx:]
  pred_tracks = pred_tracks[0][idx:]
  pred_visibility = pred_visibility[0][idx:]
  vis = Visualizer(
      linewidth=1,
      mode='cool',
      # tracks_leave_trace=-1
  )
  vis.visualize(
      writer = vis_writer,
      video=video[None],
      tracks=pred_tracks[None],
      visibility=pred_visibility[None],
      step = num,
      filename=f'video_{num}')

def show_video(video_path):
  video_file = open(video_path, "r+b").read()
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video width="640" height="480" autoplay loop controls><source src="{video_url}"></video>""")

def loss_fn(trajs_e, trajs_g, mask):
  # Compute the errors
  errors = torch.norm(trajs_e - trajs_g, dim=-1)

  # Get the visibility flags

  # Filter the errors for visible points
  visible_errors = errors[mask > 0]

  # Compute the mean error for visible points
  error = torch.mean(visible_errors)

  return error

import torch
import numpy as np

EPS = 1e-6

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    """
    Compute the mean of x considering only the valid points defined by mask.

    Args:
    x (torch.Tensor): Data tensor.
    mask (torch.Tensor): Mask tensor, should be the same shape as x.
    dim (int or tuple of int, optional): Dimension(s) to reduce.
    keepdim (bool, optional): Whether to keep the dimensions of reduction.

    Returns:
    torch.Tensor: The mean value considering only valid points.
    """
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom
    return mean

def evaluate_trajectories(trajs_e, trajs_g, valids, W, H):
    """
    Evaluate the predicted trajectories against ground truth trajectories.

    Args:
    trajs_e (torch.Tensor): Predicted trajectories of shape (B, S, 2).
    trajs_g (torch.Tensor): Ground truth trajectories of shape (B, S, 2).
    valids (torch.Tensor): Validity mask of shape (B, S) with 1 for valid and 0 for invalid.
    W (float): Width of the evaluation space.
    H (float): Height of the evaluation space.

    Returns:
    dict: Metrics containing distance thresholds and average distance.
    """
    # Distance thresholds
    thrs = [1, 2, 4, 8, 16]
    d_sum = 0.0
    metrics = {}

    # Scaling factors
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])  # Shape: (1, 1, 2)
    sc_pt = torch.from_numpy(sc_py).float().to(trajs_e.device)

    for thr in thrs:
        # Calculate the L2 norm (Euclidean distance) and apply threshold
        d_ = (torch.norm(trajs_e[:, 1:] / sc_pt - trajs_g[:, 1:] / sc_pt, dim=-1) < thr).float()  # Shape: (B, S-1)

        # Reduce masked mean considering only valid points
        d_ = reduce_masked_mean(d_, valids[:, 1:], dim=1).mean().item() * 100.0

        # Accumulate the distance metrics
        d_sum += d_

        # Store individual threshold metrics
        metrics['d_%d' % thr] = d_

    # Calculate average distance metric
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg

    return metrics

optimizer = torch.optim.AdamW(model.model.parameters(), lr=5e-5)

import torch
import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import bisect
class PoseTrackDataset(Dataset):
  def __init__(self, main_folder, json_folder,start_frames_json,max_frames,interp_shape):
      self.main_folder = main_folder
      self.json_folder = json_folder
      self.max_frames = max_frames
      self.interp_shape = interp_shape
      with open(start_frames_json, 'r') as json_file:
        load_dict = json.load(json_file)
      self.loaded_dict = {int(k): v for k, v in load_dict.items()}

  def __len__(self):
      last_key = list(self.loaded_dict.keys())[-1]
      another_last_key = list(self.loaded_dict[last_key])[-1]
      return len(self.loaded_dict[last_key][another_last_key]) + last_key
      # return len(self.valid_subdirectories)

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

  def load_anno(self, json_path, img_path, list_values):
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

    initial_frame = list_values[0]
    person = list_values[1]
    init_frame = list_values[2]
    init_frame_idx = list_values[3]

    # Extracting T (Max Video Length)
    files = os.listdir(img_path)
    jpg_files = [f for f in files if f.endswith('.jpg')]
    frame_numbers = [extract_frame_number(os.path.join(img_path, f)) for f in jpg_files]
    T = max(frame_numbers)
    frame_lst = frames[person]
    frame_to_index = {frame: k for k, frame in enumerate(frame_lst)}
    subclip_frames,count_values,init_queries_lst = best_starting_frame(frame_lst)
    if person is None:
      default_queries = torch.zeros((17, 3))
      default_trajectories = torch.zeros((self.max_frames, 17, 2))
      default_visibility = torch.zeros((self.max_frames, 17))
      total_starts = [0]
      return default_queries, default_trajectories, default_visibility, total_starts,default_visibility
    else:
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
          visib[k] = visibility[person][frame_to_index[frame_number]]
      if trajs_e.shape[0] != self.max_frames:
        req_frames = self.max_frames - trajs_e.shape[0]
        trajs_e = self.make_palindrome(trajs_e, self.max_frames)
        visib = self.make_palindrome(visib, self.max_frames)

      input_frame = persons[person][init_frame_idx]
      frame_tensor = torch.full((17, 1), init_frame - initial_frame)
      queries = torch.cat((frame_tensor, input_frame), dim=1)
      visib_frame = visib[init_frame - initial_frame]
      valids = visib_frame.unsqueeze(0).repeat(30, 1)
      return queries, trajs_e, visib, [initial_frame],valids

  def __getitem__(self, idx):

    def find_greatest_leq(sorted_list, query):
      # Find the index where 'query' should be inserted to maintain sorted order
      index = bisect.bisect_right(sorted_list, query)
      # The greatest value less than or equal to 'query' will be the element at index-1
      if index > 0:
          return sorted_list[index - 1]
      else:
          return None

    # subdir = self.valid_subdirectories[idx]
    if idx >= len(self):
      print("Not those many values present")
      return None
    list_values = list(self.loaded_dict.keys())
    leq_idx = find_greatest_leq(list_values, idx)
    subdir = list(self.loaded_dict[leq_idx].keys())[0]
    start_frame_idx = idx - leq_idx
    if start_frame_idx < 0:
      print("Some error in start frame idx")
      return None
    list_values = self.loaded_dict[leq_idx][subdir][start_frame_idx]
    img_path = os.path.join(self.main_folder, subdir)
    anno_path = os.path.join(self.json_folder, f"{subdir}.json")
    queries, trajs_e, vis,total_starts,valids = self.load_anno(anno_path, img_path,list_values)
    video,W,H = self.load_video(img_path, total_starts)
    queries = queries.clone()
    queries[:,1:] *= queries.new_tensor(
        [
            (self.interp_shape[1] - 1) / (W - 1),
            (self.interp_shape[0] - 1) / (H - 1),
        ]
    )
    # Adjust tracks
    trajs_e = trajs_e.clone()
    trajs_e *= trajs_e.new_tensor(
        [
            (self.interp_shape[1] - 1) / (W - 1),
            (self.interp_shape[0] - 1) / (H - 1),
        ]
    )

    return video, queries, trajs_e, vis,valids

train_folder = '/content/drive/MyDrive/PoseTrack2/d1/images/train'
train_json_folder = '/content/drive/MyDrive/PoseTrack2/d1/PoseTrack21/posetrack_data/train'
train_start_frames_folder = '/content/drive/MyDrive/PoseTrack2/d1/dict.json'
train_dataset = PoseTrackDataset(train_folder, train_json_folder,train_start_frames_folder, 30, (384,512))

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
import numpy as np

def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Set random seeds for reproducibility
set_random_seeds(42)

# Get the indices of the dataset
indices = list(range(len(train_dataset)))

# Shuffle the indices
random.shuffle(indices)

# Select the first 10 indices
sample_indices = indices[:10]

# Create a Subset of the dataset using the selected indices
subset_dataset = Subset(train_dataset, sample_indices)

# Create a DataLoader to iterate through the subset
train_dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

sample_indices

import torch
import numpy as np

EPS = 1e-6

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    """
    Compute the mean of x considering only the valid points defined by mask.

    Args:
    x (torch.Tensor): Data tensor.
    mask (torch.Tensor): Mask tensor, should be the same shape as x.
    dim (int or tuple of int, optional): Dimension(s) to reduce.
    keepdim (bool, optional): Whether to keep the dimensions of reduction.

    Returns:
    torch.Tensor: The mean value considering only valid points.
    """
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom
    return mean

def evaluate_trajectories(trajs_e, trajs_g, valids, W, H):
    """
    Evaluate the predicted trajectories against ground truth trajectories.

    Args:
    trajs_e (torch.Tensor): Predicted trajectories of shape (B, S, 2).
    trajs_g (torch.Tensor): Ground truth trajectories of shape (B, S, 2).
    valids (torch.Tensor): Validity mask of shape (B, S) with 1 for valid and 0 for invalid.
    W (float): Width of the evaluation space.
    H (float): Height of the evaluation space.

    Returns:
    dict: Metrics containing distance thresholds and average distance.
    """
    # Distance thresholds
    thrs = [1, 2, 4, 8, 16]
    d_sum = 0.0
    metrics = {}

    # Scaling factors
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])  # Shape: (1, 1, 2)
    sc_pt = torch.from_numpy(sc_py).float().to(trajs_e.device)

    for thr in thrs:
        # Calculate the L2 norm (Euclidean distance) and apply threshold
        d_ = (torch.norm(trajs_e[:, 1:] / sc_pt - trajs_g[:, 1:] / sc_pt, dim=-1) < thr).float()  # Shape: (B, S-1)

        # Reduce masked mean considering only valid points
        d_ = reduce_masked_mean(d_, valids[:, 1:], dim=1).mean().item() * 100.0

        # Accumulate the distance metrics
        d_sum += d_

        # Store individual threshold metrics
        metrics['d_%d' % thr] = d_

    # Calculate average distance metric
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg

    return metrics

def print_metrics(metrics):
  for key, value in metrics.items():
    print(f"{key}: {value}")

def save_metrics(metrics, save_dir, epoch):
  file_path = os.path.join(save_dir, 'metrics.txt')
  with open(file_path, 'a') as file:
    file.write(f'Epoch: {epoch}\n')
    # Iterate through the dictionary items
    for key, value in metrics.items():
        # Write each key-value pair on a new line
        file.write(f'{key}: {value}\n')

def make_video(pred_tracks, pred_visibility, video, vis_writer, epoch,batch_idx, idx):
  video = video[0][idx:]
  pred_tracks = pred_tracks[0][idx:]
  pred_visibility = pred_visibility[0][idx:]
  vis = Visualizer(
      linewidth=1,
      mode='cool',
      # save_dir=save_path
      # tracks_leave_trace=-1
  )
  vis.visualize(
      writer = vis_writer,
      video=video[None],
      tracks=pred_tracks[None],
      visibility=pred_visibility[None],
      step = epoch,
      filename=f'video_{batch_idx}')

import os
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model, optimizer, loss_fn, train_dataloader, load_model=False, new_lr=None, new_momentum=None, num_steps=5000):
    set_random_seeds()

    ckpt_dir = '/content/drive/MyDrive/PoseTrack_results/dataset_size_10/checkpoints'
    logs_dir = '/content/drive/MyDrive/PoseTrack_results/dataset_size_10/new_logs'
    new_ckpt_dir = '/content/drive/MyDrive/PoseTrack_results/dataset_size_10/new_checkpoints'


    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(new_ckpt_dir, exist_ok = True)

    writer = SummaryWriter(logs_dir)

    loss_threshold = 0.01
    start_step = 0
    # if load_model:
    #     checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    #     if checkpoints:
    #         latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    #         checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
    #         checkpoint = torch.load(checkpoint_path)
    #         model.model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         start_step = checkpoint['epoch']*10
    #         print(f"Loaded model from {checkpoint_path}, starting from epoch {start_step}")

    #         if new_lr is not None:
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] = new_lr
    #             print(f"Learning rate updated to {new_lr}")

    #         if new_momentum is not None:
    #             if 'betas' in optimizer.param_groups[0]:  # For Adam/AdamW optimizers
    #                 for param_group in optimizer.param_groups:
    #                     param_group['betas'] = (new_momentum, param_group['betas'][1])
    #                 print(f"Momentum (beta1) updated to {new_momentum}")
    #             elif 'momentum' in optimizer.param_groups[0]:  # For SGD optimizers
    #                 for param_group in optimizer.param_groups:
    #                     param_group['momentum'] = new_momentum
    #                 print(f"Momentum updated to {new_momentum}")

    if load_model:
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step'] + 1
            print(f"Loaded model from {checkpoint_path}, starting from step {start_step}")

            if new_lr is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Learning rate updated to {new_lr}")

            if new_momentum is not None:
                if 'betas' in optimizer.param_groups[0]:  # For Adam/AdamW optimizers
                    for param_group in optimizer.param_groups:
                        param_group['betas'] = (new_momentum, param_group['betas'][1])
                    print(f"Momentum (beta1) updated to {new_momentum}")
                elif 'momentum' in optimizer.param_groups[0]:  # For SGD optimizers
                    for param_group in optimizer.param_groups:
                        param_group['momentum'] = new_momentum
                    print(f"Momentum updated to {new_momentum}")

    if torch.cuda.is_available():
        model = model.cuda()

    step = start_step
    data_iter = iter(train_dataloader)
    losses = []
    d_avg_metrics = []
    while step < num_steps:
        model.model.train()


        try:
            video, queries, trajs_e, visibility, valids = next(data_iter)
        except StopIteration:
            avg_loss = sum(losses) / len(losses)
            losses = []
            avg_d_metrics = sum(d_avg_metrics) / len(d_avg_metrics)
            d_avg_metrics = []
            epoch = step/len(train_dataloader)
            writer.add_scalar('Training Loss', avg_loss, step/len(train_dataloader))
            writer.add_scalar('d_avg Average Metric', avg_d_metrics, step/len(train_dataloader))
            if avg_loss < loss_threshold:
                print(f"Loss has converged, Step {step}")
                break
            print(f"Epoch: {epoch}, Loss: {avg_loss}, d_avg Metric: {avg_d_metrics}")
            data_iter = iter(train_dataloader)
            video, queries, trajs_e, visibility, valids = next(data_iter)

        if torch.cuda.is_available():
            video = video.cuda()
            queries = queries.cuda()
            trajs_e = trajs_e.cuda()
            visibility = visibility.cuda()
            valids = valids.cuda()

        pred_tracks, pred_visibility = model(video, queries=queries)
        loss = loss_fn(trajs_e, pred_tracks, visibility * valids)
        losses.append(loss.item())
        metrics = evaluate_trajectories(trajs_e, pred_tracks, valids * visibility, 512, 384)
        step_d_avg = metrics['d_avg']
        writer.add_scalar('Step Loss', loss.item(), step)
        writer.add_scalar('d_avg Step Metric', step_d_avg, step)
        d_avg_metrics.append(step_d_avg)
        if torch.isnan(loss):
            print(f"Loss is NaN, Step {step}")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Optimized: Step {step}, Loss: {loss.item()}, d_avg: {step_d_avg}")

        if step % 10 == 0:
            file_path = os.path.join(new_ckpt_dir, f'model_step_{step}.pth')
            torch.save({
                'step': step,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, file_path)
            print("Saved model")

        if step % 100 == 0:
            idx = queries[0][0][0]
            make_video(pred_tracks, pred_visibility, video, writer, step, 0, int(idx))
            print("Yay! Saved Video")

        step += 1


    writer.close()

train(model, optimizer, loss_fn, train_dataloader, load_model=True)































import os
os._exit(00)

