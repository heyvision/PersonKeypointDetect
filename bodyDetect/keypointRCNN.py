import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import torch
import torchvision
import numpy as np
import cv2
import pyrealsense2 as rs
import time

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COCO_PERSON_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

def predict(src, model):
  img_cv = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
  # print(img_cv.shape)
  # tensor_cv = np.transpose(img_cv, (2, 0, 1))
  tensor_cv = torchvision.transforms.ToTensor()(img_cv)
  tensor_cv = tensor_cv.cuda()
  
  # print (tensor_cv.size())
  x = [tensor_cv]
  predictions = model(x)

  boxes = predictions[0]['boxes'].cpu().detach().numpy()
  labels = predictions[0]['labels'].cpu().detach().numpy()
  keypoints = predictions[0]['keypoints'].cpu().detach().numpy()
  keypoints_scores = predictions[0]['keypoints_scores'].cpu().detach().numpy()

  # print(predictions[0])
  boxes_num = boxes.shape[0]
  person_id = -1
  for i in range(boxes_num):
    if(labels[i] == 1): #first person
      person_id = i
      break
  if person_id == -1:
    # print("No person in image!")
    return src

  # print("keypoints of person's shape is:", keypoints[person_id,:,:].shape)
  # index_sort = np.argsort(-keypoints_scores[person_id])
  # for i in range (10):
  #   print (COCO_PERSON_KEYPOINT_NAMES[index_sort[i]])
  cv2.rectangle(src, (boxes[person_id][0], boxes[person_id][1]), 
      (boxes[person_id][2], boxes[person_id][3]), (255,0,0), 2)
  i_keypoints = keypoints[person_id,:,:]
  for j in range(i_keypoints.shape[0]):
    if keypoints_scores[person_id][j] > 0:
      cv2.circle(src, (i_keypoints[j][0], i_keypoints[j][1]), 5, (0,0,255), -1)
      text = COCO_PERSON_KEYPOINT_NAMES[j] # + str(round(keypoints_scores[person_id][j], 2))
      if not i_keypoints[j][2]:
        text += '(hide)'
      cv2.putText(src, text, (i_keypoints[j][0], i_keypoints[j][1]), 
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
  # cv2.line()
  return src


def main():

  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

  # Start streaming
  profile = pipeline.start(config)
  depth_sensor = profile.get_device().first_depth_sensor()
  # depth_sensor.set_option(rs.option.visual_preset, rs.Preset.HighAccuracy)
  depth_scale = depth_sensor.get_depth_scale()
  clipping_distance_in_meters = 3 # 3 meter
  clipping_distance = clipping_distance_in_meters / depth_scale

  align_to = rs.stream.color
  align = rs.align(align_to)

  model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
  model = model.cuda()
  model.eval()

  frame_count = 0
  try:
    while True:
      time_start = time.time()

      frames = pipeline.wait_for_frames()
      aligned_frames = align.process(frames)
      aligned_depth_frame = aligned_frames.get_depth_frame()
      color_frame = aligned_frames.get_color_frame()
      if not aligned_depth_frame or not color_frame:
        continue
      color_image = np.asanyarray(color_frame.get_data())
      pred_img = predict(color_image, model)
      cv2.imshow("prediction", pred_img)

      frame_count += 1
      print('time cost:',round(time.time()-time_start, 4), 's')
      if cv2.waitKey(1) == 27:
        break
  finally:
    pipeline.stop()

def one_image():
  src = cv2.imread('/home/hit/Pictures/test2_Color.png')
  # src = cv2.resize(src, (320, 180))

  model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
  model = model.cuda()
  # print(next(model.parameters()).is_cuda)
  model.eval()

  pred_img = predict(src, model)
  cv2.imshow("prediction", pred_img)
  cv2.waitKey(0)

if __name__ == "__main__":
  #print(torch.cuda.current_device())
  if torch.cuda.is_available():
    print("Device",torch.cuda.get_device_name(0),"is available")
  else:
    print("Cuda is unavailable")

  main()
  #one_image()