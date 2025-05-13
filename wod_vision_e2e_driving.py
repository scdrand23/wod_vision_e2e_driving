from typing import Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math
import numpy as np
import cv2
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2

DATASET_FOLDER = 'data'

TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training_*.tfrecord*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'val_*.tfrecord*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'test_*.tfrecord*')

filenames = tf.io.matching_files(VALIDATION_FILES)
dataset = tf.data.TFRecordDataset(filenames, compression_type='')
dataset_iter = dataset.as_numpy_iterator()

bytes_example = next(dataset_iter)
data = wod_e2ed_pb2.E2EDFrame()
data.ParseFromString(bytes_example)

def return_front3_cameras(data: wod_e2ed_pb2.E2EDFrame):
  """Return the front_left, front, and front_right cameras as a list of images"""
  image_list = []
  calibration_list = []
  # CameraName Enum reference:
  # https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50
  order = [2, 1, 3]
  for camera_name in order:
    for index, image_content in enumerate(data.frame.images):
      if image_content.name == camera_name:
        # Decode the raw image string and convert to numpy type.
        calibration = data.frame.context.camera_calibrations[index]
        image = tf.io.decode_image(image_content.image).numpy()
        image_list.append(image)
        calibration_list.append(calibration)
        break

  return image_list, calibration_list


front3_camera_image_list, front3_camera_calibration_list = return_front3_cameras(data)
concatenated_image = np.concatenate(front3_camera_image_list, axis=1)
# Save the concatenated image to file instead of displaying it
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'front3_cameras.png')
cv2.imwrite(output_path, cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR))
print(f"Image saved to {output_path}")


def project_vehicle_to_image(vehicle_pose, calibration, points):
  """Projects from vehicle coordinate system to image with global shutter.

  Arguments:
    vehicle_pose: Vehicle pose transform from vehicle into world coordinate
      system.
    calibration: Camera calibration details (including intrinsics/extrinsics).
    points: Points to project of shape [N, 3] in vehicle coordinate system.

  Returns:
    Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
  """
  # Transform points from vehicle to world coordinate system (can be
  # vectorized).
  pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
  world_points = np.zeros_like(points)
  for i, point in enumerate(points):
    cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    world_points[i] = (cx, cy, cz)

  # Populate camera image metadata. Velocity and latency stats are filled with
  # zeroes.
  extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [4, 4])
  intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
  metadata = tf.constant([
      calibration.width,
      calibration.height,
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
  ],
                         dtype=tf.int32)
  camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

  # Perform projection and return projected image coordinates (u, v, ok).
  return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                            camera_image_metadata,
                                            world_points).numpy()


def draw_points_on_image(image, points, size):
  """Draws points on an image.

  Args:
    image: The image to draw on.
    points: A numpy array of shape (N, 2) representing the points to draw.
  """
  for point in points:
    cv2.circle(image, (int(point[0]), int(point[1])), size, (255, 0, 0), -1)
  return image


future_waypoints_matrix = np.stack([data.future_states.pos_x, data.future_states.pos_y, data.future_states.pos_z], axis=1)
vehicle_pose = data.frame.images[0].pose





images_with_drawn_points = []
for i in range(len(front3_camera_calibration_list)):
  waypoints_camera_space = project_vehicle_to_image(vehicle_pose, front3_camera_calibration_list[i], future_waypoints_matrix)
  images_with_drawn_points.append(draw_points_on_image(front3_camera_image_list[i], waypoints_camera_space, size=15))
concatenated_image = np.concatenate(images_with_drawn_points, axis=1)
plt.figure(figsize=(20, 20))
plt.imshow(concatenated_image)
plt.axis('off')
plt.savefig('outputs/waypoints_visualization.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Assume we have a predicted stopping trajectory.
predicted_trajectory = wod_e2ed_submission_pb2.TrajectoryPrediction(pos_x=np.zeros(20, dtype=np.float32),
                                                                    pos_y=np.zeros(20, dtype=np.float32))
frame_name = data.frame.context.name
frame_trajectory = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(frame_name=frame_name, trajectory=predicted_trajectory)
# The final prediction should be a list of FrameTrajectoryPredictions.
predictions = [frame_trajectory]

# Pack for submission.
num_submission_shards = 1  # Please modify accordingly.
submission_file_base = 'outputs/MySubmission'  # Please modify accordingly.
if not os.path.exists(submission_file_base):
  os.makedirs(submission_file_base)
sub_file_names = [
    os.path.join(submission_file_base, part)
    for part in [f'part{i}' for i in range(num_submission_shards)]
]
# As the submission file may be large, we shard them into different chunks.
submissions = []
num_predictions_per_shard =  math.ceil(len(predictions) / num_submission_shards)
for i in range(num_submission_shards):
  start = i * num_predictions_per_shard
  end = (i + 1) * num_predictions_per_shard
  submissions.append(
      wod_e2ed_submission_pb2.E2EDChallengeSubmission(
          predictions=predictions[start:end]))
  

for i, shard in enumerate(submissions):
  shard.submission_type  =  wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
  shard.authors[:] = ['Dereje Shenkut']  # Please modify accordingly.
  shard.affiliation = 'Carnegie Mellon University'  # Please modify accordingly.
  shard.account_name = 'dshenkut@andrew.cmu.edu'  # Please modify accordingly.
  shard.unique_method_name = 'E2E Driving'  # Please modify accordingly.
  shard.method_link = 'https://github.com/DerejeShenkut/wod_vision_e2e_driving'  # Please modify accordingly.
  shard.description = 'This is a simple method that predicts a stopping trajectory for a vehicle in a given frame.'  # Please modify accordingly.
  shard.uses_public_model_pretraining = True # Please modify accordingly.
  shard.public_model_names.extend(['Model_name']) # Please modify accordingly.
  shard.num_model_parameters = "200k" # Please modify accordingly.
  with tf.io.gfile.GFile(sub_file_names[i], 'wb') as fp:
    fp.write(shard.SerializeToString())


print(submissions[0])
