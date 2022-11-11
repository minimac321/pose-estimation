# pose-estimation

What is pose estimation ?
- Pose estimation predicts different poses based on a person’s body parts and joint positioning in an image or video. We can automatically detect the joints, arms, hips, and spine position while performing a squat.

Use cases:
- Athlete rehabilitating after an injury or undergoing strength training; the pose estimation may help sports analysts analyze vital points from the starting position to the end position of a squat


We use [MediaPipe](https://github.com/google/mediapipe) Holistic model to generate 33 key points features for human pose estimation. See image below

 ![ alt text for screen readers](inputs/pose_estimation_connections_guide.png "Pose estimation guide")
 

Similarity Metrics:
- Given two 3d pose's - calculate whether the poses are the same. See: https://medium.com/@cavaldovinos/human-pose-estimation-pose-similarity-dc8bf9f78556

Results:

## TODO
Show test image and outputted result