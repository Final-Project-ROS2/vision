
===========================================================
EXAMPLE CALIBRATION RESULTS (k1_only model)
============================================================
Number of images used: 29 from 40
Image size: (480, 640)
Mean reprojection error: 0.0357 pixels

Camera Matrix (K):
[[626.4740297    0.         317.90571208]
 [  0.         624.36106177 248.60021972]
 [  0.           0.           1.        ]]

Distortion Coefficients:
  k1=0.018713
  k2=0.000000  (fixed)
  p1=-0.000055
  p2=0.000589
  k3=0.000000  (fixed)

Focal Length:
  fx=626.47 pixels
  fy=624.36 pixels

Principal Point:
  cx=317.91 pixels
  cy=248.60 pixels
============================================================


# Note that RealSense is factory calibrated

Depth + RGB already have known extrinsics

SDK handles distortion internally

Your k2/k3 are fake overfit values


The Correct Way (Official + Accurate)

RealSense provides:

rs.rs2_deproject_pixel_to_point()


This converts pixel + depth → 3D coordinate using factory intrinsics.


## ✅ Correct Pipeline (Do This)
1️⃣ Align depth to color
align = rs.align(rs.stream.color)
frames = align.process(frames)


This is CRITICAL.

Otherwise your pixel won’t match depth.

2️⃣ Get depth at pixel
depth = depth_frame.get_distance(u, v)


This returns depth in meters.

3️⃣ Get intrinsics
color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

4️⃣ Deproject
point = rs.rs2_deproject_pixel_to_point(
    color_intrinsics,
    [u, v],
    depth
)


Now:

point[0] = X
point[1] = Y
point[2] = Z


In meters.

