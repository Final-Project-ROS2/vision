the key to connecting is using the Cvbridge() class to create a bridge between ROS2 and OPENCV

the image is puslished to camera/image_waw so subscribe to that
self.lastest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8)

Then we can use the CvBridge() we declared earlier to convert the mmsg from camera/image_raw by using imgsmsg_to_cv2

cv2.imshow RSGB camera image, self.lastest_image


Now we can use the image like any other opencv message