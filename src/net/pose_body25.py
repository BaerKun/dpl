from pose_body25.posebody25detector import PoseBody25Detector, draw_body_pose
import cv2

detector = PoseBody25Detector('../../model/pose_body25.pt')
img = cv2.imread('COCO_val2014_000000000192.jpg')
poses = detector(img)
img = draw_body_pose(img, poses)
cv2.imshow("pose body 25", img)
cv2.waitKey()
