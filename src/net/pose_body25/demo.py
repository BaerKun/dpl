from posebody25detector import PoseBody25Detector
from util import draw_body_pose
import cv2
import utils
import os

weights_path = utils.project_root + "/weights/pose_body25.pt"
image_dir = os.path.join(utils.project_root, "images")

detector = PoseBody25Detector(weights_path)
for img_path in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, img_path))
    poses = detector(img, show_heatmap_paf=False)
    img = draw_body_pose(img, poses)
    cv2.imshow("pose body 25", img)
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()
        break