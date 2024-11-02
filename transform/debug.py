import cv2


label_map = ('R', 'INACTIVE', 'ACTIVE')
image_path = './img/173055440074.png'
label_path = './txt/173055440074.txt'

image = cv2.imread(image_path)

with open(label_path, 'r') as f:
    lines = f.readlines()

    for line in lines:
        elems = line.strip().split(' ')

        label = int(elems[0])
        for i in range(1, 11, 2):
            x = int(float(elems[i]))
            y = int(float(elems[i + 1]))
            cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

        cv2.putText(image, label_map[label], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imshow('debug_img', image)
cv2.waitKey(0)
