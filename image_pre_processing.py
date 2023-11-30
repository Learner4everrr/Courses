import cv2
import zipfile
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

class ZipDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Assume the zip file contains image pairs, e.g., 'image1.jpg' and 'image2.jpg'.
            self.file_list = [name for name in zip_ref.namelist() if name.endswith('.jpg')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Load data for the given index.
            file_name = self.file_list[index]
            with zip_ref.open(file_name) as file:
                # Process the data, e.g., read an image.
                image = Image.open(file).convert('RGB')

            # Apply transformations if provided.
            if self.transform:
                image = self.transform(image)

            return image

# 用于加载MTCNN模型
detector = MTCNN()

# 设置CelebA数据集的路径
data_root = "./celeba"

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.ToTensor(),
])

# 加载CelebA数据集
celeba_dataset = ZipDataset(zip_path='./celeba/celeba/celeba.zip', transform=transform)
celeba_dataloader = DataLoader(celeba_dataset, batch_size=1, shuffle=True)

print('dataset loaded')

# 遍历CelebA数据集中的图像
for i, data in enumerate(celeba_dataloader, 1):
    image = data
    img_np = image.squeeze().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)

    img_np_cp = img_np.copy()
    #img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #img_np = img_np.astype(np.float32) * 255.0
    #img_np = img_np.astype(np.uint8)
    #print(img_np)
    #plt.imshow(img_np)
    #plt.title(f"Image {i}")
    #cv2.imwrite('t.jpg', img_np)
    #plt.show()

    '''img_np = cv2.imread('w.jpg')
                print(img_np.shape)
                plt.imshow(img_np)
                plt.title(f"Image {i}")
                plt.show()'''



    # Convert img_np to np.uint8
    #img_np = img_np.astype(np.uint8)
    print(img_np.shape)

    # 检测人脸
    faces = detector.detect_faces(img_np)
    print(faces)

    # 如果检测到人脸，标记眼睛、鼻子、嘴巴和耳朵
    if faces:
        face = faces[0]
        x, y, w, h = face['box']
        left_eye = face['keypoints']['left_eye']
        right_eye = face['keypoints']['right_eye']
        nose = face['keypoints']['nose']
        mouth_left = face['keypoints']['mouth_left']
        mouth_right = face['keypoints']['mouth_right']
        #left_ear = face['keypoints']['left_ear']
        #right_ear = face['keypoints']['right_ear']

        # 在眼睛位置画白色矩形
        line_width = 20
        mask_color = (255, 255, 0)
        cv2.rectangle(img_np_cp, (int(left_eye[0])-line_width, int(left_eye[1])-line_width), (int(left_eye[0])+line_width, int(left_eye[1])+line_width), mask_color, 4*line_width)
        cv2.rectangle(img_np_cp, (int(left_eye[0])-line_width, int(left_eye[1])-line_width), (int(left_eye[0])+line_width, int(left_eye[1])+line_width), mask_color, 4*line_width)
        cv2.rectangle(img_np_cp, (int(right_eye[0])-line_width, int(right_eye[1])-line_width), (int(right_eye[0])+line_width, int(right_eye[1])+line_width), mask_color, 4*line_width)
        cv2.rectangle(img_np_cp, (int(nose[0])-line_width, int(nose[1])-line_width), (int(nose[0])+line_width, int(nose[1])+line_width), mask_color, 4*line_width)
        cv2.rectangle(img_np_cp, (int(mouth_left[0])-line_width, int(mouth_left[1])-line_width), (int(mouth_left[0])+line_width, int(mouth_left[1])+line_width), mask_color, 4*line_width)
        cv2.rectangle(img_np_cp, (int(mouth_right[0])-line_width, int(mouth_right[1])-line_width), (int(mouth_right[0])+line_width, int(mouth_right[1])+line_width), mask_color, 4*line_width)
        #cv2.rectangle(img_np, (int(left_ear[0])-5, int(left_ear[1])-5), (int(left_ear[0])+5, int(left_ear[1])+5), (255, 255, 255), -1)
        #cv2.rectangle(img_np, (int(right_ear[0])-5, int(right_ear[1])-5), (int(right_ear[0])+5, int(right_ear[1])+5), (255, 255, 255), -1)

        # 显示标记后的图片
        #plt.figure(1)
        print(i)
        plt.subplot(2, 2, int(i*2-1))
        plt.imshow(img_np)  # Display the first image in the first subplot
        #plt.title(f"Image {i}: original")

        plt.subplot(2, 2, int(i*2))
        plt.imshow(img_np_cp)  # Display the second image in the second subplot
        #plt.title(f"Image {i}: Eyes, nose, mouth marked")

        cv2.save(img_np_cp, 'res.png')
        

    if i == 2:
        break

plt.show()
