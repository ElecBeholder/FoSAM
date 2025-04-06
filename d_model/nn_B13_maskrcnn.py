import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex  # 或者自己实现 mIoU 计算

# 加载预训练的 DeepLab 模型
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# 预处理图片
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = T.Compose([
        T.Resize((1024, 1024)),  # 根据模型需求调整大小
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0), image

# 加载图片和标签
image_path = '/home/lwx/b_data_train/data_c_cook/cityscapes_rgblabel/train/sp30/erfurt-000097-000019_3x512x512.float32.X.png'
label_path = '/home/lwx/b_data_train/data_c_cook/cityscapes_rgblabel/train/sp30/erfurt-000097-000019_1x512x512.uint8.Y.pt'
input_image, original_image = preprocess_image(image_path)

# 加载标签
true_label_tensor = torch.load(label_path, weights_only=True)  # 使用 weights_only=True 来避免警告
true_label_tensor = true_label_tensor.squeeze()  # 如果标签张量有额外维度，去掉它

# 推理
with torch.no_grad():
    output = model(input_image)['out'][0]
pred = torch.argmax(output, dim=0).cpu()

# # 计算 mIoU
# miou_calculator = JaccardIndex(num_classes=21, task='multiclass')  # 添加 task='multiclass'
# miou = miou_calculator(pred, true_label_tensor)
# print(f"mIoU: {miou:.4f}")

# 可视化原图、预测结果和真实标签
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(original_image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(pred, cmap='jet')
ax[1].set_title('Predicted Segmentation')
ax[1].axis('off')

ax[2].imshow(true_label_tensor, cmap='jet')
ax[2].set_title('Ground Truth')
ax[2].axis('off')

plt.tight_layout()
plt.savefig('./test.png')
