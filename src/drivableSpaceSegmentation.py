import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights 
import cv2
import numpy as np
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(ROOT_DIR)  
from BiSeNet.lib.models import BiSeNetV2
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

# Initialize model (19 Cityscapes classes)
model = BiSeNetV2(n_classes=19)
model.load_state_dict(torch.load('../model_final_v2_city.pth', map_location='cpu'), strict=False)

model.eval()


# Define transformation for input image
def preprocess(img):
    img = cv2.resize(img, (1024, 512))  # BiSeNet's default input size
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = torch.from_numpy(img.copy()).float().unsqueeze(0) / 255.0 
    return img.to("cuda" if torch.cuda.is_available() else "cpu")

def segment_image(image):
    image_tensor = preprocess(image)

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)[0].squeeze()

    output = output.numpy()
    # print("Input image shape:", image_tensor.shape)
    output_predictions = output.argmax(0)

    mask = output_predictions
    large_mask = np.isin(mask, [0]).astype(np.uint8) * 255
    col_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    col_mask[large_mask == 255] = [0, 255, 0]  # Green (BGR) for roads
    col_mask[large_mask == 0] = [0, 0, 0] 
    cv2.imwrite("../outputs/drivable_space_mask.png", col_mask)

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                      interpolation=cv2.INTER_NEAREST)
    # print("Mask output shape:", mask.shape)
    # Now create colored mask with matching dimensions
    mask = np.isin(mask, [0]).astype(np.uint8) * 255
    colored_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # print("Colored Mask output shape:", colored_mask.shape)
    colored_mask[mask == 255] = [0, 255, 0]  # Green (BGR) for roads
    colored_mask[mask == 0] = [0, 0, 0]    # Red (BGR) for non-roads


    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    cv2.imwrite("../outputs/segmentation.png", overlay)


    return mask

