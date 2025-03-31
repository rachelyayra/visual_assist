from ultralytics import YOLO
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "MiDaS_small" 

midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  

def detect_objects(image_path):
    results = yolo_model(image_path)
    return results[0].boxes


def filter_objects(drivable_mask, results):
    detected_objects = results.xyxy.cpu().numpy()  # Extract bounding boxes

    filtered_objects = []
    for box in detected_objects:
        x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates

        # Extract object region from the drivable mask
        object_mask = drivable_mask[y1:y2, x1:x2]
        
        # If a significant portion of the object is in the drivable space, keep it
        if np.sum(object_mask) / object_mask.size > 0.5:  
            filtered_objects.append(box)

    return filtered_objects


# Function to get depth map
def get_depth_map(image):
    input_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(f'Depth MAP {input_tensor.shape}')
    input_tensor = transform(input_tensor).to(device) 
    with torch.no_grad():
        prediction = midas(input_tensor).squeeze()

    depth_map = prediction.cpu().numpy()
    resized_depth = cv2.resize(
    depth_map, 
    (image.shape[1], image.shape[0]), 
    interpolation=cv2.INTER_NEAREST  # Best for discrete depth values
)
    print(f'Depth MAP {resized_depth.shape}')
    plt.imshow(resized_depth, cmap='viridis')  # Or 'plasma', 'magma' for depth
    plt.axis('off')  # Remove axes
    plt.savefig('../outputs/depth_map.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close() 
    
    return resized_depth 

def estimate_distance(objects, drivable_space, depth_map):
    min_distance = float('inf')
    closest_object = None
    

    for (x1, y1, x2, y2) in objects:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # Check if object is inside drivable space
        depth_map = np.nan_to_num(depth_map, nan=1000, posinf=1000, neginf=1000)
        object_depth = np.median(depth_map[y1:y2, x1:x2])
        print(depth_map.shape)
        print(object_depth)
        estimated_distance = 10 / (object_depth + 1e-6)  # Scale factor (adjust based on real-world tests)
        if estimated_distance < min_distance:
            min_distance = estimated_distance
            closest_object = (x1, y1, x2, y2, estimated_distance)

    return closest_object, min_distance