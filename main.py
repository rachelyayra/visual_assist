import cv2
import numpy


# Import model scripts
from drivableSpaceSegmentation import segment_image
from objectDetection import detect_objects, filter_objects, get_depth_map, estimate_distance
# Load models

# Load image
def main(input_path = "./input/sidwal2.jpg"):
    
    image = cv2.imread(input_path)

    # Step 1: Drivable Space Segmentation
    drivable_space_mask = segment_image(image)

    # Step 2: Obstacle Detection (Find objects in the image)
    obstacles = detect_objects(image)
    # print(obstacles)

    # Step 3: Mask obstacles outside drivable space
    filtered_obstacles = filter_objects(drivable_space_mask, obstacles)
    print(filtered_obstacles)

    # Step 4: Depth Estimation (Estimate depth of filtered obstacles)
    depth_map = get_depth_map(image)
    closest_object, distance = estimate_distance(filtered_obstacles, drivable_space_mask, depth_map)

    if closest_object:
        x1, y1, x2, y2, estimated_distance = closest_object
        print(f"Distance: {estimated_distance:.2f} meters")
    else:
        print("No obstacles detected in drivable space.")
    return estimate_distance

if __name__ == "__main__":
    main()