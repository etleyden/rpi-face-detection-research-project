import os, sys
import random
import cv2

# only 1317 samples in the negative directory, so we have to limit pos samples

def generate_negatives_txt():
    # create the negatives file: {[path]}
    negatives_path = "archive/neg_images"
    is_first = True
    with open('negatives.txt', 'w') as f:
        for filename in os.listdir(negatives_path):
            if is_first:
                is_first = False
            else:
                f.write("\n")
            f.write(f'{negatives_path}/{filename}')

# create the positives file: {[path] [num_objects] [[x, y, w, h],...]}
images_dir = 'archive/pos_images/train'
labels_dir = 'archive/yolo_labels/train'

# Path for the output .info file
output_info_file = 'positives.info'

def convert_yolo_to_opencv(yolo_bbox, img_width, img_height):
    """
    Convert YOLO format (x_center, y_center, width, height) to OpenCV format (x, y, width, height).
    - YOLO bbox is normalized, so we multiply by the image width and height.
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert to absolute coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Calculate the top-left corner (x, y)
    x = int(x_center - (width / 2))
    y = int(y_center - (height / 2))
    
    return x, y, int(width), int(height)

def generate_positives_info():
    # Open the output .info file
    with open(output_info_file, 'w') as info_file:
        pos_imgs = []
        # Loop through all the images in the images directory
        for image_filename in os.listdir(images_dir):
            current_img = ""
            # Ensure the file is an image
            bad_image = False
            if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
                image_path = os.path.join(images_dir, image_filename)
                
                # Read the corresponding YOLO annotation
                label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = os.path.join(labels_dir, label_filename)
                
                # Check if the annotation file exists
                if not os.path.exists(label_path):
                    print(f"Warning: Annotation for {image_filename} not found.")
                    continue
                
                # Read the image to get its dimensions
                img = cv2.imread(image_path)
                img_height, img_width = img.shape[:2]
                
                # Read the YOLO annotation file
                with open(label_path, 'r') as label_file:
                    lines = label_file.readlines()
                    
                    # Number of objects in the image
                    num_objects = len(lines)
                    
                    # Write image path and number of objects to the .info file
                    current_img += f"{image_path} {num_objects}"
                    
                    # For each object in the annotation file
                    for line in lines:
                        # YOLO format: class_id x_center y_center width height
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        bbox = [float(coord) for coord in parts[1:]]  # x_center, y_center, width, height
                        
                        # Convert YOLO bbox to OpenCV bbox format
                        x, y, width, height = convert_yolo_to_opencv(bbox, img_width, img_height)
                        
                        # Write the bounding box info to the .info file
                        if width * height <= 0:
                            print(f"invalid bounding box: {image_path}")
                            bad_image = True
                            break
                        current_img += f" {x} {y} {width} {height}"
                    
            if not bad_image: pos_imgs.append(current_img)
            
        # select a number of negatives
        print(f"NUM SAMPLES: {len(pos_imgs)}")

        is_first=True
        for img in pos_imgs:
            if is_first:
                is_first = False
            else: 
                info_file.write("\n")

            info_file.write(f"{img}")

    print(f"positives.info file generated successfully at {output_info_file}.")

# Run the script
if __name__ == '__main__':
    generate_negatives_txt()
    generate_positives_info()
