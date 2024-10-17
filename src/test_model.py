import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from jsonargparse import ArgumentParser, ActionConfigFile
from tqdm import tqdm

if len(sys.argv) < 2:
    print(f"Usage: python test_model.py <model_xml>")


def main():
    parser = ArgumentParser()

    # add config
    parser.add_argument('--config', action=ActionConfigFile)
    
    # Add other args
    parser.add_argument('--model', type=str, help="Path to a pretrained haar feature-based cascade classifier model.xml")
    parser.add_argument('--images', type=str, help="Base path to the test image set")
    parser.add_argument('--annotations', type=str, help="Path to .mat annotations")
    parser.add_argument('--iou_out', type=str, help="Path to store list of IOU results")
    parser.add_argument('--results_out', type=str, help="Path to store precision and recall")

    # parse the arguments
    args = parser.parse_args()
    model_path = args.model
    image_directory = args.images
    annotations_path = args.annotations
    iou_out = args.iou_out
    results_out = args.results_out
    print(f"""
model path: {model_path}
image dir: {image_directory}
annotations: {annotations_path}
""")

    # set up the haar classifier
    face_cascade = cv2.CascadeClassifier(model_path)
    
    ious = []

    files = os.listdir(image_directory)

    total_expected_faces = 0
    
    for image_filename in tqdm(files, desc="Detecting faces for each image in the directory..."):
        f = os.path.join(image_directory, image_filename)

        #detect faces
        img = cv2.imread(f)
        img_height, img_width = img.shape[:2]
        faces = detect_faces(face_cascade, img)

        
        #load the annotations
        with open(os.path.join(annotations_path, image_filename.replace("jpg", "txt"))) as ann_f:
            annotations = ann_f.read().splitlines()
            num_obj = len(annotations)
            objs = []
            for line in annotations:
                bbox = [float(coord) for coord in line.split()[1:]]
                objs.append(list(convert_yolo_to_opencv(bbox, img_width, img_height)))
            total_expected_faces += num_obj
        
        # compute the IOU values
        ious += match_boxes(faces, objs)


    fig, axs = plt.subplots(1, 2)

    # save raw iou results
    with open(iou_out) as f:
        for iou in ious:
            f.write(f"{iou}\n")
    
    # save the precision/recall curve
    with open(results_out) as f:
        curve = []
        f.write(f"False Positives: {correct.count(0.0)}\n")
        for i in np.linspace(0, 1, 100):
            correct = sum(x > i for x in ious)
            precision = correct / len(ious)
            recall = correct / total_expected_faces
            curve.append((recall, precision))
        
        f.write(f"{precision, recall}\n")

    recall_list, precision_list = zip(*curve)
    axs[0].hist(ious, bins=50)
    axs[1].scatter(recall_list, precision_list)

    plt.show() 


def iou(boxA, boxB):
    # boxA and boxB are both in the format [x, y, width, height]
    
    # Coordinates of the intersection box
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    # Area of both the predicted and expected boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    # Compute the area of the union
    unionArea = boxAArea + boxBArea - interArea
    
    # Compute the IoU
    iou_value = interArea / unionArea if unionArea > 0 else 0
    
    return iou_value

def match_boxes(predicted_boxes, expected_boxes):
    top_n = len(predicted_boxes)
    iou_list = []
    
    # Compute IoU for all pairs of predicted and expected boxes
    for i, pred_box in enumerate(predicted_boxes):
        for j, exp_box in enumerate(expected_boxes):
            iou_value = iou(pred_box, exp_box)
            #iou_list.append(iou_value)
            iou_list.append((iou_value, i, j))  # (IoU value, predicted index, expected index)
    
    # Sort by IoU in descending order
    iou_list.sort(reverse=True, key=lambda x: x[0])
    
    # Initialize a set to keep track of the matched indices
    matched_pred = set()
    matched_exp = set()
    
    # Find the top `n` pairs with the highest IoU values
    top_matches = []
    for iou_value, i, j in iou_list:
        if i not in matched_pred and j not in matched_exp:
            top_matches.append(iou_value)
            matched_pred.add(i)
            matched_exp.add(j)
        if len(top_matches) >= top_n:
            break
    
    return top_matches


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

def detect_faces(face_cascade: cv2.CascadeClassifier, img):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        grayscale_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return list(faces)


if __name__ == "__main__":
    main()

