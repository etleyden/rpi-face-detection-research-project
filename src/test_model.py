# Test the accuracy of a haar feature-based cascade classifier
# - Output all the IOU values for every identified face
# - Compute the precision and recall of the model over the threshold curve (where the threshold is IOU)
# - Write all of this information to files, so it can be viewed instead of having to run the model again. 
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
    
    predicted_faces = {}
    actual_faces = {}

    files = os.listdir(image_directory)

    for k in np.linspace(1.05, 1.4, 35):
        for l in range(1, 11):
            predicted_faces[(k, l)] = []
            actual_faces[(k, l)] = []
            count = 0
            for image_filename in tqdm(files, desc=f"Face detection for scaleFactor {k} and minNeighbors {l}"):
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

                # draw the images
                # for face in faces:
                #     cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)
                # for face in objs:
                #     cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 255, 255), 2)
                # cv2.imwrite(f"{count}.jpg", img) 

                # log predicted and actual faces
                predicted_faces[(k, l)].append(faces)
                actual_faces[(k, l)].append(objs)

                count += 1
                if count > 50: break

    fig, axs = plt.subplots(1, 2)

    ious_per_threshold = {}
    precision_per_threshold = {}
    recall_per_threshold = {}
    f1_per_threshold = {}
    for k in np.linspace(1.05, 1.4, 35):
        for l in range(1, 11):
            ious_per_threshold[(k, l)] = []
            precision_per_threshold[(k, l)] = []
            recall_per_threshold[(k, l)] = []
            f1_per_threshold[(k, l)] = []
            for i in np.linspace(0, 1, 100):
                total_tp = 0
                total_fn = 0
                total_fp = 0
                ious_in_threshold = []
                for j_idx, j in enumerate(predicted_faces[(k, l)]):
                    tp, fn, fp, ious = match_boxes(predicted_faces[(k, l)][j_idx], actual_faces[k, l][j_idx], i)
                    total_tp += tp
                    total_fn += fn
                    total_fp += fp
                    ious_in_threshold += ious
                precision = total_tp / (total_tp + total_fp)
                recall = total_tp / (total_tp + total_fn)
                precision_per_threshold[(k, l)].append(precision)
                recall_per_threshold[(k, l)].append(recall)
                f1_per_threshold[(k, l)].append((2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0)
                ious_per_threshold[(k, l)].append(ious_in_threshold)
        
    # save the precision/recall curve
    max_index = max(range(len(f1_per_threshold[(1.1, 5)])), key=f1_per_threshold[(1.1, 5)].__getitem__) 

    axs[0].hist(ious_per_threshold[(1.1, 5)][max_index], bins=50)
    axs[0].set_xlabel("IOU Distribution")
    axs[0].set_ylabel("Count")
    for k in np.linspace(1.05, 1.4, 35):
        for l in range(1, 11): 
            axs[1].scatter(recall_per_threshold[(k, l)], precision_per_threshold[(k, l)])
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_xlim(left=0, right=1.0)
    axs[1].set_ylim(bottom=0, top=1.0)
    fig.tight_layout()

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

def match_boxes(predicted_boxes, actual_boxes, iou_threshold):
    ground_truth_boxes = actual_boxes.copy()
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    iou_list = []
    # Match predicted boxes with ground truth boxes
    for pred_box in predicted_boxes:
        best_iou = 0
        best_gt_box = None
        
        for gt_box in ground_truth_boxes:
            iou_value = iou(pred_box, gt_box)
            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_box = gt_box
        
        if best_iou >= iou_threshold and best_gt_box is not None:
            true_positives += 1
            ground_truth_boxes.remove(best_gt_box)  # Remove matched ground truth box
        else:
            false_positives += 1
        iou_list.append(best_iou)

    # Any remaining ground truth boxes are false negatives
    false_negatives = len(ground_truth_boxes)
    return true_positives, false_negatives, false_positives, iou_list

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

def detect_faces(face_cascade: cv2.CascadeClassifier, img, scaleFactor=1.1, minNeighbors=5):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        grayscale_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return [face.tolist() for face in faces]


if __name__ == "__main__":
    main()

