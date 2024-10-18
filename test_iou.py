def iou(box1, box2):
    # Box format: [x_min, y_min, x_max, y_max]
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    # Calculate intersection area
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    
    if x_min_inter < x_max_inter and y_min_inter < y_max_inter:  # There is an overlap
        intersection = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    else:
        intersection = 0  # No overlap
    
    # Calculate areas of the bounding boxes
    area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area_box2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    # Union area
    union = area_box1 + area_box2 - intersection
    
    # IoU
    iou_value = intersection / union
    return iou_value

def match_boxes(predicted_boxes, ground_truth_boxes, iou_threshold):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    iou_list = []
    # Match predicted boxes with ground truth boxes
    for pred_box in predicted_boxes:
        best_iou = 0
        best_gt_box = None
        
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            iou_value = iou(pred_box, gt_box)
            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_box = gt_box
        
        print(best_gt_box)
        if best_iou >= iou_threshold:
            true_positives += 1
            ground_truth_boxes.remove(best_gt_box)  # Remove matched ground truth box
        else:
            false_positives += 1
        iou_list.append(best_iou)

    # Any remaining ground truth boxes are false negatives
    false_negatives = len(ground_truth_boxes)
    return true_positives, false_negatives, false_positives, iou_list

    # Example usage:
ground_truth_boxes = [[50, 50, 150, 150], [200, 200, 300, 300]]  # Example ground truth boxes
predicted_boxes = [[55, 55, 145, 145], [250, 250, 290, 290]]  # Example predicted boxes
true_positives, false_positives, false_negatives, ious = match_boxes(predicted_boxes, ground_truth_boxes, 0.5)

print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")