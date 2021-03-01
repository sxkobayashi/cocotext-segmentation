"""
Bounding box utilities.
All boxes are represented as (left, top, width, height)
"""
import csv

def read_boxes_from_csv(csv_filename):
    """ csv file should contain lines of 5-tuple (filename, x, y, w, h)
    """
    boxes_dict = {}
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            filename = row[0]
            box = [round(float(x)) for x in row[1:]]
            boxes_dict[filename] = box
    return boxes_dict


def int_box(box):
    return [round(n) for n in box]


def center_box(im_width, im_height, box_width, box_height):
    """ Get the center box of an image.
    """
    x_center = im_width // 2
    y_center = im_height // 2
    x = x_center - box_width // 2
    y = y_center - box_height // 2
    return [x, y, box_width, box_height]


def shift_box(box, x_shift, y_shift, im_width, im_height):
    """ Shift the box by (y_shift, x_shift) within (im_height, im_width)
    If the shift is outside of the image region it is forced back
    """
    x1,y1,w,h = box

    w = min(w, im_width)
    h = min(h, im_height)

    x2 = x1 + x_shift
    y2 = y1 + y_shift
    
    x2 = min(x2, im_width - w)
    y2 = min(y2, im_height - h)
    
    x2 = max(x2, 0)
    y2 = max(y2, 0)

    return [x2, y2, w, h]


def area(box):
    return box[2] * box[3]


def cut_within_frame(box, im_width, im_height):
    """ Shift a box (x,y,w,h) into the coordinate of [0 ~ im_width, 0 ~ im_height].
    """
    x1,y1,w,h = box
    x2, y2 = x1 + w - 1, y1 + h - 1

    x2 = min(x2, im_width - 1)
    y2 = min(y2, im_height - 1)

    x1 = max(0,x1)
    y1 = max(0,y1)

    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


def overlapped(box1, box2):
    """ Test if box1 and box2 overlaps. boxes are represented as [x,y,w,h], consistent with coco-text.
    """
    box_left, box_right = (box1, box2) if box1[0] < box2[0] else (box2, box1)
    x1,y1,w1,h1 = box_left
    x2,y2,w2,h2 = box_right
    if x1 + w1 < x2 or y1 > y2 + h2 or y2 > y1 + h1:
        return False
    return True


def rescale_box(box, ratio):
    """ Rescale the box [x,y,w,h] at its center.
    """
    x,y,w,h = box
    x_center, y_center = x + w/2, y + h/2
    w_new, h_new = w * ratio, h * ratio
    x_new, y_new = x_center - w_new / 2, y_center - h_new / 2
    return [x_new, y_new, w_new, h_new]


def merge_box(box1, box2):
    """ Merge two boxes by its axis-aligned union.
    """
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    rect = [min(x1,x2), min(y1,y2), max(x1+w1, x2+w2), max(y1+h1, y2+h2)]
    merged_box = [rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]]
    return merged_box


def get_overlapped_box_pair(boxes):
    if len(boxes) < 2:
        return []
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            box1 = rescale_box(boxes[i], 1.5)
            box2 = rescale_box(boxes[j], 1.5)
            if overlapped(box1, box2):
                return [i, j]
    return []


def merge_bounding_boxes(boxes):

    overlapped_pair = get_overlapped_box_pair(boxes)

    while len(overlapped_pair) > 0:
        idx1, idx2 = overlapped_pair
        box1, box2 = boxes[idx1], boxes[idx2]
        box_merged = merge_box(box1, box2)
        del boxes[idx2]
        del boxes[idx1]
        boxes.append(box_merged)
        overlapped_pair = get_overlapped_box_pair(boxes)


def filter_small_boxes(boxes, minimum_area):
    return [x for x in boxes if area(x) > minimum_area]



if __name__=='__main__':
    # Unit tests
    assert center_box(8, 8, 6, 4) == [1,2,6,4]
    assert [2,3,6,4] == shift_box([1,2,6,4], 1, 1, 8, 8)
    assert [0,1,6,4] == shift_box([1,2,6,4], -1, -1, 8, 8)
    assert [0,0,6,4] == shift_box([1,2,6,4], -2, -3, 8, 8)
