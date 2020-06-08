"""
This scripts hold helper functions for different purposes.
Almost each module script in lpr is using one or more functions from here.

Methods:
    check_standart(plate_name):          To count province code, middle letters and last numbers on plate to understand if it is suitable for standart TR licence plate
    get_coords(frame,pts):               Draw rectangle on points and return real coordinate values instead of normalized ones
    reorder_motorbike_plate(plate_name): Reorder if licence plate characters are mixed or does not standart. For TR licence plates
    change_char_from_height(L):          Change character order according to characters height(y) coordinates for motorbike licence plates
    show(frame):                         Show function is using to opencv imshow method
"""

import cv2

try:
    import NotNumpyArrayError
except:
    try:
        from platerecognition.exceptions import NotNumpyArrayError
    except:
        try:
            from LPR.platerecognition.exceptions import NotNumpyArrayError
        except:
            from detection.LPR.platerecognition.exceptions import NotNumpyArrayError


def check_standart(plate_name):
    """
    Check plate country standart format

    Turkish licence plates format should be (Provience Code[1-2] [1-3] [2-4]) characters maximum.

    Parameters:
        plate_name(str): Plate string

    Return:
        flags(list): Contains 3 format boolean array
    """

    if plate_name == "":
        return [False, False, False]

    province_code = 0
    letters = 0
    last_numbers = 0
    provinceFound = False

    for i in plate_name:
        if i.isnumeric():
            if provinceFound == False:
                province_code += 1
            else:
                last_numbers += 1
        else:
            letters += 1
            provinceFound = True

    flags = []

    if (province_code < 2):
        # print("Province code missing...")
        flags.append(False)
    elif (province_code > 2):
        # print("Province code more than expected...")
        flags.append(False)
    else:
        flags.append(True)

    if (letters < 1):
        # print("Some letters are missing...")
        flags.append(False)
    elif (letters > 3):
        # print("There are letters more than expected(3)...")
        flags.append(False)
    else:
        flags.append(True)

    if (last_numbers < 2):
        # print("Some numbers in last digits on plate are missing...")
        flags.append(False)
    elif (last_numbers > 4):
        # print("There are numbers in last digits on plate more than expected(4)...")
        flags.append(False)
    else:
        flags.append(True)

    if (flags[0] == True):
        if (int(plate_name[:2]) > 81):
            flags[0] = False
    if (flags[0] == True):
        if (int(plate_name[:2]) <= 0):
            flags[0] = False

    return flags


def nms(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def get_coords(frame, pts):
    if type(frame).__module__ is not 'numpy':
        raise NotNumpyArrayError

    pt_1 = []
    pt_2 = []
    for i in range(4):
        pt1 = tuple(pts[:, i].astype(int).tolist())
        pt2 = tuple(pts[:, (i + 1) % 4].astype(int).tolist())
        pt_1.append(pt1)
        pt_2.append(pt2)
        # cv2.line(frame,pt1,pt2,(0,255,0),2)
    return pt_1, pt_2


def distance(a, b):
    sum = 0
    for i in range(2):
        for j in range(4):
            for k in range(2):
                sum += (a[i][j][k] - b[i][j][k])
    return sum


def remove_duplicated_plates(coords, strings, standarts, cropeds):
    size = len(coords)
    indices = []
    for i in range(size):
        for j in range(size - (i + 1)):
            dist = distance(coords[i], coords[j + (i + 1)])
            if (dist > -110 and dist < 110):
                indices.append(j + (i + 1))

    indices = list(set(indices))  ##remove duplicated indexes
    for index in sorted(indices, reverse=True):
        del coords[index]
        del strings[index]
        del standarts[index]
        del cropeds[index]

    return coords, strings, standarts, cropeds


def reorder_motorbike_plate(plate_name):
    """
    Reorder if licence plate characters are mixed or does not standart. For TR licence plates

    Parameters:
        plate_name(str): Digital name of plate

    Return:
        new_plate(str): Reordered licence plate if process performed. Otherwise, return original frame

    Tip:
        Uncomment "prints" right below to understand clearly what happens after process plate characters. Feed function with motorbike image!
    """

    ##if province code has been found
    if plate_name[:2].isnumeric():
        new_plate = plate_name[:2]
        a = b = ""
        for i in range(len(plate_name) - 2):
            if (plate_name[i + 2].isnumeric()):
                a += plate_name[i + 2]
            else:
                b += plate_name[i + 2]
        new_plate += b + a
        return new_plate
    return plate_name


def change_char_from_height(L):
    """
    Change character order according to characters height(y) coordinates for motorbike licence plates

    Parameters:
        L(Dshape): A licence dshape(darknet special shape object) that holds box coordinates of characters

    Return:
        L(Dshape): Reordered plate characters according to their y coordinate value
    """
    if (len(L) < 3):
        return L

    digit_1 = L[0]
    digit_2 = L[1]

    y1 = digit_1.tl()[1]
    y2 = digit_2.tl()[1]

    if ((y2 - y1) > 0.2):
        for i in range(len(L) - 2):
            if (L[i + 2].tl()[1] - y1) < 0.1:
                temp = L[i + 2]
                L[i + 2] = L[1]
                L[1] = temp
                break
    return L


def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T

        A[i * 2, 3:6] = -xil[2] * xi
        A[i * 2, 6:] = xil[1] * xi
        A[i * 2 + 1, :3] = xil[2] * xi
        A[i * 2 + 1, 6:] = -xil[0] * xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))

    return H


def show(frame):
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
