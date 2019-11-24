import cv2
import time
import numpy as np
from random import randint

import colorAnalysis

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

DEBUG = True
#Display image
def display(img, frameName="OpenCV Image"):
    if not DEBUG:
        return
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)

def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output, detected_keypoints, frameWidth, frameHeight):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            #print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

import pandas as pd

def findColorValue(codeList):
    colordfs = pd.read_excel("color_names.xlsx", sheet_name="Sheet1")

    colorList = []
    for idx, row in colordfs.iterrows():
        #print(row["code"])
        codes = row["code"].split(",")
        if len(codes) == 3:
            colorList.append({"name": row["Name"], "dist": 0, "r": float(codes[0]), "g": float(codes[1]), "b": float(codes[2])})

    for item in colorList:
        item["dist"] = abs(codeList[0] - item["r"]) + abs(codeList[1] - item["g"]) + abs(codeList[2] - item["b"])

    # print(colorList)
    colorList = sorted(colorList, key=lambda k: k["dist"])

    # redDist = abs(codeList[0] - 255) + codeList[1] + codeList[2]
    # greenDist = abs(codeList[1] - 255) + codeList[0] + codeList[2]
    # blueDist = abs(codeList[2] - 255) + codeList[1] + codeList[0]

    # if redDist < greenDist:
    #     if redDist < blueDist:
    #         minColor = "Red"
    #     else:
    #         minColor = "Blue"
    # else:
    #     if greenDist < blueDist:
    #         minColor = "Green"
    #     else:
    #         minColor = "Blue"
    return colorList[0]["name"]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def analyseImage(image1):
    #image1 = cv2.imread("img4.jpg")
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]

    t = time.time()
    

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight / frameHeight) * frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    #print("Time Taken in forward pass = {}".format(time.time() - t))

    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        #print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    frameClone = image1.copy()
    # frameClone = np.zeros_like(image1)
    # for i in range(nPoints):
    #     for j in range(len(detected_keypoints[i])):
    #         cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
    # cv2.imshow("Keypoints",frameClone)

    valid_pairs, invalid_pairs = getValidPairs(output, detected_keypoints, frameWidth, frameHeight)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

    upperbodyPairs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                      [1, 8], [1, 11]]

    lowerbodyPairs = [[8, 9], [9, 10], [11, 12], [12, 13]]



    colorData = {}
    idx = 0
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    for n in range(len(personwiseKeypoints)):
        # if n > 0:
        #     break
        idx += 1
        upperbodyPoints = []
        lowerbodyPoints = []
        colorName1 = ""
        colorName2 = ""
        for i in range(17):
            if (POSE_PAIRS[i] in upperbodyPairs):
                # continue
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                #cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 2, cv2.LINE_AA)
                upperbodyPoints.append([B[0], A[0]])
                upperbodyPoints.append([B[1], A[1]])

            if (POSE_PAIRS[i] in lowerbodyPairs):
                # continue
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                #cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 2, cv2.LINE_AA)
                lowerbodyPoints.append([B[0], A[0]])
                lowerbodyPoints.append([B[1], A[1]])

        # for pt in upperbodyPoints:
        #     cv2.circle(frameClone, (pt[0], pt[1]), 8, colors[i], -1, cv2.LINE_AA)
        if len(upperbodyPoints) > 0:
            x, y, w, h = cv2.boundingRect(np.array(upperbodyPoints, np.float32))
            #cnt = np.array(upperbodyPoints, np.float32)
            #cnt = np.array(cnt).reshape((-1, 1, 2)).astype(np.int32)
            # cv2.drawContours(frameClone, cnt, 0, (0, 0, 255), 2)

            cv2.rectangle(frameClone, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi = frameClone[y:y + h, x:x + w]
            color = colorAnalysis.main(roi)
            colorName1 = findColorValue(color)
            (label_width, label_height), baseline = cv2.getTextSize(colorName1, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(frameClone, (x, y - label_height - 10), (x + label_width + 5, y), (255, 255, 255), -1)
            cv2.putText(frameClone, colorName1, (x, y - 5), FONT, 0.7, (255, 0, 0), FONT_THICKNESS)

        #print(colorName1)

        if len(lowerbodyPoints) > 0:
            x, y, w, h = cv2.boundingRect(np.array(lowerbodyPoints, np.float32))
            #cnt = np.array(upperbodyPoints, np.float32)
            #cnt = np.array(cnt).reshape((-1, 1, 2)).astype(np.int32)
            # cv2.drawContours(frameClone, cnt, 0, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = frameClone[y:y + h, x:x + w]
            color = colorAnalysis.main(roi)
            colorName2 = findColorValue(color)
            (label_width, label_height), baseline = cv2.getTextSize(colorName2, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(frameClone, (x, y - label_height - 10), (x+label_width+5, y), (255, 255, 255), -1)
            cv2.putText(frameClone, colorName2, (x, y - 5), FONT, 0.7, (255,0, 0), FONT_THICKNESS)

            #print(colorName2)


        colorData["person_"+str(idx)] = {"upper":colorName1, "lower" :colorName2 }

    # cv2.imshow("Detected Pose", frameClone)
    # cv2.waitKey(0)

    return frameClone, colorData

import urllib
import time


def extract_person_data():
    img = cv2.imread("img.jpg")
    #display(img)
    ret, colorData = analyseImage(img)
    print(colorData)
    display(ret)


if __name__ == "__main__":
    #main()
    #splitPersonWiseData()
    extract_person_data()