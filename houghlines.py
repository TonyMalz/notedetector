import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import degrees, atan2

def plot(img):
    rows, cols = img.shape
    plt.imshow(img, cmap='gray', interpolation='none')
    rows = min(450, rows)
    plt.axis([0, cols, rows, 0])
    plt.show()

def find_horizontal_lines(image):
    lines = []
    pi = round(np.pi, 4)
    acc_rho = 1  # pixel resolution
    acc_theta = pi / 180  # angle resolution in radians
    acc_threshold = 100  # min votes needed to get returned as a line (here min number of pixels)
    # limit theta to horizontal lines
    min_theta = 1.57
    max_theta = 1.58
    # print((max_theta - min_theta) / acc_theta)
    hlines = cv2.HoughLines(image, acc_rho, acc_theta, acc_threshold, 0, 0, 0, min_theta, max_theta)

    # line rotation angle (theta):
    # 0 equals vertical lines
    # pi/2 (1.5708) equals horizontal lines
    print('Found %s horizontal lines: ' % len(hlines))
    for line in hlines:
        for rho, theta in line:
            # convert from polar to cartesian
            y = np.sin(theta) * rho
            # print(y, theta, rho)
            lines.append(y)

    lines.sort()
    return lines

def find_staff_lines(img, lines):
    staffLines = []
    for line in lines:
        row = int(line + .5)
        if img[row].max() < 10:
            continue

        staffLines.append(row)

    # assert (len(lines) % 5 == 0), "Did not detect all staff lines"
    # if lines is not None:
    #     staffLines = [[-1, -1, -1]]
    #     i = 0
    #     for yIntercept in lines:
    #
    #         if staffLines[i][0] < 0:
    #             staffLines[i][0] = yIntercept
    #         elif staffLines[i][1] < 0:
    #             staffLines[i][1] = yIntercept
    #             # calc mean diff
    #             staffLines[i][2] = (staffLines[i][0] + staffLines[i][1]) / 2
    #             staffLines.append([-1, -1, -1])
    #             i += 1
    # staffLines.pop()
    print("Found %s staff lines" % len(staffLines))
    return staffLines

def plot_edges(img, lines):
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.axis([0, 500, img.shape[0], 0])
    if lines is not None:
        for yIntercept in lines:
            plt.plot((0, 1000), (yIntercept, yIntercept), 'r-', linewidth=.5)
    plt.show()

def plot_stafflines(img, staffLines):
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.axis([0, 500, img.shape[0], 0])
    if staffLines is not None:
        for line in staffLines:
            yIntercept = line[2] # mean y value
            plt.plot((0, 1000), (yIntercept, yIntercept), 'g-', linewidth=1)
    plt.show()

def get_avg_staff_height(stafflines):
    assert (len(stafflines) >= 5 ), "Did not detect all staff lines"
    staff_edge_top_y = stafflines[0][0]
    staff_edge_bottom_y = stafflines[4][1]
    return staff_edge_bottom_y - staff_edge_top_y

def get_row_coords(staffLines):
    assert (len(staffLines) % 5 == 0), "Did not detect all staff lines"
    staffHeight = get_avg_staff_height(staffLines)

    # isolate staff rows
    # first row starts at half the distance off of the first/top staff line edge
    deltaLineMargin = staffHeight
    row_y_start = staffLines[0][0] - deltaLineMargin
    row_y_end = staffLines[4][1] + deltaLineMargin

    # contains all rows with corresponding start and end y-coords
    rows = [[row_y_start, row_y_end]]

    numrows = int(len(staffLines)/5)
    # iterate over all remaining lines
    for i in range(1, numrows):
        row_y_start = staffLines[5*i][0] - deltaLineMargin
        row_y_end = staffLines[(5*i)+4][1] + deltaLineMargin
        rows.append([row_y_start, row_y_end])

    return rows

def delete_staffline(img,row):
    _, imgCols = img.shape
    mostFreqValue = Counter(img[row].flat).most_common(1)[0][0]
    for j in range(0, imgCols):
        if img[row][j] == mostFreqValue:
            img[row][j] = 0

def split_img_into_rows(img, rowCoords):
    rows = []
    for rowY in rowCoords:
        rows.append(img[int(rowY[0]):int(rowY[1])])
    return rows

def find_vertical_lines(image, staff_height):
    lines = []
    pi = round(np.pi, 4)
    acc_rho = 1  # pixel resolution
    acc_theta = pi / 180  # angle resolution in radians
    acc_threshold = int(staff_height * 0.6)  # min votes needed to get returned as a line (here min number of pixels)
    # limit theta to vertical lines
    min_theta = 0.
    max_theta = 0.01
    # print((max_theta - min_theta) / acc_theta)
    hlines = cv2.HoughLines(image, acc_rho, acc_theta, acc_threshold, 0, 0, 0, min_theta, max_theta)

    # line rotation angle (theta):
    # 0 equals vertical lines
    # pi/2 (1.5708) equals horizontal lines
    print('Found %s vertical lines: ' % len(hlines))
    for line in hlines:
        for rho, theta in line:
            # convert from polar to cartesian
            x = np.cos(theta) * rho
            # print(y, theta, rho)
            lines.append(x)

    lines.sort()
    return lines

def get_img_rotation(image):
    img = np.copy(image)
    rows, cols = img.shape
    img = cv2.bitwise_not(img)
    pi = round(np.pi, 4)
    acc_rho = 1  # pixel resolution
    acc_theta = pi / 180  # angle resolution in radians
    acc_threshold = 100  # min votes needed to get returned as a line (here min number of pixels)
    hlines = cv2.HoughLinesP(img, acc_rho, acc_theta, acc_threshold, minLineLength=cols/2, maxLineGap=20)
    angle = 0
    i = 0
    line_angles = []
    for line in hlines:
        for x1, y1, x2, y2 in line:
            # plt.plot((x1, x2), (y1, y2), 'r-', linewidth=.5)
            ang = atan2(y2 - y1, x2 - x1)
            angle += ang
            line_angles.append(ang)
            i += 1
    # plot(image)
    angle = angle / i
    angle = degrees(angle)
    # angle2 = degrees(Counter(line_angles).most_common(1)[0][0])

    return angle - 0.2  # account for float rounding errors

def rotate_img(img, angle, cols, rows):
    if -0.5 < angle < 0.5:
        return img
    rot_center = (cols / 2, rows / 2)
    rot_degrees = angle  # counter clockwise
    rot_scaling = 1
    M = cv2.getRotationMatrix2D(rot_center, rot_degrees, rot_scaling)
    img = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return img

# test img transforms

def extract_horizontal_lines(img):
    horizontal = cv2.bitwise_not(img)
    ksize = int(horizontal.shape[1] / 30)
    kernel = np.ones((1, ksize), np.uint8)
    horizontal = cv2.erode(horizontal, kernel)
    horizontal = cv2.dilate(horizontal, kernel)
    return horizontal

def internal_transform_staff_lines(horizontal_lines):
    # FIXME
    ll = []
    for line in horizontal_lines:
        ll.append([line, line, line])
    return ll

def delete_all_stafflines(img, staffLines):
    img = cv2.bitwise_not(img)
    for i in range(0, len(staffLines)):
        staffTop = staffLines[i][0]  # - rowCoords[0][0] # y of first row
        staffMiddle = staffLines[i][2]
        staffBottom = staffLines[i][1]

        delete_staffline(img, int(staffTop + .5))
        delete_staffline(img, int(staffMiddle + .5))
        delete_staffline(img, int(staffBottom + .5))
    plot(img)
    return img

def delete_horizontal_lines(img):
    bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    blur = cv2.GaussianBlur(bw, (3, 3), 0)
    # opening = np.copy(img)
    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
    # TODO erode relative to avg staff line height
    vertical = np.copy(blur)
    kernel = np.ones((3, 1), np.uint8)
    vertical = cv2.erode(vertical, kernel)
    kernel = np.ones((3, 2), np.uint8)
    vertical = cv2.dilate(vertical, kernel)
    return vertical

def get_row_info(img, rowCoords, staffLines):
    row_info = [[[], [], []]]  # boundary, stafflines, bars
    vertical = delete_horizontal_lines(cv2.bitwise_not(img))
    # _, bw = cv2.threshold(vertical, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print('Avg staff height: ', get_avg_staff_height(staffLines))
    rows = split_img_into_rows(vertical, rowCoords)
    for i in range(len(rows)):
        edges = cv2.Canny(cv2.bitwise_not(rows[i]), 100, 200)
        vlines = find_vertical_lines(edges, get_avg_staff_height(staffLines))

        rowStaffLines = []
        # plot stafflines
        rowTopY = rowCoords[i][0]
        rowBottomY = rowCoords[i][1]
        row_info[i][0].append(int(rowTopY + .5))
        row_info[i][0].append(int(rowBottomY + .5))

        for line in staffLines:
            staffYAvg = line[2]
            if rowTopY < staffYAvg < rowBottomY:
                row_info[i][1].append(int(staffYAvg + .5))
                staffY = int(staffYAvg - rowTopY + .5)

                rowStaffLines.append([x-rowTopY for x in line])
                #plt.plot((0, 1500), (staffY, staffY), 'g--')

        # find bars
        bars = []
        for xVal in vlines:
            # look at the second vertical line of pixels
            # since dilate should at least provide two lines of pixels for each vertical line
            col = int(xVal + 2.5)
            staffTop = int(rowStaffLines[0][1] + 2)
            staffBottom = int(rowStaffLines[4][0] - 2)
            stop = False
            row = rows[i]

            for currentRow in range(staffTop, staffBottom + 1):
                val = row[currentRow][col]
                # TODO cut off value
                if val < 90:
                    stop = True
                if currentRow == staffBottom:
                    # done with descending since we reached bottom staff line
                    if stop == False:
                        # we found a bar since we didn't stop until the end
                        bars.append(col - 1)  # shift one pixel back since it was the origin of the bar

        print("found bar x coords: ", bars)
        row_info[i][2] = bars
        # for bar in bars:
        #     plt.plot((bar, bar), (0, 1000), 'r-')
        # plot(rows[i])
        row_info.append([[], [], []])
    row_info.pop()
    return row_info


#
# img = cv2.imread('data/Testdata/Entchen_rotated_big.jpg', cv2.IMREAD_GRAYSCALE)
#
# rot_angle = get_img_rotation(img)  # counter clockwise
# img = rotate_img(img, rot_angle)
# # img = cv2.imread('data/Testdata/Entchen.jpg', cv2.IMREAD_GRAYSCALE)
#
# himg = extract_horizontal_lines(img)
# staffLines = find_horizontal_lines(himg)
# # for yIntercept in lines:
# #     yIntercept = int(yIntercept + .5)
# #     plt.plot((0, 1000), (yIntercept, yIntercept), 'r-', linewidth=1)
# # plot(horizontal)
#
#
# # staffLines = find_staff_lines(lines)
#
# staffLines = internal_transform_staff_lines(staffLines)
# rowCoords = get_row_coords(staffLines)
# print("Found %d rows" % len(rowCoords))
# # img = delete_all_stafflines(img, staffLines)
#
# #     plt.plot((0, 1000), (staffTop, staffTop), 'r--', linewidth=1)
# #     plt.plot((0, 1000), (staffMiddle, staffTop), 'g-', linewidth=1)
# #     plt.plot((0, 1000), (staffBottom, staffBottom), 'r--', linewidth=1)
# #
# #
# info = get_row_info(img, rowCoords)
# print(info)
#

# img = cv2.bitwise_not(img)
# cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# circles = cv2.HoughCircles(cv2.bitwise_not(blur), cv2.HOUGH_GRADIENT, 1, 10, param1=350, param2=10, minRadius=0, maxRadius=10)
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(cimg,(i[0],i[1]),1,(0,0,255),3)
#
# cv2.imshow('detected circles', cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



