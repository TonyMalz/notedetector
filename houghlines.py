import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


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
    print('Found %s lines: ' % len(hlines))
    for line in hlines:
        for rho, theta in line:
            # convert from polar to cartesian
            y = np.sin(theta) * rho
            # print(y, theta, rho)
            lines.append(y)

    lines.sort()
    return lines

def find_lines(image):
    lines = []

    pi = round(np.pi, 4)
    acc_rho = 1  # pixel resolution
    acc_theta = pi / 180  # angle resolution in radians
    acc_threshold = 100  # min votes needed to get returned as a line (here min number of pixels)
    # limit theta to an angle of max 90 degree
    min_theta = 1.57
    max_theta = 1.58
    # print((max_theta - min_theta) / acc_theta)
    hlines = cv2.HoughLines(image, acc_rho, acc_theta, acc_threshold, 0, 0, 0, min_theta, max_theta)

    # line rotation angle (theta):
    # 0 equals vertical lines
    # pi/2 (1.5708) equals horizontal lines
    print('Found %s lines: ' % len(hlines))
    for line in hlines:
        for rho, theta in line:
            # convert from polar to cartesian
            x = np.cos(theta)
            y = np.sin(theta)
            x0 = x * rho
            y0 = y * rho
            # get x, y coordinates of 2 separate points (distance 2000 units/pixels) of this line
            # p1
            x1 = int(x0 + 1000 * (-y))
            y1 = int(y0 + 1000 * (x))
            # p2
            x2 = int(x0 - 1000 * (-y))
            y2 = int(y0 - 1000 * (x))
            lines.append([(x1,x2),(y1,y2),(x0,y0)])
            print(rho, theta, x, y, x0, y0, x1, y1, x2, y2)
    return lines

img = cv2.imread('entchen.png', cv2.IMREAD_GRAYSCALE)
# plt.imshow(img, cmap='gray')
# plt.show()
edges = cv2.Canny(img, 100, 200)
# plt.imshow(edges[45:200, 70:290], cmap='gray')
# plt.show()

# staffLines = find_lines(edges)
# plt.imshow(img, cmap='gray', interpolation='none')
# plt.axis([0, 500, img.shape[0], 0])
# if staffLines is not None:
#     for coords in staffLines:
#         print(coords)
#         plt.plot(coords[0], coords[1], 'r-', linewidth=.5)
# plt.show()

def find_staff_lines(lines):
    assert (len(lines) % 5 == 0), "Did not detect all staff lines"
    if lines is not None:
        staffLines = [[-1, -1, -1]]
        i = 0
        for yIntercept in lines:
            if staffLines[i][0] < 0:
                staffLines[i][0] = yIntercept
            elif staffLines[i][1] < 0:
                staffLines[i][1] = yIntercept
                # calc mean diff
                staffLines[i][2] = (staffLines[i][0] + staffLines[i][1]) / 2
                staffLines.append([-1, -1, -1])
                i += 1
    staffLines.pop()
    return staffLines

# show all horizontal lines
lines = find_horizontal_lines(edges)


def plot_edges(lines):
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.axis([0, 500, img.shape[0], 0])
    if lines is not None:
        for yIntercept in lines:
            plt.plot((0, 1000), (yIntercept, yIntercept), 'r-', linewidth=.5)
    plt.show()


# show all staff lines
staffLines = find_staff_lines(lines)

def plot_stafflines(staffLines):
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.axis([0, 500, img.shape[0], 0])
    if staffLines is not None:
        for line in staffLines:
            yIntercept = line[2] # mean y value
            plt.plot((0, 1000), (yIntercept, yIntercept), 'g-', linewidth=1)
    plt.show()


def get_avg_staff_height(stafflines):
    assert (len(lines) >= 5 ), "Did not detect all staff lines"
    staff_edge_top_y = staffLines[0][0]
    staff_edge_bottom_y = staffLines[4][1]
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

rowCoords = get_row_coords(staffLines)
print("Found %d rows" % len(rowCoords))





def delete_staffline(img,row):
    _, imgCols = img.shape
    mostFreqValue = Counter(img[row].flat).most_common(1)[0][0]
    if mostFreqValue < 10:
        return
    for j in range(0, imgCols):
        if img[row][j] == mostFreqValue:
            img[row][j] = 0


img = cv2.bitwise_not(img)

for i in range(0, len(staffLines)):
    staffTop = staffLines[i][0] #- rowCoords[0][0] # y of first row
    staffMiddle = staffLines[i][2]
    staffBottom = staffLines[i][1]

    delete_staffline(img, int(staffTop + .5))
    delete_staffline(img, int(staffMiddle + .5))
    delete_staffline(img, int(staffBottom + .5))

    plt.plot((0, 1000), (staffTop, staffTop), 'r--', linewidth=1)
    plt.plot((0, 1000), (staffMiddle, staffTop), 'g-', linewidth=1)
    plt.plot((0, 1000), (staffBottom, staffBottom), 'r--', linewidth=1)


plt.imshow(img, cmap='gray', interpolation='none')
plt.axis([0, img.shape[1], img.shape[0], 0])
plt.show()

def split_img_into_rows(img, rowCoords):
    rows = []
    for rowY in rowCoords:
        rows.append(img[int(rowY[0]):int(rowY[1])])
    return rows

rows = split_img_into_rows(img, rowCoords)
# for row in rows:
#     plt.imshow(row, cmap='gray', interpolation='none')
#     plt.show()

bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
plt.imshow(bw, cmap='gray', interpolation='none')
plt.axis([0, bw.shape[1], bw.shape[0], 0])
plt.show()

blur = cv2.GaussianBlur(bw,  (3, 3), 0)
plt.imshow(cv2.bitwise_not(blur), cmap='gray', interpolation='none')
plt.axis([0, blur.shape[1], blur.shape[0], 0])
plt.show()

# opening = np.copy(img)
# kernel = np.ones((5, 5), np.uint8)
# opening = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
# plt.imshow(opening, cmap='gray', interpolation='none')
# plt.axis([0, opening.shape[1], opening.shape[0], 0])
# plt.show()
#
vertical = np.copy(blur)
kernel = np.ones((3, 1), np.uint8)
vertical = cv2.erode(vertical, kernel)
vertical = cv2.dilate(vertical, kernel)
plt.imshow(cv2.bitwise_not(vertical), cmap='gray', interpolation='none')
plt.axis([0, vertical.shape[1], vertical.shape[0], 0])
plt.show()

_, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(cv2.bitwise_not(bw), cmap='gray', interpolation='none')
plt.axis([0, bw.shape[1], bw.shape[0], 0])
plt.show()


'''   sadfa
acc_rho = 1  # pixel resolution
acc_theta = pi / 180 # angle resolution in radians
acc_threshold = 35  # min votes needed to get returned as a line (here min number of pixels)
# limit theta to vertical lines
min_theta = 0
max_theta = 0.01
vlines = cv2.HoughLines(edges, acc_rho, acc_theta, acc_threshold, 0, 0, 0, min_theta, max_theta)

plt.imshow(img, cmap='gray')
plt.axis([0, 400, img.shape[0], 0])
# line rotation angle (theta):
# 0 equals vertical lines
# pi/2 (1.5708) equals horizontal lines
print('Found %s lines: ' % len(vlines))
for line in vlines:
    for rho, theta in line:
        # get x, y coordinates of 2 separate points (distance 2000 units/pixels) of this line
        x = np.cos(theta)
        y = np.sin(theta)
        x0 = x * rho
        y0 = y * rho
        # p1
        x1 = int(x0 + 1000 * (-y))
        y1 = int(y0 + 1000 * (x))
        # p2
        x2 = int(x0 - 1000 * (-y))
        y2 = int(y0 - 1000 * (x))

        plt.plot((x1, x2), (y1, y2), 'r-', linewidth=.5)
        # print (rho,theta,y0,y1,y2,x0,x1,x2)
plt.show()


bw = cv2.adaptiveThreshold(cv2.bitwise_not(img), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
plt.imshow(bw,cmap='gray')
plt.axis([0, 400, img.shape[0], 0])
plt.show()


horizontal = np.copy(bw)
ksize = int(horizontal.shape[1] / 30)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize,1))
kernel = np.ones((1,ksize),np.uint8)
horizontal = cv2.erode(horizontal,kernel)
horizontal = cv2.dilate(horizontal,kernel)
plt.imshow(horizontal,cmap='gray')
plt.axis([0, 400, horizontal.shape[0], 0])
plt.show()

vertical = np.copy(bw)
ksize = int(vertical.shape[0] / 30)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,ksize))
kernel = np.ones((3,1),np.uint8)
print (kernel)
vertical = cv2.erode(vertical,kernel)
vertical = cv2.dilate(vertical,kernel)
plt.imshow(vertical,cmap='gray')
plt.axis([0, 400, vertical.shape[0], 0])
plt.show()

vertical2 = cv2.bitwise_not(vertical)
plt.imshow(vertical2, cmap='gray')
plt.axis([0, 400, vertical.shape[0], 0])
for line in hlines:
    for rho, theta in line:
        # get x, y coordinates of 2 separate points (distance 2000 units/pixels) of this line
        x = np.cos(theta)
        y = np.sin(theta)
        x0 = x * rho
        y0 = y * rho
        # p1
        x1 = int(x0 + 1000 * (-y))
        y1 = int(y0 + 1000 * (x))
        # p2
        x2 = int(x0 - 1000 * (-y))
        y2 = int(y0 - 1000 * (x))

        plt.plot((x1, x2), (y1, y2), 'r-', linewidth=.1)
plt.show()
'''
