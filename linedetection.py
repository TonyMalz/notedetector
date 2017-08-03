""" @author: Tony """
import cv2
import numpy as np
from collections import Counter
from math import degrees, atan2, sqrt, fabs

""" primitive interface, used as an interface for making line primitives available to the inference module """
class Primitive:
    def __init__(self, type, detectedBy, width, height, xCoordinate, yCoordinate, toleranceValue, certainty):
        self.type = type
        self.detectedBy = detectedBy

        self.width = width
        self.height = height

        self.xCoordinate = xCoordinate
        self.yCoordinate = yCoordinate

        self.toleranceValue = toleranceValue
        self.certainty = certainty

        self.colors = []
        self.hypotheses = []

""" Line class primitive, used by Stave class """
class Line:
    def __init__(self):
        self.y = -1
        self.x = -1
        self.y2 = -1
        self.x2 = -1

        self.theta = -1
        self.rho = -1
        self.image = None
        self.rotation = -1
        self.length = -1
        self.confidence = 0.

    def getY(self):
        return self.y

    def getYInt(self):
        return int(round(self.y))

    def getX(self):
        return self.x

    def getXInt(self):
        return int(round(self.x))

    def setY(self, y):
        self.y = y

    def setX(self, x):
        self.x = x

    def getY2(self):
        self._calcLineCoords()
        return self.y2

    def getY2Int(self):
        self._calcLineCoords()
        return int(round(self.y2))

    def getX2(self):
        self._calcLineCoords()
        return self.x2

    def getX2Int(self):
        self._calcLineCoords()
        return int(round(self.x2))

    def setY2(self, y):
        self.y2 = y

    def setX2(self, x):
        self.x2 = x

    def getTheta(self):
        return self.theta

    def getRho(self):
        return self.rho

    def setTheta(self, theta):
        self.theta = theta

    def setRho(self, rho):
        self.rho = rho

    def _calcLineCoords(self):
        if self.x2 >= 0 or self.y2 >= 0:
            return
        a = np.cos(self.theta)
        b = np.sin(self.theta)
        x0 = a * self.rho
        y0 = b * self.rho

        offset = 1000
        if self.image is not None:
            offset = max(self.image.shape[0],self.image.shape[1])
        self.x2 = x0 - offset * (-b)
        self.y2 = y0 - offset * a

    def getSlope(self):
        if self.x2 >= 0 and self.y2 >= 0:
            if self.x2 == self.x: # vertical line
                return -111
            if self.y2 == self.y: # horizontal line
                return 0
            return float(self.y2 - self.y) / (self.x2 - self.x)
        return -1

    def getRotation(self):
        if self.rotation == -1:
            self.rotation = degrees(atan2(self.getY2() - self.getY(), self.getX2() - self.getX()))
        return self.rotation

    def getLength(self):
        if self.length == -1:
            self.calcLength()
        return self.length

    def calcLength(self):
        x = self.getX2() - self.getX()
        y = self.getY2() - self.getY()
        self.length = fabs(sqrt(x*x + y*y))

    def setImage(self, image):
        self.image = image

    def getConfidence(self ):
        return self.confidence

    def __str__(self):
        return 'Line x: %.2f y: %.2f conf: %.2f x2: %.2f y2: %.2f rotation: %.2f length: %.2f slope: %.2f theta: %.2f rho: %.2f' % (self.x, self.y,self.getConfidence(), self.getX2(), self.getY2(), self.getRotation(),self.getLength(),self.getSlope(), self.theta, self.rho)

    def setConfidence(self, conf):
        self.confidence = conf

    def setRotation(self, rotation):
        self.rotation = rotation

""" Stave representation, containing constituent staff lines objects and methods for querying important metrics """
class Stave:
    def __init__(self):
        self.x = -1
        self.y = -1
        self.width = -1
        self.height = -1
        self.upperBound = -1
        self.lowerBound = -1
        self.leftBound = -1
        self.rightBound = -1

        self.lineCount = 0
        self.lines = []
        self.image = None
        self.staffLines = [(0,0),(0,0),(0,0),(0,0),(0,0)]
        self.confidence = -1

        self.avgStaffLineHeight = -1
        self.avgStaffSpaceHeight = -1

        # primitives
        self.bars = []
        self.verticalLines = []
        self.verticalLineFragments = []

    def __str__(self):
        return 'top: %.2f lines: %d ' % (self.y, self.lineCount)

    def setAvgStaffLineHeight(self, height):
        self.avgStaffLineHeight = height

    def setAvgStaffSpaceHeight(self, height):
        self.avgStaffSpaceHeight = height

    def getBarsAsPrimitives(self):
        primitives = []
        self.bars.sort(key=lambda line: line.getX())
        for line in self.bars:
            boundingBoxWidth = abs(line.getX() - line.getX2())
            boundingBoxHeight = abs(line.getY() - line.getY2())
            boundingBoxCenterX = (line.getX() + line.getX2()) / 2
            boundingBoxCenterY = (line.getY() + line.getY2()) / 2
            # get absolute coordinate
            # since y coordinates are relative to current staff
            boundingBoxCenterY += self.getUpperBound()
            # there is no confidence for a single line found by hough lines probabilistic yet
            confidence = 0.5
            prim = Primitive(type='rectangle',detectedBy='houghlines', width=boundingBoxWidth, height=boundingBoxHeight,xCoordinate=boundingBoxCenterX,yCoordinate=boundingBoxCenterY,toleranceValue=1,certainty=confidence )
            primitives.append(prim)
        return primitives

    def getVerticalsAsPrimitives(self):
        # 1. detect vertical bar x coordinates
        startX = -1
        verticalBars = []
        lineDistanceX = 0
        prevX = -1
        l = Line()
        self.verticalLines.sort(key=lambda line:line.getX())
        topY = self.getTopY() - self.getAvgStaffLineIntervall() * 2
        if topY < 0:
            topY = 0
        bottomY = self.getBottomY() + self.getAvgStaffLineIntervall() * 2
        if bottomY > self.image.shape[0]:
            bottomY = self.image.shape[0]

        for line in self.verticalLines:
            if startX == -1:
                startX = line.getX()
                prevX = startX
                l.setX(startX)
                # hough lines have no y coordinates attached to them which can be used as a bounding box
                # set top and bottom y of the current stave instead
                l.setY(topY)
                l.setY2(bottomY)
                l.setX2(0)
                continue

            currentX = line.getX()
            lineDistanceX = currentX - startX
            # group all lines together not exceeding twice the size of a typical staff line in horizontal direction
            if lineDistanceX <= self.avgStaffSpaceHeight:
                prevX = currentX
                continue
            # current line is possible new vertical bar
            # save previous x coordinate which determines the with of the previous bar
            l.setX2(prevX)
            verticalBars.append(l)
            # start new bar with current x coordinate
            l = Line()
            startX = currentX
            prevX = startX
            l.setX(startX)
            l.setY(topY)
            l.setY2(bottomY)
            l.setX2(0)

        # add last bar if it has not been closed and added before
        if l.getX2() == 0:
            l.setX2(l.getX())
            verticalBars.append(l)

        # adjust size for bars where only one edge was found
        shiftSize = self.avgStaffSpaceHeight / 2
        for line in verticalBars:
            if line.getX() == line.getX2():
                line.setX(line.getX()-shiftSize)
                line.setX2(line.getX2()+shiftSize)
                line.setConfidence(0.3)
            else:
                line.setConfidence(0.6)

        # 2. get additional info from line fragments regarding x and y coordinate confidences
        for line in verticalBars:
            fragmentsInRange = self.getVerticalLineFragmentsInRange(line.getX(), line.getX2())
            if len(fragmentsInRange) > 0:
                line.setConfidence(line.getConfidence() + 0.3)

        # 3. convert bars to primitive objects
        primitives = []
        for line in verticalBars:
            boundingBoxWidth = abs(line.getX() - line.getX2())
            boundingBoxHeight = abs(line.getY() - line.getY2())
            boundingBoxCenterX = (line.getX() + line.getX2()) / 2
            boundingBoxCenterY = (line.getY() + line.getY2()) / 2
            confidence = line.getConfidence()
            prim = Primitive(type='verticalLine', detectedBy='houghlines', width=boundingBoxWidth,
                                height=boundingBoxHeight, xCoordinate=boundingBoxCenterX,
                                yCoordinate=boundingBoxCenterY, toleranceValue=1, certainty=confidence)
            primitives.append(prim)

        return primitives

    def getVerticalLineFragmentsInRange(self,x1,x2):
        fragments = []
        for line in self.verticalLineFragments:
            if line.getX() >= x1 and line.getX2() <= x2:
                fragments.append(line)
        return fragments

    def getBars(self):
        return self.bars
    def setBars(self, bars):
        self.bars = bars
    def addBar(self, bar):
        self.bars.append(bar)

    def getVerticalLines(self):
        return self.verticalLines
    def setVerticalLines(self, verticalLines):
        self.verticalLines = verticalLines
    def addVerticalLine(self, line):
        self.verticalLines.append(line)

    def getVerticalLineFragments(self):
        return self.verticalLineFragments
    def setVerticalLineFragments(self, verticalLineFragments):
        self.verticalLineFragments = verticalLineFragments
    def addVerticalLineFragment(self, lineFragment):
        self.verticalLineFragments.append(lineFragment)


    def getConfidence(self):
        if self.confidence == -1:
            # get avg confidence from staff lines
            confidence = 0.
            for _, conf in self.staffLines:
                confidence += conf
            self.confidence = float(confidence) / len(self.staffLines)
        return self.confidence

    def addStaffLineProspect(self, y, confidence=0.):
        count = self.lineCount + 1
        if count <= 15:
            self.lines.append((y, confidence))
            self.lineCount = count
        else:
            print 'Too many staff lines in stave'

    def setStaffLine(self, lineNumber, y, confidence=0.):
        if 1 <= lineNumber <= 5:
            self.staffLines[lineNumber - 1] = (y, confidence)
        else:
            print 'No valid staff line number'

    def getAvgStaffLineIntervall(self):
        dY = 0
        for i in range(1,5):
            y1, _ = self.staffLines[i-1]
            y2, _ = self.staffLines[i]
            dY += y2 - y1
        return dY / 4.

    def setUpperBound(self, y):
        self.upperBound = y

    def setLowerBound(self, y):
        self.lowerBound = y

    def setLeftBound(self, x):
        self.leftBound = x

    def setRightBound(self, x):
        self.rightBound = x

    def getRightBound(self):
        return self.rightBound

    def getLeftBound(self):
        return self.leftBound

    def getUpperBound(self):
        if self.upperBound < 0:
            self.upperBound = self.y - self.getHeight()
            if self.upperBound < 0:
                self.upperBound = 0
        return self.upperBound

    def getLowerBound(self):
        if self.lowerBound == -1 or self.image is not None and self.lowerBound > self.image.shape[0]:
            self.lowerBound = self.getBottomY() + self.getHeight()
            if self.image is not None and self.lowerBound > self.image.shape[0]:
                self.lowerBound = self.image.shape[0]

        return self.lowerBound

    def setTopY(self, y):
        self.y = y

    def getTopY(self):
        return self.y

    def getBottomY(self):
        return self.y + self.getHeight()

    def getRowImage(self):
        if self.image is None:
            return None
        return self.image[self.getUpperBound():self.getLowerBound()]

    def getHeight(self):
        if self.height == -1:
            #get bottom staff line y coordinate
            y, _ = self.staffLines[4]
            if y > 0:
                self.height = y + 1 - self.y
        return self.height

    def getWidth(self):
        if self.width == -1:
            if self.image is not None:
                self.width = self.image.shape[1]
        return self.width

    def getStaffLine(self, lineNumber):
        if 1 <= lineNumber <= 5:
            y, conf = self.staffLines[lineNumber - 1]
            line = Line()
            line.setX(0)
            line.setY(y)
            line.setY2(y)
            line.setX2(self.getWidth())
            line.setImage(self.image)
            line.setConfidence(conf)
            line.setRotation(0) #horizontal
            return line
        else:
            print 'No valid staff line number'
            return None

    def getStaffLines(self):
        lines = []
        for y, conf in self.staffLines:
            line = Line()
            line.setX(0)
            line.setY(y)
            line.setY2(y)
            line.setX2(self.getWidth())
            line.setImage(self.image)
            line.setConfidence(conf)
            line.setRotation(0)  # horizontal
            lines.append(line)
        return lines

    def setImage(self, image):
        if image is None or not isinstance(image, np.ndarray):
            print 'No valid image'
            return
        self.image = image

""" Given an image, this component finds all lines within it using Hough transform analysis """
class LineDetector:
    def __init__(self):
        self.image = None
        self.imagePath = None
        self.imageGray = None
        self.imageEdges = None
        self.imageBinary = None
        self.imageWidth = -1
        self.imageHeight = -1
        self.minLineLength = -1

        self.lines = []
        self.verticalLines = []
        self.horizontalLines = []
        self.lineFragments = []

        self.enableLineFragments = True

    def getImageWidth(self):
        return self.imageWidth

    def getImageHeight(self):
        return self.imageHeight

    def setImagePath(self, path):
        self.imagePath = path
        self.image = cv2.imread(path)
        if self.image is None:
            print 'No valid image path'
        else:
            self._setImgDim()

    def setMinLineLength(self, lineLength):
        self.minLineLength = int(round(lineLength))

    def getMinLineLength(self):
        return self.minLineLength

    def _setImgDim(self):
        self.imageWidth = self.image.shape[1]
        self.imageHeight = self.image.shape[0]

    def setImage(self, image):
        if image is None or not isinstance(image, np.ndarray):
            print 'No valid image'
            return
        self.image = image
        self._setImgDim()

    def getImage(self):
        return self.image

    def getGrayImage(self):
        return self.imageGray

    def getEdgeImage(self):
        return self.imageEdges

    def getBinaryImage(self):
        return self.imageBinary

    def getImagePath(self):
        if self.imagePath is None:
            print 'No image path specified!'
        return self.imagePath

    @staticmethod
    def convertPolar2Cartesian(rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    def getLines(self):
        return self.lines

    def getHorizontalLines(self):
        return self.horizontalLines

    def getVerticalLines(self):
        return self.verticalLines

    def getLineFragments(self):
        return self.lineFragments

    def disableLineFragments(self):
        self.enableLineFragments = False

    def run(self):
        if self.image is None:
            print ('No image')
            return False

        #pre process image: generate gray, edge and binary versions
        self._preProcessImg()

        # find all lines by the general hough transform algorithm
        self._findLinesByHoughtransform()

        if self.enableLineFragments:
            # additionally find all line fragments using the probabilistic version of hough transforms
            self._findLinesByHoughtransformProbabilistic()

        # TODO use vertical and horizontal run length analysis to get additional line information
        # self._findLinesByRLE()

        return True

    def _preProcessImg(self):
        self.imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.imageEdges = cv2.Canny(self.imageGray, 150, 240, apertureSize=3)
        self.imageBinary = cv2.adaptiveThreshold(self.imageGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,41,10) #ex4 19,9 1216 x 1696  ex3  41,10 2369 x 3327
        #self.imageBinary = cv2.GaussianBlur(self.imageBinary, (3, 3), 0)
        #_, self.imageBinary = cv2.threshold(self.imageBinary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #self.imageBinary = self.isolateHorizontalLines(self.imageBinary)

    """ remove everything but horizontal lines """
    @staticmethod
    def isolateHorizontalLines(img):
        if img is None or not isinstance(img, np.ndarray):
            print 'No valid image'
            return None
        horizontal = cv2.bitwise_not(img)
        kernelsize = int(horizontal.shape[1] / 30)
        kernel = np.ones((1, kernelsize), np.uint8)
        horizontal = cv2.erode(horizontal, kernel)
        #horizontal = cv2.dilate(horizontal, kernel)
        return cv2.bitwise_not(horizontal)

    """ remove everything but vertical lines """
    @staticmethod
    def isolateVerticalLines(img, stafflineHeight=2):
        if img is None or not isinstance(img, np.ndarray):
            print 'No valid image'
            return None
        vertical = cv2.bitwise_not(img)
        kernel = np.ones((stafflineHeight * 2, 1), np.uint8)
        vertical = cv2.erode(vertical, kernel)
        kernel = np.ones((stafflineHeight * 1, 1), np.uint8)
        #vertical = cv2.dilate(vertical , kernel)
        return cv2.bitwise_not(vertical)

    @staticmethod
    def stripStaffLines(img, stafflineHeight=2):
        return LineDetector.isolateVerticalLines(img,stafflineHeight)

    def _findLinesByHoughtransformProbabilistic(self):
        pi = round(np.pi, 4)
        acc_rho = 1  # pixel resolution
        acc_theta = pi / 180  # angle resolution in radians
        if self.minLineLength > 0:
            acc_threshold = self.minLineLength  # min votes needed to get returned as a line (here min number of pixels)
        else:
            acc_threshold = self.imageWidth / 4
        minLineLength = None # has no effect at all!
        maxLineGap = 4
        lines = cv2.HoughLinesP(cv2.bitwise_not(self.imageBinary), acc_rho, acc_theta, acc_threshold, minLineLength, maxLineGap)
        #lines = cv2.HoughLinesP(self.imageEdges, acc_rho, acc_theta, acc_threshold, minLineLength, maxLineGap)
        if lines is None:
            return
        for lineCoords in lines:
            for x1, y1, x2, y2 in lineCoords:
                l = Line()
                l.setX(x1)
                l.setY(y1)
                l.setX2(x2)
                l.setY2(y2)
                l.setImage(self.image)
                self.lineFragments.append(l)

    @staticmethod
    def findLinesByHoughtransformProbabilistic(img, minLineLength=0):
        if not isinstance(img,np.ndarray):
            print 'invalid image'
            return []

        imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imageBinary = cv2.adaptiveThreshold(imageGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                 cv2.THRESH_BINARY, 41, 10)
        imageWidth = img.shape[1]
        pi = round(np.pi, 4)
        acc_rho = 1  # pixel resolution
        acc_theta = pi / 180  # angle resolution in radians
        if minLineLength > 0:
            acc_threshold = minLineLength  # min votes needed to get returned as a line (here min number of pixels)
        else:
            acc_threshold = imageWidth / 4
        minLineLength = None # has no effect at all!
        maxLineGap = 4
        lines = cv2.HoughLinesP(cv2.bitwise_not(imageBinary), acc_rho, acc_theta, acc_threshold, minLineLength, maxLineGap)
        if lines is None:
            return []

        lineFragments = []
        for lineCoords in lines:
            for x1, y1, x2, y2 in lineCoords:
                l = Line()
                l.setX(x1)
                l.setY(y1)
                l.setX2(x2)
                l.setY2(y2)
                l.setImage(img)
                lineFragments.append(l)

        return lineFragments

    @staticmethod
    def findLinesByHoughtransform(img, minLineLength=0):
        if not isinstance(img, np.ndarray):
            print 'invalid image'
            return []
        minLineLength = int(minLineLength)

        imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imageEdges = cv2.Canny(imageGray, 50, 240, apertureSize=3)
        imageBinary = cv2.adaptiveThreshold(imageGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                            cv2.THRESH_BINARY, 41, 10)
        imageWidth = img.shape[1]

        pi = round(np.pi, 4)
        acc_rho = 1  # pixel resolution
        acc_theta = pi / 180  # angle resolution in radians
        if minLineLength > 0:
            acc_threshold = minLineLength  # min votes needed to get returned as a line (here min number of pixels)
        else:
            acc_threshold = imageWidth / 4

        # limit theta
        # 0 equals vertical lines
        # pi/2 (1.5708) equals horizontal lines
        min_theta = 0
        max_theta = pi
        # print((max_theta - min_theta) / acc_theta)
        detectedLines = cv2.HoughLines(imageEdges, acc_rho, acc_theta, acc_threshold)
        if detectedLines is None:
            return []

        lines = []
        for line in detectedLines:
            for rho, theta in line:
                l = Line()
                l.setRho(rho)
                l.setTheta(theta)
                x , y = LineDetector.convertPolar2Cartesian(rho, theta)
                l.setX(x)
                l.setY(y)
                l.setImage(img)
                lines.append(l)
        return lines

    def _findLinesByHoughtransform(self):
        pi = round(np.pi, 4)
        acc_rho = 1  # pixel resolution
        acc_theta = pi / 180  # angle resolution in radians
        if self.minLineLength > 0:
            acc_threshold = self.minLineLength  # min votes needed to get returned as a line (here min number of pixels)
        else:
            acc_threshold = self.imageWidth / 4

        # limit theta
        # 0 equals vertical lines
        # pi/2 (1.5708) equals horizontal lines
        min_theta = 0
        max_theta = pi
        # print((max_theta - min_theta) / acc_theta)
        detectedLines = cv2.HoughLines(self.imageEdges, acc_rho, acc_theta, acc_threshold)
        if detectedLines is None:
            return
        for line in detectedLines:
            for rho, theta in line:
                l = Line()
                l.setRho(rho)
                l.setTheta(theta)
                x , y = self.convertPolar2Cartesian(rho, theta)
                l.setX(x)
                l.setY(y)
                l.setImage(self.image)
                self.lines.append(l)

                if 1.57 <= theta <= 1.58:
                    self.horizontalLines.append(l)

                if 0.0 <= theta <= 0.01:
                    self.verticalLines.append(l)

    def getDebugImage(self):
        img = self.getImage()
        print 'Image size: %d x %d ' % (self.getImageWidth(), self.getImageHeight())
        print 'Detected the following lines in image %s' % self.getImagePath()

        print '--- Horizontal Lines : -----'
        for line in self.getHorizontalLines():
            cv2.line(img, (line.getXInt(), line.getYInt()), (line.getX2Int(), line.getY2Int()), (0, 255, 0), 1)
        print 'found %d horizontal lines' % len(self.getHorizontalLines())

        print '--- Vertical Lines : -----'
        for line in self.getVerticalLines():
            # print line
            # cv2.line(img, (line.getXInt(), line.getYInt()), (line.getX2Int(), line.getY2Int()), (0, 255, 0), 1)
            pass
        print 'found %d vertical lines' % len(self.getVerticalLines())

        print '--- Horizontal Line Fragments : -----'
        for line in self.getLineFragments():
            if -1 <= line.getRotation() <= 1:
                cv2.line(img, (line.getXInt(), line.getYInt()), (line.getX2Int(), line.getY2Int()), (200, 0, 200), 1)
                pass
        print 'found %d line fragments' % len(self.getLineFragments())
        return img

""" Finds all staffs within the given image, uses a LineDetector (if not already provided) """
class StaffDetector:
    def __init__(self):
        self.rotateImage = True
        self.staffs = []
        self.avgStaffHeight = -1
        self.avgStaffLineHeight = -1
        self.avgStaffSpaceHeight = -1
        self.lineDetector = None
        self.image = None
        self.staffNotFoundConfidence = 0.1
        self.lineFragmentsDict = None # [y] = (x,y)

        self.top = 0
        self.left = 0
        self.bottom = 0
        self.right = 0

    def getAvgStaffHeight(self):
        return self.avgStaffHeight

    def getAvgStaffLineHeight(self):
        return self.avgStaffLineHeight

    def getAvgStaffSpaceHeight(self):
        return self.avgStaffSpaceHeight

    def getStaffs(self):
        return self.staffs

    def setLineDetector(self, detector):
        if detector is None or not isinstance(detector, LineDetector):
            print 'invalid line detector'
            return
        self.lineDetector = detector

    def getLineDetector(self):
        if self.lineDetector is None:
            self.lineDetector = LineDetector()
            self.lineDetector.setImage(self.getImage())

        return self.lineDetector

    def setImage(self, image):
        if image is None or not isinstance(image, np.ndarray):
            print 'No valid image'
            return
        if self.rotateImage:
            image = get_rotated_image(image)
            self.rotateImage = False
        self.image = image
        self.bottom = image.shape[0]
        self.right = image.shape[1]

    def getImage(self):
        return self.image

    def getImageForGui(self, cropped=False):
        # TODO adjust for cropped coordinates
        return get_enhanced_image_for_gui(self, cropImage=cropped)

    def getImageInternal(self, cropped=False):
        # TODO adjust for cropped coordinates
        return get_gray_image(self, cropImage=cropped)

    @staticmethod
    def calcStaffThickness(detectorObj):
        fragmentY = []
        for fragLine in detectorObj.getLineFragments():
            # select horizontal lines
            if -0.01 <= fragLine.getRotation() <= 0.01:
                fragmentY.append(fragLine.getY())

        fragmentY.sort()
        # print fragmentY
        if len(fragmentY) < 1:
            return

        prevY = 0
        staffProspects = []
        staffThickness = []
        thickness = 0
        # calc distance consecutive runs of lines
        for y in fragmentY:
            if prevY == 0:
                staffProspects.append(y)
                thickness = 1
                prevY = y
                continue
            dY = y - prevY
            prevY = y
            # consecutive line
            if dY <= 1:
                thickness += dY
                continue
            staffProspects.append(y)
            staffThickness.append(thickness)
            thickness = 1

        staffThickness = Counter(staffThickness)
        staffLine_height, _ = staffThickness.most_common(1)[0]

        staffSpaces = []
        staffCount = len(staffProspects)
        for i in range(0, staffCount):
            if i + 1 <= staffCount-1:
                staffSpaces.append(staffProspects[i + 1] - staffProspects[i])

        staffSpaces = Counter(staffSpaces)
        staffSpace_height, _ = staffSpaces.most_common(1)[0]
        staffSpace_height -= staffLine_height / 2
        print 'Average staff line height: %d' % staffLine_height
        print 'Average staff space height: %d' % staffSpace_height
        return staffLine_height, staffSpace_height

    @staticmethod
    def calcConfidence(topEdgePresent, bottomEdgePresent, fraglinePresent):
        conf = 0.
        if topEdgePresent:
            conf += 0.2
        if bottomEdgePresent:
            conf += 0.2
        if fraglinePresent:
            conf += 0.6
        return conf

    @staticmethod
    def isFragmentInRange(fragments, start, stop):
        if start < 0:
            return False
        if not isinstance(fragments, dict):
            print 'fragments is not a dictionary'
            return False

        if stop < start:
            temp = start
            start = stop
            stop = start

        found = False
        # search inclusive
        for key in range(start, stop + 1):
            if key in fragments:
                found = True
                break

        return found

    @staticmethod
    def getFragmentsInRange(fragments, start, stop):
        frags = []
        if start < 0:
            return frags
        if not isinstance(fragments, dict):
            print 'fragments is not a dictionary'
            return frags

        if stop < start:
            temp = start
            start = stop
            stop = temp

        # search inclusive
        for key in range(start, stop + 1):
            if key in fragments:
                frags.append(fragments[key])

        return frags

    def _findStaffsByHorizontalLines(self):
        houghLinesY = []
        for line in self.lineDetector.getHorizontalLines():
            houghLinesY.append(line.getYInt())
        houghLinesY.sort()

        fragmentDict = {}
        for line in self.lineDetector.getLineFragments():
            if -1 <= line.getRotation() <= 1:
                # print line.getRotation()
                fragmentDict[line.getYInt()] = line
        self.lineFragmentsDict = fragmentDict

        foundTopEdge = False
        foundFragment = False
        foundBottomEdge = False

        edgeCount = len(houghLinesY)
        topEdge = -1
        createNewStave = True
        stave = None
        staveList = []
        i = 0
        # group detected lines into staffs
        while True:
            if i > 1000:
                break
            if i >= edgeCount:
                break

            if topEdge == -1:
                topEdge = houghLinesY[i]
                foundTopEdge = True

            if createNewStave:
                createNewStave = False
                stave = Stave()
                stave.setTopY(topEdge)
                stave.setImage(self.lineDetector.getImage())

            if i + 1 < edgeCount:
                # get 2nd edge
                bottomEdge = houghLinesY[i + 1]
                dTop2Top = topEdge - stave.getTopY()
                dTop2Bottom = bottomEdge - stave.getTopY()
                dYEdges = bottomEdge - topEdge

                # check if any edge belongs to new stave
                maxStaveHeight = self.avgStaffHeight + 2 * self.avgStaffLineHeight
                if dTop2Top > maxStaveHeight or dTop2Bottom > maxStaveHeight:
                    if dYEdges <= self.avgStaffLineHeight * 2:
                        # save previous stave
                        staveList.append(stave)

                        # start new stave if both edges belong to it
                        stave = Stave()
                        stave.setTopY(topEdge)
                        stave.setImage(self.lineDetector.getImage())

                        foundBottomEdge = True
                        foundFragment = self.isFragmentInRange(fragmentDict, topEdge, bottomEdge)
                        stave.addStaffLineProspect(topEdge + self.avgStaffLineHeight / 2,
                                                   self.calcConfidence(foundTopEdge, foundBottomEdge, foundFragment))

                        foundBottomEdge = False
                        foundFragment = False
                        topEdge = -1
                        i += 2
                        continue

                    # check if only bottom edge belongs to new stave
                    if dTop2Bottom > maxStaveHeight:
                        # at top edge to old stave
                        foundFragment = self.isFragmentInRange(fragmentDict, topEdge,
                                                               topEdge + self.avgStaffLineHeight) or self.isFragmentInRange(
                            fragmentDict, topEdge, topEdge - self.avgStaffLineHeight)
                        stave.addStaffLineProspect(topEdge,
                                                   self.calcConfidence(foundTopEdge, foundBottomEdge, foundFragment))
                        staveList.append(stave)

                        # start loop with next edge as new top edge belonging to a new stave
                        topEdge = bottomEdge
                        foundBottomEdge = False
                        foundFragment = False
                        createNewStave = True
                        continue

                # both edges are within current stave
                # check if bottom edge matches top
                if dYEdges <= self.avgStaffLineHeight * 2:
                    foundBottomEdge = True
                    foundFragment = self.isFragmentInRange(fragmentDict, topEdge, bottomEdge)
                    stave.addStaffLineProspect(topEdge + self.avgStaffLineHeight / 2,
                                               self.calcConfidence(foundTopEdge, foundBottomEdge, foundFragment))

                    foundBottomEdge = False
                    foundFragment = False
                    topEdge = -1
                    i += 2
                    continue

                if dYEdges >= self.avgStaffLineHeight + self.avgStaffSpaceHeight:
                    # bottom edge is possible top edge of new staff line
                    foundFragment = self.isFragmentInRange(fragmentDict, topEdge,
                                                           topEdge + self.avgStaffLineHeight) or self.isFragmentInRange(
                        fragmentDict, topEdge, topEdge - self.avgStaffLineHeight)
                    stave.addStaffLineProspect(topEdge,
                                               self.calcConfidence(foundTopEdge, foundBottomEdge, foundFragment))

                    topEdge = bottomEdge
                    foundBottomEdge = False
                    foundFragment = False
                    continue

                # ignore edge if it doesn't match previous conditions
                foundBottomEdge = False
                foundFragment = False
                i += 1
                continue
            else:
                # last edge
                foundBottomEdge = False
                foundFragment = self.isFragmentInRange(fragmentDict, topEdge,
                                                       topEdge + self.avgStaffLineHeight) or self.isFragmentInRange(
                    fragmentDict,
                    topEdge,
                    topEdge - self.avgStaffLineHeight)
                stave.addStaffLineProspect(topEdge, self.calcConfidence(foundTopEdge, foundBottomEdge, foundFragment))
                foundFragment = False
                staveList.append(stave)
                stave = None
                i += 1
        # add dangling staff to list if there is any left
        if stave is not None:
            staveList.append(stave)

        self.staffs = staveList
        print '\nFound %d staffs:' % len(self.staffs)

    @staticmethod
    def _getBucket(dY, lineHeight, spaceHeight):
        if dY <= lineHeight + spaceHeight/2.:
            return 0
        if dY <= 2*lineHeight + spaceHeight * 1.5:
            return 1
        if dY <= 3*lineHeight + spaceHeight * 2.5:
            return 2
        if dY <= 4*lineHeight + spaceHeight * 3.5:
            return 3
        return 4

    def _estimateMissingStaffLines(self, staffLines):
        # TODO use line fragments to estimate coordinates
        maxExpectedStaffDistance = self.avgStaffLineHeight + self.avgStaffSpaceHeight

        # swap lines if gap too big
        for lineIdx in range(1, len(staffLines)):
            lineYPrev, confPrev = staffLines[lineIdx - 1]
            lineY, conf = staffLines[lineIdx]

            dY = lineY - lineYPrev
            if confPrev > self.staffNotFoundConfidence and dY > maxExpectedStaffDistance and lineIdx + 1 < len(staffLines):
                lineYNext, confNext = staffLines[lineIdx + 1]
                # do swap if next line was not detected
                if confNext == self.staffNotFoundConfidence:
                    staffLines[lineIdx] = (lineYNext, confNext)
                    staffLines[lineIdx + 1] = (lineY, conf)

        for lineIdx in range(0, len(staffLines)):
            lineY, conf = staffLines[lineIdx]

            if conf == self.staffNotFoundConfidence:
                prevIdx = lineIdx - 1
                nextIdx = lineIdx + 1

                if prevIdx < 0:
                    # first line
                    yNext, confNext = staffLines[nextIdx]
                    if confNext > self.staffNotFoundConfidence:
                        staffLines[lineIdx] = (yNext - maxExpectedStaffDistance, self.staffNotFoundConfidence)
                    continue

                if nextIdx >= len(staffLines):
                    # last item
                    yPrev, confPrev = staffLines[prevIdx]
                    if confPrev > self.staffNotFoundConfidence:
                        staffLines[lineIdx] = (yPrev + maxExpectedStaffDistance, self.staffNotFoundConfidence)
                    continue

                yNext, _ = staffLines[nextIdx]
                yPrev, _ = staffLines[prevIdx]
                staffLines[lineIdx] = (int((yNext + yPrev) / 2), self.staffNotFoundConfidence)

    def _findStaffLinesInStaffs(self):
        for stave in self.staffs:
            #print stave
            staffLines = []

            lines = sorted(stave.lines, key=lambda x: x[0])
            staffHeightInterval = (self.avgStaffLineHeight + self.avgStaffSpaceHeight)
            staveTopY = stave.getTopY()

            # pre-compute staff line estimates
            for lineIdx in range(0, 5):
                estimatedY = self.avgStaffLineHeight / 2 + staveTopY + lineIdx * staffHeightInterval
                staffLines.append((estimatedY, self.staffNotFoundConfidence))

            # find staff lines by best matching horizontal line
            lineIdx = -1
            linesWithHighestConfidence = []
            print ''
            for lineY, lineConf in lines:
                #print 'y: %d c: %.2f' % (lineY, lineConf)
                if lineIdx == -1:
                    linesWithHighestConfidence.append((lineY, lineConf))
                    lineIdx += 1
                    continue

                y, conf = linesWithHighestConfidence[lineIdx]
                dY = lineY - y
                # replace previous line if it has lower confidence
                if dY <= self.avgStaffLineHeight + self.avgStaffSpaceHeight\
                        /2:
                    if conf < lineConf:
                        linesWithHighestConfidence[lineIdx] = (lineY, lineConf)
                    continue

                linesWithHighestConfidence.append((lineY, lineConf))
                lineIdx += 1

            lineIdx = 0
            #print 'highest confidence'
            for lineY, lineConf in linesWithHighestConfidence:
                #print 'y: %d c: %.2f' % (lineY, lineConf)

                # determine nearest staff line for current line
                dYEdges = lineY - staveTopY
                lineIdx = self._getBucket(dYEdges, self.avgStaffLineHeight, self.avgStaffSpaceHeight)

                staffY, staffConf = staffLines[lineIdx]
                dY = lineY - staffY
                if dY <= self.avgStaffLineHeight + self.avgStaffSpaceHeight\
                        /2 and lineConf > staffConf:
                    staffLines[lineIdx] = (lineY, lineConf)
                elif dY > self.avgStaffLineHeight + self.avgStaffSpaceHeight\
                        /2 and lineIdx + 1 < 5:
                    # update next staff line if current line has higher confidence
                    lineIdx += 1
                    staffY, staffConf = staffLines[lineIdx]
                    if lineConf > staffConf:
                        staffLines[lineIdx] = (lineY, lineConf)


            # interpolate missing staffs
            self._estimateMissingStaffLines(staffLines)
            #print 'Staffs'
            #print staffLines
            lineIdx = 1
            for lineY, lineConf in staffLines:
                print 'y: %d c: %.2f' % (lineY, lineConf)
                stave.setStaffLine(lineNumber=lineIdx, y=lineY, confidence=lineConf)
                lineIdx += 1

            #avgLineIntervall = stave.getAvgStaffLineIntervall()
            #print avgLineIntervall
            #print ''

    def run(self):
        # find all lines in image
        detector = self.getLineDetector()
        if detector.run():
            # calculate staff thickness pre pass
            staffLine_height, staffSpace_height = self.calcStaffThickness(detector)

            # find possible staff lines on filtered image
            img = detector.isolateHorizontalLines(detector.getImage())

            # start new line detection on cleaned up image
            detector2 = LineDetector()
            detector2.setImage(img)
            if detector2.run():
                # calculate staff thickness again after filter pass
                staffLine_height, staffSpace_height = self.calcStaffThickness(detector)

            self.avgStaffLineHeight = staffLine_height
            self.avgStaffSpaceHeight = staffSpace_height
            self.avgStaffHeight = staffLine_height * 5 + staffSpace_height * 4

            # find staves based on staff line metrics
            self._findStaffsByHorizontalLines()

            # update staffs based on previous detection metrics
            for staff in self.staffs:
                staff.setAvgStaffLineHeight(self.avgStaffLineHeight)
                staff.setAvgStaffSpaceHeight(self.avgStaffSpaceHeight)

            # now find staff lines in each staff
            self._findStaffLinesInStaffs()

            # get boundaries btw staffs
            for i in range(1, len(self.staffs)):
                prev = self.staffs[i-1]
                staff = self.staffs[i]
                bound = (staff.getTopY() + prev.getBottomY())/2
                staff.setUpperBound(bound)
                prev.setLowerBound(bound-1)

            # get left and right coordinates
            # TODO get proper coordinates not just smallest and largest x values
            minLeft = -1
            minTop = -1
            maxRight = -1
            maxBottom = -1

            lastStaff = None
            for staff in self.staffs:
                lastStaff = staff
                xLeft = []
                xRight = []
                for line in self.getFragmentsInRange(self.lineFragmentsDict,staff.getTopY(),staff.getBottomY()):
                    xLeft.append(line.getXInt())
                    xRight.append(line.getX2Int())

                xLeft.sort()
                staff.setLeftBound(xLeft[0])
                xRight.sort(reverse=True)
                staff.setRightBound(xRight[0])

                if minLeft == -1:
                    minLeft = xLeft[0]
                else:
                    minLeft = min(minLeft, xLeft[0])

                if maxRight == -1:
                    maxRight = xRight[0]
                else:
                    maxRight = max(maxRight, xRight[0])

                if minTop == -1:
                    minTop = staff.getUpperBound()

            if lastStaff is not None:
                maxBottom = lastStaff.getLowerBound()

            # set bounding rectangle based off of the above calculated stave metrics
            self.setBoundingRect(minTop, minLeft, maxBottom, maxRight)

            # find primitives within each staff
            self.findPrimitives()

    def findPrimitives(self):
        for staff in self.staffs:
            imgRow = staff.getRowImage()
            # clean up staff image by removing all staff lines
            # remaining (more or less) horizontal lines are now more likely to be bars connected to notes or rests
            img = LineDetector.stripStaffLines(imgRow, self.getAvgStaffLineHeight())

            # find vertical lines
            # line should have at least 3 times the height of a note head
            minLenght = self.getAvgStaffSpaceHeight() * 3
            for line in LineDetector.findLinesByHoughtransform(img, minLenght):
                if line.getRotation() <= -89 or line.getRotation() >= 89:
                    staff.addVerticalLine(line)

            # find bars belonging to notes
            minLenght = self.getAvgStaffSpaceHeight()
            for line in LineDetector.findLinesByHoughtransformProbabilistic(img, minLenght):
                if -30 <= line.getRotation() <= 30 and line.getLength() > minLenght * 1.2:
                    staff.addBar(line)

            # find vertical line fragments
            minLenght = self.getAvgStaffSpaceHeight()
            for line in LineDetector.findLinesByHoughtransformProbabilistic(img, minLenght):
                if line.getRotation() <= -89 or line.getRotation() >= 89:
                    staff.addVerticalLineFragment(line)

    def getStaffCount(self):
        return len(self.staffs)

    def setBoundingRect(self, top, left, bottom, right):
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

    def getBoundingRect(self):
        return (self.left, self.top) , (self.right, self.bottom)

""" takes an image and completely removes everything but horizontal lines """
def extract_horizontal_lines(img):
    horizontal = cv2.bitwise_not(img)
    ksize = int(horizontal.shape[1] / 30)
    kernel = np.ones((1, ksize), np.uint8)
    horizontal = cv2.erode(horizontal, kernel)
    horizontal = cv2.dilate(horizontal, kernel)
    return cv2.bitwise_not(horizontal)

""" returns y coordinates of all horizontal lines found in the given image """
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
    # print('Found %s horizontal lines: ' % len(hlines))
    for line in hlines:
        for rho, theta in line:
            # convert from polar to cartesian
            y = np.sin(theta) * rho
            # print(y, theta, rho)
            lines.append(y)

    lines.sort()
    return lines

""" returns triplet of y coordinates (top, bottom, middle) of each staff line """
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

""" returns average staff height """
def get_avg_staff_height(stafflines):
    assert (len(stafflines) % 5 == 0), "Did not detect all staff lines"
    deltaY = 0
    c = 0
    for i in range(0, int(len(stafflines)/5)):
        staff_edge_top_y = stafflines[i * 5][0]
        staff_edge_bottom_y = stafflines[i * 5 + 4][1]
        deltaY += staff_edge_bottom_y - staff_edge_top_y
        c += 1

    assert (c != 0)
    return deltaY / c

""" returns average staff line height """
def get_avg_staffline_height(stafflines):
    assert (len(stafflines) % 5 == 0), "Did not detect all staff lines"
    deltaY = 0
    for line in stafflines:
        deltaY += line[1] - line[0]  # y bottom - y top

    return deltaY / len(stafflines)

""" returns average staff space height """
def get_avg_staffspace_height(stafflines):
    assert (len(stafflines) % 5 == 0), "Did not detect all staff lines"
    deltaY = 0
    c = 0
    for i in range(1, len(stafflines)):
        if i % 5 == 0:
            continue
        deltaY += stafflines[i][0] - stafflines[i-1][1]  # y top - y bottom of prev line
        c += 1

    assert (c != 0)
    return deltaY / c

""" returns triplets of all staff line y coordinates (top, bottom, middle) found in given image """
def get_staffline_coordinates(img):
    hlines = extract_horizontal_lines(img)
    edges = cv2.Canny(hlines, 100, 200)
    lines = find_horizontal_lines(edges)
    stafflines = find_staff_lines(lines)

    return stafflines

def get_row_coords(staffLineCoords):
    assert (len(staffLineCoords) % 5 == 0), "Did not detect all staff lines"
    staffHeight = get_avg_staff_height(staffLineCoords)

    # isolate staff rows
    # first row starts at half the distance off of the first/top staff line edge
    deltaLineMargin = staffHeight
    row_y_start = staffLineCoords[0][0] - deltaLineMargin
    row_y_end = staffLineCoords[4][1] + deltaLineMargin

    # contains all rows with corresponding start and end y-coords
    rowCoords = [[row_y_start, row_y_end]]

    numrows = int(len(staffLineCoords) / 5)
    # iterate over all remaining lines
    for i in range(1, numrows):
        row_y_start = staffLineCoords[5 * i][0] - deltaLineMargin
        row_y_end = staffLineCoords[(5 * i) + 4][1] + deltaLineMargin
        rowCoords.append([row_y_start, row_y_end])

    return rowCoords

""" splits image into rows only containing staffs """
def get_staff_rows(img, stafflinecoords):
    rowCoords = get_row_coords(stafflinecoords)
    rows = []
    for rowY in rowCoords:
        rows.append(img[int(rowY[0]):int(rowY[1])])
    return rows

""" make image binary. values of returned image are either 0 or 255 """
def get_binary_image(img):
    bw = cv2.adaptiveThreshold(cv2.bitwise_not(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    return cv2.bitwise_not(bw)

""" delete all staff lines """
def delete_stafflines(img,  stafflinecoords):
    img = cv2.bitwise_not(img)
    brightness_threshold = 60
    for i in range(0, len(stafflinecoords)):
        staffTop = int(stafflinecoords[i][0] + .5)  # - rowCoords[0][0] # y of first row
        staffMiddle = int(stafflinecoords[i][2] + .5)
        staffBottom = int(stafflinecoords[i][1] + .5)

        _, imgCols = img.shape
        for j in range(0, imgCols):
            topempty = img[staffTop - 1][j] < brightness_threshold
            bottomempty = img[staffBottom + 1][j] < brightness_threshold
            if topempty and bottomempty:
                img[staffTop][j] = 0
                img[staffMiddle][j] = 0
                img[staffBottom][j] = 0

    # kill remaining horizontal lines
    kernel = np.ones((3, 1), np.uint8)
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)
    #plot(cv2.GaussianBlur(img, (3, 3), 0))
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    return cv2.bitwise_not(img)

def image_without_staves(img):
    #img = cv2.imread(img)
    #img_gui, img_int = prepareImage(img)

    staffLineCoords = get_staffline_coordinates(img)

    staffHeight = get_avg_staff_height(staffLineCoords)
    print(staffHeight)

    staffLineHeight = get_avg_staffline_height(staffLineCoords)
    print(staffLineHeight)

    staffSpaceHeight = get_avg_staffspace_height(staffLineCoords)
    print(staffSpaceHeight)

    bw = get_binary_image(img)

    img_withoutstafflines = delete_stafflines(img, staffLineCoords)

    return img_withoutstafflines

def showImg(img, scaled=False):
    if img is None:
        return
    imgHeight = img.shape[0]

    if scaled and imgHeight > 720:
        scaleF = 680. / imgHeight
        img = cv2.resize(img, None, fx=scaleF, fy=scaleF)
        cv2.imshow('img (scaled)', img)
    else:
        cv2.imshow('img', img)
    cv2.waitKey(0)


""" pre processing functions """

""" iteratively get best matching image rotation angle for horizontal lines """
def getImageRotation(image):
    print 'find best image rotation...'
    prevCount = 0
    foundLines = False
    results = {}

    # rotate counter clockwise
    for angle in range(0, 100, 5):
        angle = float(angle) / 10
        rotImg = rotate_img(image, angle)
        count = getHorizontalLineCount(LineDetector.findLinesByHoughtransform(rotImg))
        if count > 0:
            foundLines = True
            results[angle] = count
        print 'angle: %f count: %d' % (angle, count)
        if count + prevCount == 0 and foundLines:
            #abort search since last two searches yielded nothing
            break
        prevCount = count

    # rotate clock wise
    for angle in range(-5, -100, -5):
        angle = float(angle) / 10
        rotImg = rotate_img(image, angle)
        count = getHorizontalLineCount(LineDetector.findLinesByHoughtransform(rotImg))
        if count > 0:
            foundLines = True
            results[angle] = count
        print 'angle: %f count: %d' % (angle, count)
        if count + prevCount == 0 and foundLines:
            #abort search since last two searches yielded nothing
            break
        prevCount = count

    bestAngle, bestCount = sorted(results.items(), key=lambda x: x[1], reverse=True)[0]

    #search with smaller step size (0.1 degrees) near current maximum
    startAngle = bestAngle - 0.5
    stopAngle = bestAngle + 0.3
    print '\nstart detailed search'
    while startAngle < stopAngle:
        startAngle += 0.1
        rotImg = rotate_img(image, startAngle)
        count = getHorizontalLineCount(LineDetector.findLinesByHoughtransform(rotImg))
        if count > bestCount:
            bestCount = count
            bestAngle = startAngle
        print 'angle: %f count: %d' % (startAngle, count)

    print 'Best rotation angle: %.2f found %d horizontal lines\n' % (bestAngle, bestCount)
    return bestAngle
""" used by getImageRotation """
def getHorizontalLineCount(lines):
    c = 0
    for line in lines:
        if 1.57 <= line.getTheta() <= 1.58:
            c+=1
    return c

""" rotate image by given angle (degrees) """
def rotate_img(image, angle):
    if -0.4 <= angle <= 0.4:
        return image
    if image is None or not isinstance(image, np.ndarray):
        print 'No valid image'
        return None
    rot_center = (image.shape[1] / 2, image.shape[0] / 2)
    rot_degrees = angle  # counter clockwise
    rot_scaling = 1
    M = cv2.getRotationMatrix2D(rot_center, rot_degrees, rot_scaling)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return image

""" enhance contrast by applying histogram equalization"""
def enhance_contrast(image):
    if isinstance(image, np.ndarray):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # convert the YUV image back to RGB format
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return image

""" crop image by given top and bottom coordinates """
def crop_image(image, minX, maxX, minY, maxY, padding=0):
    if not isinstance(image, np.ndarray):
        print 'invalid image format'
        return image
    if padding > 0:
        minpX = minX - padding
        minpY = minY - padding
        maxpX = maxX + padding
        maxpY = maxY + padding

        if minpX < 0:
            minpX = 0
        if minpY < 0:
            minpY = 0
        if maxpX > image.shape[1]:
            maxpX = image.shape[1]-1
        if maxpY > image.shape[0]:
            maxpY = image.shape[0]-1

        return image[minpY:maxpY + 1, minpX:maxpX + 1]

    return image[minY:maxY+1,minX:maxX+1]

""" sharpens edges of given image """
def sharpen_edges(image):
    if not isinstance(image, np.ndarray):
        print 'invalid image format'
        return image
    kernel = np.array([ [-1, -1, -1, -1, -1],
                        [-1,  2,  2,  2, -1],
                        [-1,  2,  8,  2, -1],
                        [-1,  2,  2,  2, -1],
                        [-1, -1, -1, -1, -1]] ) / 8.0 # normalize to avoid increased brightness
    return cv2.filter2D(image, -1, kernel)

""" applies some image enhancing techniques to alleviate music symbol recognition on given sheet for the human eye """
def get_enhanced_image_for_gui(staffDetector, cropImage=False):
    if not isinstance(staffDetector, StaffDetector):
        print "invalid staff detector object"
        return None

    img = staffDetector.getImage()
    if cropImage:
        # 1. crop image (to relevant parts)
        top, bottom = staffDetector.getBoundingRect()
        # cv2.rectangle(img, top, bottom, (0, 0, 255), 1)
        # showImg(img,scaled=True)
        img = crop_image(img, top[0], bottom[0], top[1], bottom[1], staffDetector.getAvgStaffSpaceHeight() * 3)

    # 2. increase contrast
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = sharpen_edges(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

""" get gray image for internal use"""
def get_gray_image(staffDetector, cropImage=False):
    if not isinstance(staffDetector, StaffDetector):
        print "invalid staff detector object"
        return None

    img = staffDetector.getImage()
    if cropImage:
        # 1. crop image (to relevant parts)
        top, bottom = staffDetector.getBoundingRect()
        # cv2.rectangle(img, top, bottom, (0, 0, 255), 1)
        # showImg(img,scaled=True)
        img = crop_image(img, top[0], bottom[0], top[1], bottom[1], staffDetector.getAvgStaffSpaceHeight() * 3)

    # 2. increase contrast
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def get_rotated_image(image):
    angle = getImageRotation(image)
    return rotate_img(image, angle)

""" detection test """
if __name__ == "__main__":

    testImagePath = 'data/MusicSheets/incorrectSize/brSmall.png'
    testImagePath = 'data/MusicSheets/incorrectSize/Domglocken.jpg'
    testImagePath = 'data/MusicSheets/incorrectSize/Ex4.jpg'
    testImagePath = 'data/MusicSheets/TestSheet.png'
    testImagePath = 'data/MusicSheets/evaluationsheet.png'
    testImagePath = 'data/MusicSheets/incorrectSize/Entchen_rotated.jpg'
    testImagePath = 'data/MusicSheets/piano.jpg'
    testImagePath = 'data/MusicSheets/incorrectSize/tannenbaum.jpg'
    testImagePath = 'data/MusicSheets/incorrectSize/Entchen.png'
    testImagePath = 'data/MusicSheets/Choir.png'

    # 1. correct image for rotation
    img = cv2.imread(testImagePath)

    #normalizedImg = enhance_contrast(rotatedImg)
    #showImg(normalizedImg)
    #showImg(rotatedImg)

    # 2. find staves on rotated image
    staffDetector = StaffDetector()
    staffDetector.setImage(img)
    staffDetector.run()

    #staffs = staffDetector.getStaffs()
    cropImage = False
    img_gui = staffDetector.getImageForGui(cropped=cropImage)
    img_internal = staffDetector.getImageInternal(cropped=cropImage)
    #showImg(img_gui, scaled=True)
    # showImg(img_internal,scaled=True)
    print 'Staff height: %f' % staffDetector.getAvgStaffHeight()
    img = img_internal
    staffs = staffDetector.getStaffs()
    # for staff in staffs:
    #     for line in staff.getStaffLines():
    #         cv2.line(img, (line.getXInt(), line.getYInt()), (line.getX2Int(), line.getY2Int()), (0, 255, 0), 1)
    #     for box in staff.getVerticalsAsPrimitives():
    #         width = box.width
    #         height = box.height
    #         x = box.xCoordinate
    #         y = box.yCoordinate
    #         xTop = int(x - width/2)
    #         xBottom = int(x + width/2)
    #         yTop = int(y - height/2)
    #         yBottom = int(y + height/2)
    #         #cv2.rectangle(img, (xTop, yTop), (xBottom, yBottom), (0, 0, 255), 1)
    # showImg(img,scaled=True)
    # exit(0)

    # img = staffDetector.getLineDetector().getBinaryImage()
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # print 'Found %d staffs' % staffDetector.getStaffCount()
    for staff in staffs:
        for line in staff.getStaffLines():
            cv2.line(img, (line.getXInt(), line.getYInt()), (line.getX2Int(), line.getY2Int()), (0, 255, 0), 1)
        #
        upperBound = staff.getUpperBound()
        lowerBound = staff.getLowerBound()
        left = staff.getLeftBound()
        right = staff.getRightBound()
        top = staff.getTopY()
        bottom = staff.getBottomY()
        print '\n top: %d bottom: %d' % (staff.getTopY(), staff.getBottomY())
        print ' left: %d right: %d' % (staff.getLeftBound(), staff.getRightBound())
        print ' confidence: %.2f ' % (staff.getConfidence())
        #img = staff.getRowImage()

        for line in staff.getVerticalLines():
            #cv2.line(img, (line.getXInt(), 0), (line.getX2Int(), 1000), (255, 0, 0), 1)
            pass

        for box in staff.getVerticalsAsPrimitives():
            #print box
            #cv2.rectangle(img, (box.getXInt(), 0), (box.getX2Int(), 200), (255, 0, 0), 1)
            width = box.width
            height = box.height
            x = box.xCoordinate
            y = box.yCoordinate
            xTop = int(x - width/2)
            xBottom = int(x + width/2)
            yTop = int(y - height/2)
            yBottom = int(y + height/2)
            cv2.rectangle(img, (xTop, yTop), (xBottom, yBottom), (255, 0, 0), 1)

        for line in staff.getVerticalLineFragments():
            #cv2.line(img, (line.getXInt(), line.getYInt()), (line.getX2Int(), line.getY2Int()), (0, 0, 255), 1)
            pass

        for box in staff.getBarsAsPrimitives():
            width = box.width
            height = box.height
            x = box.xCoordinate
            y = box.yCoordinate
            xTop = int(x - width / 2)
            xBottom = int(x + width / 2)
            yTop = int(y - height / 2)
            yBottom = int(y + height / 2)
            cv2.rectangle(img, (xTop, yTop), (xBottom, yBottom), (0, 0, 255), 1)
            #cv2.line(img, (line.getXInt(), line.getYInt()), (line.getX2Int(), line.getY2Int()), (0, 0, 255), 1)
            pass

    #showImg(img,scaled=True)
    cv2.imwrite('test1.png', img)



