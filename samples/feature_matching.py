import cv2 as cv
import numpy as np
import math

class ImageWithMatches:
    def __init__(self, img, target_size):
        scale = max(img.shape[0] / target_size, img.shape[1] / target_size)
        shape = np.int0([img.shape[1] / scale, img.shape[0] / scale])
        self.img = cv.resize(img, shape)

    def shift(self, tx, ty):
        mat = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        self.img = cv.warpAffine(self.img, mat, (self.img.shape[1], self.img.shape[0]))


    def find_features(self):
        orb = cv.ORB_create(nfeatures=100000)
        self.kp = orb.detect(self.img, None)
        self.kp, self.des = orb.compute(self.img, self.kp)

def lerp(a, b, x):
    return np.add(np.multiply(a, x), np.multiply(b, 1 - x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def length(a, b):
    x = np.add(a, np.multiply(b, -1))
    return (x.dot(x)) ** 0.5

def pts_from_match(match, left_kp, right_kp):
    return (np.int0(left_kp[match.queryIdx].pt),
            np.int0(right_kp[match.trainIdx].pt))

def match_features(des_left, des_right):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    # Match descriptors.
    matches = bf.knnMatch(des_left, des_right, k = 2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Sort them in the order of their distance (between feature vectors, not points).
    return sorted(good, key = lambda x:x.distance)

def remove_match_outliers(pts):
    pts = sorted(pts, key = lambda x: length(*x))
    n = len(pts)

    q = [length(*pts[i]) for i in [(n * x) // 4 for x in range(4)]]
    iqr = q[3] - q[1]

    def good(x):
       dist = length(*x)
       return dist >= q[1] - 1.5 * iqr and dist <= q[3] + 1.5 * iqr

    return [x for x in pts if good(x)]

# grab both stereograms, scale them down to fit in 1000x1000
left = ImageWithMatches(cv.imread('../../data/left.tif'), 1000)
right = ImageWithMatches(cv.imread('../../data/right.tif'), 1000)

# Don't currently have a good algorithm to find this automatically
# The right stereogram was shifted down relative to the left
# So this manually realigns them
left.shift(0, -10.2)

# Find the features using ORB
left.find_features()
right.find_features()

# Match the features using Brute Force Matching
# (Matches features using feature vector, not distance)
matches = match_features(left.des, right.des)
print('ORB + Brute Force Matcher, Matches:', len(matches))

# Get the corresponding points from each match
to_draw = [pts_from_match(match, left.kp, right.kp) for match in matches]
to_draw = remove_match_outliers(to_draw)
n = len(to_draw)
print('Removed outliers:', len(matches) - n)

# Get a reasonable magnitude to use to normalize colors
norm_dx = sorted([abs(u[0] - v[0]) for u, v in to_draw])[-1]

# Sanity check to ensure the images aren't shifted vertically
mean_dy = sum([v[1] - u[1] for u, v in to_draw]) / n
variance_dy = sum([(v[1] - u[1] - mean_dy) ** 2 for u, v in to_draw]) / n

print('norm dx:', norm_dx)
print('mean dy:', mean_dy)
print('stddev dy:', variance_dy ** 0.5)

# Draws the linear interpolation of the left and right features with weight, z in [0.0, 1.0]
def draw_frame(frame, z, sample_colors = None):
    for u,v in to_draw:
        color = (0, 255, 0)
        if sample_colors:
            color = lerp(sample_colors[0][u[1]][u[0]],
                         sample_colors[1][v[1]][v[0]],
                         sigmoid((u[0] - v[0]) / norm_dx))
        else:
            color = lerp((255, 0, 0), (0, 0, 255), sigmoid((u[0] - v[0]) / norm_dx))

        cv.circle(frame, np.int0(lerp(u, v, z)), 2, color, -1)

example = left.img.copy()
draw_frame(example, 1)
cv.imshow("still frame", example)
cv.waitKey(0)

video_path = 'feature_matching.avi'

fps = 24
seconds = 2
fourcc = cv.VideoWriter_fourcc(*'MP42')
video = cv.VideoWriter(video_path, fourcc, fps, (left.img.shape[1], left.img.shape[0]), True)
right_scaled = cv.resize(right.img, (left.img.shape[1], left.img.shape[0]))
for i in range(fps * seconds):
    z = (math.sin(i * math.pi / fps / seconds) + 1) / 2
    frame = np.add(np.multiply(left.img, z), np.multiply(right_scaled, 1 - z)).astype(np.uint8) # np.zeros(left.img.shape, dtype=np.uint8)
    draw_frame(frame, z)
    video.write(frame)
video.release()
