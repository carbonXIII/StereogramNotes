from feature_matching import ImageWithFeatures, remove_match_outliers
import cv2 as cv
import numpy as np
import random
import math

def svd(mat):
    b = mat[:, 0].copy()
    ret = np.linalg.lstsq(mat[:, 1:], -b, rcond=None)[0]
    return np.r_[1, ret]

def eight_point(matches):
    A = [];
    for match in matches:
        u,v = match
        A.append([u[0]*v[0],
                  u[0]*v[1],
                  u[0],
                  u[1]*v[0],
                  u[1]*v[1],
                  u[1],
                  v[0],
                  v[1],
                  1])

    A = np.array(A)
    F_ = svd(A).reshape((3, 3))

    u, s, v = np.linalg.svd(F_)

    s[-1] = 0
    s_mat = np.zeros((3, 3))
    np.fill_diagonal(s_mat, s)

    return np.matmul(u, np.matmul(s_mat, v))

def line_point_dist(line, point):
    return np.dot(point, line) / math.hypot(line[0], line[1])

def eight_point_ransac(matches, batch_size = 32, iterations = 200, threshold = 5):
    batch_size = min(len(matches) / 2, batch_size)

    best = None
    for _ in range(iterations):
        reps = random.choices(matches, k = batch_size)
        cand = eight_point(reps)

        k = 0
        for u, v in matches:
            U = np.r_[u, 1]
            V = np.r_[v, 1]
            dist = abs(line_point_dist(np.matmul(cand, U), V))
            if dist < threshold:
                k += 1

        if not best or best[1] < k:
            best = (cand, k)

    return best

def find_inliers(matches, F, threshold = 5):
    ret = []
    for u, v in matches:
        U = np.r_[u, 1]
        V = np.r_[v, 1]
        dist = abs(line_point_dist(np.matmul(F, U), V))
        if dist < threshold:
            ret += [(u, v)]
    return ret

def find_epipoles(matches, F):
    A = []
    B = []
    for u, v in matches:
        U = np.r_[u, 1]
        V = np.r_[v, 1]
        A.append(np.matmul(F, U))
        B.append(np.matmul(F.transpose(), V))

    A = np.array(A)
    B = np.array(B)

    return svd(A), svd(B)

def find_rectification_homographies(matches, F, right_shape, f):
    el, er = find_epipoles(matches, F)

    h, w = right_shape[:2]
    T = [[1,0,-w/2],
         [0, 1,-h/2],
         [0, 0, 1]]

    er = np.matmul(T, er)
    er /= er[-1]
    magr = math.hypot(er[0], er[1])

    print(er)

    alpha = -1 if er[0] < 0 else 1

    R = [[ alpha * er[0] / magr, alpha * er[1] / magr, 0],
         [-alpha * er[1] / magr, alpha * er[0] / magr, 0],
         [0, 0, 1]]

    G = [[1, 0, 0],
         [0, 1, 0],
         [-1/f, 0, 1]]

    Hr = np.matmul(np.linalg.inv(T), np.matmul(G, np.matmul(R, T)))

    return None, Hr

def main():
    left = ImageWithFeatures(cv.imread('../../data/left.tif'), 1000, 0, -5.87105309784089)
    right = ImageWithFeatures(cv.imread('../../data/right.tif'), 1000)

    matches = left.match(right)
    matches = remove_match_outliers(matches)
    print('Initial Matches:', len(matches))

    F, k = eight_point_ransac(matches)
    print(F, k)

    matches = find_inliers(matches, F)
    print('Inlier Matches:', len(matches))

    def draw_frame():
        frame = left.img.copy()
        def __lerp(a, b, x):
            return np.add(np.multiply(a, x), np.multiply(b, 1 - x))

        def __sigmoid(x):
            return 1 / (1 + math.exp(-x))

        norm_dx = sorted([abs(u[0] - v[0]) for u, v in matches])[-1]
        for u,v in matches:
            color = (0, 255, 0)
            color = __lerp((255, 0, 0), (0, 0, 255), __sigmoid((u[0] - v[0]) / norm_dx))

            l = np.matmul(F, np.r_[u, 1])
            s = (0, -l[2] / l[1])
            t = (frame.shape[1], (-l[0] * frame.shape[1] - l[2]) / l[1])
            cv.line(frame, u, v, color, 2)
            # cv.line(frame, np.int0(s), np.int0(t), color, 2)
            #cv.circle(frame, u, 5, (255, 0, 0), -1)
            # cv.circle(frame, v, 5, (0, 0, 255), -1)

        cv.imshow('epipolar lines', frame)
        cv.waitKey(0)

    draw_frame()

    hl, hr = find_rectification_homographies(matches,
                                             F,
                                             right.img.shape,
                                             500)

    frame = right.img.copy()
    frame = cv.warpPerspective(frame, hr, (1000, 1000))
    cv.imshow('warped', frame)
    cv.waitKey(0)

if __name__ == '__main__':
    main()
