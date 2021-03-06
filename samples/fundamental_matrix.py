from feature_matching import ImageWithFeatures, remove_match_outliers
import cv2 as cv
import numpy as np
import random
import math

def h2e(X):
    return (X / X[-1])[:-1]

def rectify_shearing(H1, image_width, image_height):

	##### ##### ##### ##### ##### 
	##### CREDIT
	##### ##### ##### ##### ##### 

	# Loop & Zhang - via literature
	#	* http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
	# TH. - via stackexchange user
	# 	* http://scicomp.stackexchange.com/users/599/th
	#	* http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification

	##### ##### ##### ##### ##### 
	##### PARAMETERS
	##### ##### ##### ##### ##### 

	# Let H1 be the rectification homography of image1 (ie. H1 is a homogeneous space)
	# Let H2 be the rectification homography of image2 (ie. H2 is a homogeneous space)
	# image_width, image_height be the dimensions of both image1 and image2

	##### ##### ##### ##### ##### 

	"""
	Compute shearing transform than can be applied after the rectification transform to reduce distortion.
	Reference:
		http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
		"Computing rectifying homographies for stereo vision" by Loop & Zhang
	"""

	w = image_width
	h = image_height

	'''
	Loop & Zhang use a shearing transform to reduce the distortion
	introduced by the projective transform that mapped the epipoles to infinity
	(ie, that made the epipolar lines parallel).
	Consider the shearing transform:
			| k1 k2 0 |
	S	=	| 0  1  0 |
			| 0  0  1 |
	Let w and h be image width and height respectively.
	Consider the four midpoints of the image edges:
	'''

	a = np.float32([ (w-1)/2.0,	0,			1 ])
	b = np.float32([ (w-1),		(h-1)/2.0,	1 ])
	c = np.float32([ (w-1)/2.0,	(h-1),		1 ])
	d = np.float32([ 0,			(h-1)/2.0,	1 ])

	'''
	According to Loop & Zhang:
	"... we attempt to preserve perpendicularity and aspect ratio of the lines bd and ca"
	'''

	'''
	Let H be the rectification homography and,
	Let a' = H*a be a point in the affine plane by dividing through so that a'2 = 1
	Note: a'2 is the third component, ie, a' = (a'[0], a'1, a'2))
	'''

	# Note: *.dot is a form of matrix*vector multiplication in numpy
	# So a_prime = H*a such that a_prime[2] = 1 (hence the use of h2e function)

	a_prime = h2e(np.matmul(H1, a.transpose()))
	b_prime = h2e(np.matmul(H1, b.transpose()))
	c_prime = h2e(np.matmul(H1, c.transpose()))
	d_prime = h2e(np.matmul(H1, d.transpose()))

	''' Let x = b' - d' and y = c' - a' '''

	x = b_prime - d_prime
	y = c_prime - a_prime

	'''
	According to Loop & Zhang:
		"As the difference of affine points, x and y are vectors in the euclidean image plane.
			Perpendicularity is preserved when (Sx)^T(Sy) = 0, and aspect ratio is preserved if [(Sx)^T(Sx)]/[(Sy)^T(Sy)] = (w^2)/(h^2)"
	'''

	''' The real solution presents a closed-form: '''

	k1 = (h*h*x[1]*x[1] + w*w*y[1]*y[1]) / (h*w*(x[1]*y[0] - x[0]*y[1]))
	k2 = (h*h*x[0]*x[1] + w*w*y[0]*y[1]) / (h*w*(x[0]*y[1] - x[1]*y[0]))

	''' Determined by sign (the positive is preferred) '''

	if (k1 < 0): # Why this?
		k1 *= -1
		k2 *= -1

	return np.float32([
		[k1,	k2,	0],
		[0,		1,	0],
		[0,		0,	1]])

def disparity_uncalibrated(left, right, verbose = False):
    matches = np.array(left.match(right))

    x1 = matches[:, 0]
    x2 = matches[:, 1]

    F, F_mask = cv.findFundamentalMat(x1, x2, method=cv.FM_LMEDS)
    F_mask = F_mask.flatten()
    x1 = x1[F_mask == 1]
    x2 = x2[F_mask == 1]

    if verbose:
        print('F:', F)
        print('Inlier points:', len(x1))

    ret, h1, h2 = cv.stereoRectifyUncalibrated(x1, x2, F, (left.img.shape[1], left.img.shape[0]), threshold = 0)
    if not ret:
        print('Failed to rectify')
        return None

    s1 = rectify_shearing(h1, left.img.shape[1], left.img.shape[0]) 
    s2 = rectify_shearing(h2, right.img.shape[1], right.img.shape[0]) 
    h1 = np.matmul(s1, h1)
    h2 = np.matmul(s2, h2)

    n = len(x1)
    x1_ = np.vstack((x1.transpose(), np.ones(n)))

    h, w = left.img.shape
    x1_ = [[0, 0, 1],
           [w, 0, 1],
           [0, h, 1],
           [w, h, 1]]
    x1_ = np.transpose(x1_)

    x1_ = np.matmul(h1, x1_)
    x1_[0] /= x1_[-1]
    x1_[1] /= x1_[-1]
    lx = x1_[0].min()
    ly = x1_[1].min()
    hx = x1_[0].max()
    hy = x1_[1].max()

    T = [[1, 0, -lx],
         [0, 1, -ly],
         [0, 0, 1]]

    sx = 1000 / (hx - lx)
    sy = 1000 / (hy - ly)
    S = [[sx, 0, 0],
         [0, sy, 0],
         [0, 0, 1]]

    h1 = np.matmul(np.matmul(S, T), h1)
    h2 = np.matmul(np.matmul(S, T), h2)

    left_w = left.img.copy()
    left_w = cv.warpPerspective(left_w, h1, (1000, 1000))

    right_w = right.img.copy()
    right_w = cv.warpPerspective(right_w, h2, (1000, 1000))

    frame = None
    if verbose:
        z = .5
        frame = np.add(np.multiply(left_w, z), np.multiply(right_w, 1 - z)).astype(np.uint8)
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    x1_ = np.vstack((x1.transpose(), np.ones(n)))
    x1_ = np.matmul(h1, x1_)

    x2_ = np.vstack((x2.transpose(), np.ones(n)))
    x2_ = np.matmul(h2, x2_)

    dxs = []
    for i in range(len(x1_[0])):
        u = np.int0(h2e(x1_[:, i]))
        v = np.int0(h2e(x2_[:, i]))

        dx = u[0] - v[0]
        dxs += [dx]

        if verbose:
            cv.line(frame, u, v, (255, 0, 0), 2)

    if verbose:
        cv.imshow('warped', frame)

    win_size = 1
    min_disp = min(dxs)
    max_disp = max(dxs) #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16
    num_disp = ((max_disp + 15) // 16) * 16
    #Create Block matching object. 
    stereo = cv.StereoSGBM_create(minDisparity= min_disp,
                                   numDisparities = num_disp,
                                   blockSize = 1,
                                   uniquenessRatio = 5,
                                   speckleWindowSize = 5,
                                   speckleRange = 5,
                                   disp12MaxDiff = 0,
                                   P1 = 8*3*win_size**2,#8*3*win_size**2,
                                   P2 =32*3*win_size**2) #32*3*win_size**2)

    disparity = stereo.compute(left_w, right_w)

    right_stereo = cv.ximgproc.createRightMatcher(stereo)
    right_disparity = right_stereo.compute(right_w, left_w)

    wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)

    filtered_disparity = wls_filter.filter(disparity, left_w, disparity_map_right=right_disparity)

    h1i = np.linalg.inv(h1)
    ret = cv.warpPerspective(filtered_disparity, h1i, (left.img.shape[1], left.img.shape[0]))
    return ret

if __name__ == '__main__':
    # Still no good way to find ty besides manual tuning,
    # but it is very important to get it right to avoid distortion during rectification
    # 3.5 pixels seems OK
    left = ImageWithFeatures(cv.imread('../../data/left.tif', cv.IMREAD_GRAYSCALE), 1000, 0, -3.5) 
    right = ImageWithFeatures(cv.imread('../../data/right.tif', cv.IMREAD_GRAYSCALE), 1000)

    disparity = disparity_uncalibrated(left, right)
    disparity = np.ones(disparity.shape) / disparity * 50000

    disparity = disparity.clip(None, 255)

    # disparity = cv.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    disparity = np.uint8(disparity)

    cv.imshow('disparity', disparity)
    cv.waitKey(0)
