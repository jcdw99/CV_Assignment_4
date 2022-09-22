
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pandas as pd
import cv2

mode = 'fountain' if False else 'bust'

def applyhomography(A,H):
    # cast the input image to double precision floats
    A = np.array(A).astype(float)
    
    # determine number of rows, columns and channels of A
    m, n, c = A.shape
    
    # determine size of output image by forward−transforming the four corners of A
    p1 = np.dot(H,np.array([0,0,1]).reshape((3,1))); p1 = p1/p1[2]
    p2 = np.dot(H,np.array([n-1, 0,1]).reshape((3,1))); p2 = p2/p2[2]
    p3 = np.dot(H,np.array([0, m-1,1]).reshape((3,1))); p3 = p3/p3[2]
    p4 = np.dot(H,np.array([n-1,m-1,1]).reshape((3,1))); p4 = p4/p4[2]
    minx = np.floor(np.amin([p1[0], p2[0], p3[0] ,p4[0]]))
    maxx = np.ceil(np.amax([p1[0], p2[0], p3[0] ,p4[0]]))
    miny = np.floor(np.amin([p1[1], p2[1], p3[1] ,p4[1]]))
    maxy = np.ceil(np.amax([p1[1], p2[1], p3[1] ,p4[1]]))
    nn = int(maxx - minx)
    mm = int(maxy - miny)

    # initialise output with white pixels
    B = np.zeros((mm,nn,c))

    # pre-compute the inverse of H (we'll be applying that to the pixels in B)
    Hi = np.linalg.inv(H)
    
    # Loop  through B's pixels
    for x in range(nn):
        for y in range(mm):
            # compensate for the shift in B's origin
            p = np.array([x + minx, y + miny, 1]).reshape((3,1))
            
            # apply the inverse of H
            pp = np.dot(Hi,p)

            # de-homogenise
            xp = pp[0]/pp[2]
            yp = pp[1]/pp[2]
            
            # perform bilinear interpolation
            xpf = int(np.floor(xp)); xpc = xpf + 1;
            ypf = int(np.floor(yp)); ypc = ypf + 1;

            if ((xpf >= 0) and (xpc < n) and (ypf >= 0) and (ypc < m)):
                B[y, x,:] = (xpc - xp)*(ypc - yp)*A[ypf,xpf,:] \
                            + (xpc - xp)*(yp - ypf)*A[ypc,xpf,:] \
                            + (xp - xpf)*(ypc - yp)*A[ypf,xpc,:] \
                            +  (xp - xpf)*(yp - ypf)*A[ypc,xpc,:] \


    return B.astype(np.uint8)

def file_to_data(datafile_name, file_path="resources/" + mode + "/"):
    df = pd.read_csv(file_path + datafile_name, sep=',', header=None)
    return df

def decomposeP(P):
    '''
        The input P is assumed to be a 3-by-4 homogeneous camera matrix.
        The function returns a homogeneous 3-by-3 calibration matrix K,
        a 3-by-3 rotation matrix R and a 3-by-1 vector c such that
        K*R*[eye(3), -c] = P.

    '''
    P = np.array(P)
    W = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])
    # calculate K and R up to sign
    Qt, Rt = np.linalg.qr((W.dot(P[:,0:3])).T)
    K = W.dot(Rt.T.dot(W))
    R = W.dot(Qt.T)
    # correct for negative focal length(s) if necessary
    D = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    if K[0,0] < 0:
        D[0,0] = -1
    if K[1,1] < 0:
        D[1,1] = -1
    if K[2,2] < 0:
        D[2,2] = -1
    K = K.dot(D)
    R = D.dot(R)
    # calculate c
    c = -R.T.dot(np.linalg.inv(K).dot(P[:,3]))
    return K, R, c

def get_plus_mat(mat):
    return np.linalg.pinv(mat)

def F_from_projection(P1, P2):
    K1, R1, C1 = decomposeP(P1)
    K2, R2, C2 = decomposeP(P2)
    C1 = np.append(C1, 1)
    P1_plus = get_plus_mat(P1)
    e_prime = np.dot(P2, C1)
    F = np.dot(P2, P1_plus)
    packed_e = np.array([
        [0, -e_prime[2], e_prime[1]],
        [e_prime[2], 0, -e_prime[0]],
        [-e_prime[1], e_prime[0], 0],
    ])
    F = np.dot(packed_e, F)
    return F

def get_img2_line(point):

    l_prime = np.dot(F, np.array([point[0],point[1], 1]))

    a = l_prime[0]
    b = l_prime[1]
    c = l_prime[2]

    point1 = (int(-100), int(-100*(-a/b) - c/b))
    point2 = (int(10000), int(10000*(-a/b) - c/b))
    return point1, point2

def get_img1_line(point):
    l = np.cross(np.array([point[0],point[1], 1]), e)
    a = l[0]
    b = l[1]
    c = l[2]
    point1 = (int(-100), int(-100*(-a/b) - c/b))
    point2 = (int(10000), int(10000*(-a/b) - c/b))
    return point1, point2


# load in P and P_prime matricies
P1 = file_to_data('P.csv', file_path='output/')
P2 = file_to_data('P_prime.csv', file_path='output/')

# lets get the decomposition of each camera matrix
K1, R1, C1 = decomposeP(P1)
K2, R2, C2 = decomposeP(P2)

# determine the K_n constant, from K1 and K2
K_n = 0.5 * (K1 + K2)

# determine the unit vectors to form basis of orthnormal matrix Rn
r1 = (C2 - C1) / np.linalg.norm(C2 - C1)
r2 = np.cross(R1[2], r1) / np.linalg.norm(np.cross(R1[2], r1))
r3 = np.cross(r1, r2)

# combine the basis vectors into the matrix Rn
R_n = np.array([r1, r2, r3])

# now we can define the two homographies, T1 and T2
T1 = np.dot(np.dot(K_n, R_n), np.dot(R1.transpose(), np.linalg.inv(K1)))
T2 = np.dot(np.dot(K_n, R_n), np.dot(R2.transpose(), np.linalg.inv(K2)))

# read in the two images
im1 = Image.open(f"resources/{mode}/{mode}_im1.jpg")
im2 = Image.open(f"resources/{mode}/{mode}_im2.jpg")

# apply the homography, and show the results
homo1 = Image.fromarray(applyhomography(im1, T1))
homo2 = Image.fromarray(applyhomography(im2, T2))
homo1.save(f"output/{mode}_homo1.jpg")
homo2.save(f"output/{mode}_homo2.jpg")

# now we can set up the drawers, so that we can draw the epipolar lines on the images
pic2 = cv2.imread(f'output/{mode}_homo2.jpg')
pic1 = cv2.imread(f'output/{mode}_homo1.jpg')


# now we can get the new versions of the P matricies
P1_prime = np.dot(np.dot(K_n, R_n), np.append(np.identity(3), (-C1).reshape((3,1)), axis=1))
P2_prime = np.dot(np.dot(K_n, R_n), np.append(np.identity(3), (-C2).reshape((3,1)), axis=1))
e = np.dot(P1_prime, np.append(C2, 1))

F_orig = F_from_projection(P1, P2)
F = F_from_projection(P1_prime, P2_prime)

# F=F_orig
# pic2 = cv2.imread(f'resources/{mode}/{mode}_im2.jpg')
# pic1 = cv2.imread(f'resources/{mode}/{mode}_im1.jpg')

# amount of lines
lines = 20
steps = [(0,i) for i in range(0, homo1.size[1], homo1.size[1] // lines)]
colours = [tuple(np.random.randint(0, 255, size=(3, )).astype(int)) for i in range(len(steps))]

for i in range(len(steps)):
    color = ( int (colours[i][ 0 ]), int (colours[i][ 1 ]), int (colours[i][ 2 ])) 
    feature = (steps[i][0], steps[i][1])
    point1, point2 = get_img2_line(feature)
    pic2 = cv2.line(pic2, point1, point2,color=color, thickness=2)

for i in range(len(steps)):
    color = ( int (colours[i][ 0 ]), int (colours[i][ 1 ]), int (colours[i][ 2 ])) 
    feature = (steps[i][0], steps[i][1])
    point1, point2 = get_img1_line(feature)
    pic1 = cv2.line(pic1, point1, point2, color=color, thickness=2)
    
pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2RGB)    
pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)

Image.fromarray(pic2).save(f"output/{mode}2_w_lines.jpg")
Image.fromarray(pic1).save(f"output/{mode}1_w_lines.jpg")
