import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import
from PIL import Image, ImageDraw, ImageOps
from random import sample

ITERS = 3000
RANSAC_THRESH = 0.1
mode = 'fountain' if False else 'bust'

og_img1 = Image.open(f"resources/{mode}/{mode}_im1.jpg")
og_img2 = Image.open(f"resources/{mode}/{mode}_im1.jpg")

def file_to_data(datafile_name, file_path="resources/" + mode + "/"):
    df = pd.read_csv(file_path + datafile_name, sep=',', header=None)
    return df

def drawFilterMapping(datafile_name, pic_name, firstDex=0):
    return draw_mapping(datafile_name, pic_name, firstDex=firstDex, filterMode=True)

def draw_mapping(datafile_name, pic_name, firstDex=0, filterMode=False, file_path="resources/" + mode + "/"):
    # load in data, and images
    df = file_to_data(datafile_name)
    # load image 2 rather, if they are refering to image 2 via firstDex
    if firstDex > 1:
        pic_name = str(pic_name).replace("1", "2")

    pic = ImageOps.grayscale(Image.open(file_path + pic_name)).convert("RGB")
    draw = ImageDraw.Draw(pic)

    # we can either draw from image 2 -> image 1, or image 1 -> image 2, lets set x y x' y' accordingly
    x_dex = firstDex
    y_dex = firstDex + 1
    x_p_dex = 2 if x_dex == 0 else 0
    y_p_dex = x_p_dex + 1

    # read in data, based on xdex, ydex, x_p_dex and y_p_dex
    x = np.array(pd.to_numeric(df[x_dex]))
    y = np.array(pd.to_numeric(df[y_dex]))
    xprime = np.array(pd.to_numeric(df[x_p_dex]))
    yprime = np.array(pd.to_numeric(df[y_p_dex]))
    
    # constants for drawing
    circRadius = 2
    lineColor = "yellow" if firstDex < 2 else "blue"
    circColor = "blue" if firstDex < 2 else "yellow"
    linethickness = 2

    longest = -1
    shortest = 1000000
    rangeRatio = 0.20
    remaining = []
    # if we are filtering, pass through list and find longest and shortest match lines
    if filterMode:
        for i in range(len(x)):
            dist = np.sqrt(((x[i] - xprime[i]) ** 2) + ((y[i] - yprime[i]) ** 2))
            if dist > longest:
                longest = dist
            elif dist < shortest:
                shortest = dist

    # for each row in our data list
    for i in range(len(x)):
        if (filterMode):
            # get distance of this line
            dist = np.sqrt(((x[i] - xprime[i]) ** 2) + ((y[i] - yprime[i]) ** 2))
            # if this line is too long, dont draw it
            if dist > shortest + (rangeRatio * (longest - shortest)):
                continue

        # save this match, it is not too long
        remaining.append(np.array([x[i], y[i], xprime[i], yprime[i]]))
        # draw elipse in src location on image
        draw.ellipse(((x[i]-circRadius, y[i]-circRadius, x[i]+circRadius, y[i]+circRadius)), fill=circColor)
        # draw line to location on destination image
        draw.line([(x[i], y[i]), (xprime[i], yprime[i])], fill=lineColor, width=linethickness)
        # show resulting image
    
    check = "1" if firstDex < 2 else "2"
    # if we are in filter mode, return remaining matches as well
    if filterMode:
        pic.save(f"output/{mode}_dist" + check + ".jpg")
        return pic, np.array(remaining)
    # if not, return just the image containing ALL drawings
    pic.save(f"output/{mode}_raw" + check + ".jpg")
    return pic

def get_A_matrix(df):

    x = np.array(df[0])
    y = np.array(df[1])
    xprime = np.array(df[2])
    yprime = np.array(df[3])

    x_s = [x[i] for i in range(len(x))]
    y_s = [y[i] for i in range(len(y))]
    #TL TR BR BL
    x_primes = [xprime[i] for i in range(len(xprime))]
    y_primes = [yprime[i] for i in range(len(yprime))]
    matrix = []
    for i in range(len(x_s)):
        matrix.append(np.array([x_s[i]*x_primes[i], y_s[i]*x_primes[i], x_primes[i], x_s[i]*y_primes[i], y_s[i]*y_primes[i], y_primes[i], x_s[i], y_s[i], 1]))
    

    return np.array(matrix)

def draw_matches_after_ransac(df_set, pic_name, firstDex=0, file_path="resources/" + mode + "/"):
     # load image 2 rather, if they are refering to image 2 via firstDex
    if firstDex > 1:
        pic_name = str(pic_name).replace("1", "2")

    pic = ImageOps.grayscale(Image.open(file_path + pic_name)).convert("RGB")
    draw = ImageDraw.Draw(pic)

    # we can either draw from image 2 -> image 1, or image 1 -> image 2, lets set x y x' y' accordingly
    x_dex = 0
    y_dex = 1
    x_p_dex = 2 
    y_p_dex = 3

    # read in data, based on xdex, ydex, x_p_dex and y_p_dex
    x = np.array(pd.to_numeric(df_set[x_dex]))
    y = np.array(pd.to_numeric(df_set[y_dex]))
    xprime = np.array(pd.to_numeric(df_set[x_p_dex]))
    yprime = np.array(pd.to_numeric(df_set[y_p_dex]))
    
    # constants for drawing
    circRadius = 2
    lineColor = "yellow" if firstDex < 2 else "blue"
    circColor = "blue" if firstDex < 2 else "yellow"
    linethickness = 2
    # for each row in our data list
    for i in range(len(x)):
        # draw elipse in src location on image
        draw.ellipse(((x[i]-circRadius, y[i]-circRadius, x[i]+circRadius, y[i]+circRadius)), fill=circColor)
        # draw line to location on destination image
        draw.line([(x[i], y[i]), (xprime[i], yprime[i])], fill=lineColor, width=linethickness)
        # show resulting image

    thing = "1" if firstDex < 2 else "2"
    pic.save(f"output/{mode}_ransaced" + thing + ".jpg")
    
# datapoint is of the form [x,y, xprime, yprime]
def sampsondist(datapoint, FMat):
    x = np.array([datapoint[0], datapoint[1], 1])
    x_prime = np.array([datapoint[2], datapoint[3], 1])

    # numerator = (np.dot(np.transpose(x_prime), np.dot(FMat, x))) ** 2
    numerator = np.dot(np.dot(x_prime, FMat), x) ** 2
    term1 = np.dot(FMat, x)
    term2 = np.dot(np.transpose(FMat), x_prime)
    denom = term1[0]**2 + term1[1]**2 + term2[0]**2 + term2[1]**2
    return (numerator / denom)

def get_F_matrix(sample_data):
    A = get_A_matrix(sample_data)
    UA, SA, VA = np.linalg.svd(A)    
    # Obtain the F_Hat matrix
    F_Hat = (VA[8,:]).reshape((3,3))
    # Obtain F matrix by forcing rank to be 2
    U, S, V = np.linalg.svd(F_Hat)

    S_mat = np.array(
                [np.array([S[0], 0, 0]),
                np.array([0, S[1], 0]),
                np.array([0, 0, 0])]
            )

    F = np.dot(np.dot(U, S_mat), V)
    return F

def do_RANSAC(remaining_data):
    data = pd.DataFrame(remaining_data)
    best_inlier_set = []

    for k in range(ITERS):
        index_options = list(range(len(remaining_data)))
        mask = np.array([False] * len(remaining_data))
        # Grab 8 sample indicies randomly
        sample_index = sample(index_options, 8)
        for i in sample_index:
            mask[i] = True
        # Get these samples by applying the mask
        sample_data = data.loc[mask]
        # get the F matrix
        F = get_F_matrix(sample_data)

        inlier_set = []
        for data_dex in range(len(data)):

            # datapoint of the form [x, y, xprime, yprime]
            data_point = np.array(data.loc[data_dex])
            dist1 = sampsondist(data_point, F)

            if dist1 < RANSAC_THRESH:
                inlier_set.append(data_point)
       
        if len(inlier_set) > len(best_inlier_set):
            best_inlier_set = np.copy(inlier_set)

    return pd.DataFrame(np.array(best_inlier_set))

def get_Essential(Fmat, K, Kprime):
    E = np.dot(np.dot(np.transpose(Kprime), Fmat), K)
    U, S, V = np.linalg.svd(E)  
    # we need to remove the bias introduced by SVD
    detU = np.linalg.det(U)
    detV = np.linalg.det(V)
    if (detU > 0) and (detV < 0):
        E = -1*E
        V = -1*V
    elif (detU < 0) and (detV > 0):
        E = -1*E   
        U = -1*U
    return U,S,V,E

# point is of the form x, y, xprime, yprime
def get_X_Vector(P, P_prime, point):
    mat1 = np.array([
        (P[2,:] * point[1]) - P[1,:],
        P[0,:] - (point[0]*P[2,:])
    ])
    mat2 = np.array([
        (P_prime[2,:] * point[3]) - P_prime[1,:],
        P_prime[0,:] - (point[2]*P_prime[2,:])
    ])

    # stacked = mat1 - mat2
    stacked = np.array([mat1[0], mat1[1], mat2[0], mat2[1]])
    U, S, V = np.linalg.svd(stacked)    
    homo_vec = V[-1]
    homo_vec /= homo_vec[-1]
    return homo_vec[:3]

def get_Projection(U, V, K, Kprime, points):
    # mu_3 is defined as the third column of U
    appended_identity = np.append(np.diag((1,1,1)),np.array([0,0,0]).reshape((3,1)), axis=1)
    P = np.dot(K, appended_identity)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    u3 = U[:, 2]
    P_prime_option1 = np.dot(Kprime, np.append(np.dot(np.dot(U, W), V), u3.reshape((3,1)), axis=1))
    P_prime_option2 = np.dot(Kprime, np.append(np.dot(np.dot(U, W), V), (-u3).reshape((3,1)), axis=1))
    P_prime_option3 = np.dot(Kprime, np.append(np.dot(np.dot(U, W.transpose()), V), u3.reshape((3,1)), axis=1))
    P_prime_option4 = np.dot(Kprime, np.append(np.dot(np.dot(U, W.transpose()), V), (-u3).reshape((3,1)), axis=1))

    results = []
    options = [P_prime_option1, P_prime_option2, P_prime_option3, P_prime_option4]
    for p_option in options:
        vec_option = get_X_Vector(P, p_option, np.array(points.loc[39]))
        Calib, R, C = decomposeP(P)
        Calib_prime, R_prime, C_prime = decomposeP(p_option)
        val1 = np.dot(R[2,:].transpose(), (vec_option - C))
        val2 = np.dot(R_prime[2,:].transpose(), (vec_option - C_prime))
        results.append((val1 > 0) and (val2 > 0))

    trueDex = -1
    for i in range(len(results)):
        if results[i]:
            trueDex = i
            break
    if trueDex < 0:
        print("no true index found, quitting")
        exit()
    return P, options[trueDex]

def decomposeP(P):
    '''
        The input P is assumed to be a 3-by-4 homogeneous camera matrix.
        The function returns a homogeneous 3-by-3 calibration matrix K,
        a 3-by-3 rotation matrix R and a 3-by-1 vector c such that
        K*R*[eye(3), -c] = P.

    '''
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

def triangulateAll(P, P_prime, dataset):
    results = []
    for i in range(len(dataset)):
        results.append(get_X_Vector(P, P_prime, dataset.loc[i]))
    return results

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_camera_distance_range(P, P_prime, triangulated_set):
    triangulated_set = triangulated_set.transpose()
    Calib, R, C = decomposeP(P)
    Calib_prime, R_prime, C_prime = decomposeP(P_prime)
    # lets get closest and furthest points now
    closest = np.linalg.norm(C - triangulated_set[0])
    furthest = np.linalg.norm(C - triangulated_set[0])
    for i in range(1, len(triangulated_set)):
        thisObs = np.linalg.norm(C - triangulated_set[i])
        if thisObs < closest:
            closest = thisObs
        if thisObs > furthest:
            furthest = thisObs
    return furthest - closest
    
def filter_set(P, P_prime, set_to_filter, thresh):
    set_to_filter = set_to_filter.transpose()
    Calib, R, C = decomposeP(P)
    Calib_prime, R_prime, C_prime = decomposeP(P_prime)
    set_to_return = []
    for point in set_to_filter:
        if (np.linalg.norm(point - C) > thresh) or (np.linalg.norm(point - C_prime) > thresh):
            pass
        else:
            set_to_return.append(point)
    set_to_return = np.array(set_to_return)
    return set_to_return.transpose()

def plot_camera(ax, C1, C2, R1, Rw):

    factor = 1
    camera1_loc = C1
    camera2_loc = C2

    camera1_x = camera1_loc - -factor*R1[0]
    camera1_y = camera1_loc - factor*R1[1]
    camera1_z = camera1_loc - -factor*R1[2]

    camera2_x = camera2_loc - -factor*R2[0]
    camera2_y = camera2_loc - factor*R2[1]
    camera2_z = camera2_loc - -factor*R2[2]
    
    margin = 0.05 
    camsize = 80
    # plot camera1
    ax.scatter(camera1_loc[0], camera1_loc[1], camera1_loc[2], marker='d', color='r', label='Camera 1', s=camsize, alpha=0.6)
    # plot camera1 x
    ax.plot([camera1_loc[0], camera1_x[0]], [camera1_loc[1], camera1_x[1]], [camera1_loc[2], camera1_x[2]], color='green')
    ax.text(camera1_x[0] + margin, camera1_x[1] + margin, camera1_x[2] + margin, 'x1', color='green')
    # plot camera1 y
    ax.plot([camera1_loc[0], camera1_y[0]], [camera1_loc[1], camera1_y[1]], [camera1_loc[2], camera1_y[2]], color='blue')
    ax.text(camera1_y[0] + margin, camera1_y[1] + margin, camera1_y[2] + margin, 'y1', color='blue')
    # plot camera1 z
    ax.plot([camera1_loc[0], camera1_z[0]], [camera1_loc[1], camera1_z[1]], [camera1_loc[2], camera1_z[2]], color='black')
    ax.text(camera1_z[0] + margin, camera1_z[1] + margin, camera1_z[2] + margin, 'z1', color='black')


    # plot camera1
    ax.scatter(camera2_loc[0], camera2_loc[1], camera2_loc[2], marker='d', color='g', label='Camera 2', s=camsize, alpha=0.6)
    # plot camera1 x
    ax.plot([camera2_loc[0], camera2_x[0]], [camera2_loc[1], camera2_x[1]], [camera2_loc[2], camera2_x[2]], color='yellowgreen')
    ax.text(camera2_x[0] + margin, camera2_x[1] + margin, camera2_x[2] + margin, 'x2', color='yellowgreen')
    # plot camera1 y
    ax.plot([camera2_loc[0], camera2_y[0]], [camera2_loc[1], camera2_y[1]], [camera2_loc[2], camera2_y[2]], color='dodgerblue')
    ax.text(camera2_y[0] + margin, camera2_y[1] + margin, camera2_y[2] + margin, 'y2', color='dodgerblue')
    # plot camera1 z
    ax.plot([camera2_loc[0], camera2_z[0]], [camera2_loc[1], camera2_z[1]], [camera2_loc[2], camera2_z[2]], color='midnightblue')
    ax.text(camera2_z[0] + margin, camera2_z[1] + margin, camera2_z[2] + margin, 'z2', color='midnightblue')


# Draw raw image pairs over both images, part of 1A
map1to2Raw = draw_mapping(f"{mode}_matches.txt", f"{mode}_im1.jpg", 0)
map2to1Raw = draw_mapping(f"{mode}_matches.txt", f"{mode}_im1.jpg", 2)

# Draw (distance) filtered matches over both images, part of 1A
map1to2Filt, remaining1 = drawFilterMapping(f"{mode}_matches.txt", f"{mode}_im1.jpg", 0)
map2to1Filt, remaining2 = drawFilterMapping(f"{mode}_matches.txt", f"{mode}_im1.jpg", 2)

# Do the ransac based filtering algorithm, on both images, part of 1B
ransaced1 = do_RANSAC(remaining1)
ransaced2 = do_RANSAC(remaining2)

# # save these matches, so that I dont need to re-run ransac each time
ransaced1.to_csv(f"output/{mode}_ransaced1.csv", header=False, index=False)
ransaced2.to_csv(f"output/{mode}_ransaced2.csv", header=False, index=False)
# ransaced1 = file_to_data(f'{mode}_ransaced1.csv', file_path='output/')
# ransaced2 = file_to_data(f'{mode}_ransaced2.csv', file_path='output/')


draw_matches_after_ransac(ransaced1, f"{mode}_im1.jpg", 0)
draw_matches_after_ransac(ransaced2, f"{mode}_im1.jpg", 2)

# Re-estimate the F matrix using the ransaced points, which should only be valid matches
F1 = get_F_matrix(ransaced1)
F2 = get_F_matrix(ransaced2)

# with respect to image 1, if we run w image 2, we need to swap that.
K = file_to_data(f"{mode}_K1.txt")
K_prime = file_to_data(f"{mode}_K2.txt")

# bias-removed version of U, S, V and E
U1, S1, V1, E1 = get_Essential(F1, K, K_prime)
U2, S2, V2, E2 = get_Essential(F2, K_prime, K)

# get the P and P matrix corresponding to this triangulated point
P, P_prime = get_Projection(U1, V1, K, K_prime, ransaced1)

print(P)
print('\n\n')
print(P_prime)
# write cam matricies to file, for later use
pd.DataFrame(P).to_csv("output/P.csv", index=None, header=None)
pd.DataFrame(P_prime).to_csv("output/P_prime.csv", index=None, header=None)

# triangulate all points, using the obtained projection matrix, of the form [    [xDim], [yDim], [Zdim]    ] ?
triangulated = np.array(triangulateAll(P, P_prime, ransaced1)).transpose()

# lets get each camera location, so that we can remove the far away points
cam_dist_range = get_camera_distance_range(P, P_prime, triangulated)

# define distance threshold as percentage of range, lower bound of points we cut, anything bigger, we remover
distance_thresh = 0.25 * cam_dist_range

# filter far points, out, 15 is a good number for bust, 50 for fountain
filtered_set = filter_set(P, P_prime, triangulated, 15)

# lets get the colours of our plot
cols_of_dots = []
for i in range(filtered_set.shape[1]):
    pix = np.dot(P, np.array([filtered_set[0][i], filtered_set[1][i], filtered_set[2][i], 1]))
    pix /= pix[2]
    pix = pix[:2]
    ogpix = tuple((np.array(list(og_img1.getpixel((pix[0],pix[1])))) / 255))
    cols_of_dots.append(ogpix)

# now lets begin creating a 3d plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(filtered_set[0], filtered_set[1], filtered_set[2], c=cols_of_dots, s=3)

# now we should add the camera
K1, R1, C1 = decomposeP(P)
K2, R2, C2 = decomposeP(P_prime)
plot_camera(ax, C1, C2, R1, R2)

plt.legend()
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
set_axes_equal(ax)
plt.show()