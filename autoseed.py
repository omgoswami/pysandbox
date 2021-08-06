import numpy as np
from skimage import measure
from extract_patches import load_image_array, normalize_arr
from random import randint
import cv2 as cv
from scipy.spatial import Voronoi
import time


def threshold_otsu(img):
    if np.min(img) == np.max(img):
        return 0
    img = np.floor(img).astype(int)
    minv = np.min(img)
    maxv = np.max(img) + 1
    bins = np.arange(minv, maxv + 1) - 0.5
    probs, bins = np.histogram(img, bins)
    vals = np.arange(minv, maxv)
    v_sum = np.cumsum(probs * vals)
    n_sum = np.cumsum(probs)
    maximum = 0
    thresh = None
    o_u1 = None
    o_u2 = None
    for idx, val in enumerate(vals):
        if n_sum[-1] - n_sum[idx] <= 0:
            break
        u_1 = v_sum[idx] / n_sum[idx]
        u_2 = (v_sum[-1] - v_sum[idx]) / (n_sum[-1] - n_sum[idx])

        sig = n_sum[idx] * (n_sum[-1] - n_sum[idx]) * (u_1 - u_2) * (u_1 - u_2)

        if sig > maximum:
            thresh = val
            maximum = sig
            o_u1 = u_1
            o_u2 = u_2
            d_u = u_2 - u_1
    return (img < thresh).astype(int), thresh, o_u1, o_u2, d_u


def convert_to_binary(im_arr):
    '''
    convert each slice of the array to a binary mask
    :param im_arr: 3D numpy array of the image
    :return: numpy array with every slice a binary mask
    '''
    output = np.empty_like(im_arr)
    num = list(im_arr.shape)[2]
    binary, thresh, o_u1, o_u2, d_u = threshold_otsu(im_arr)
    for k in range(num):
        binary = (im_arr[:, :, k] < thresh).astype(int)
        binary[binary > 0] = 255
        output[:, :, k] = binary
    return output


def grow_region(image, seed, threshold=5):
    '''
    creating a new, segmented image with region growing algorithm
    :param image: input image
    :param seed: seed point for a region
    :return: segmented image
    '''
    '''
    here we set some parameters for the region growing algorithm 
    neighbor points are adjacent to pixel/region
    '''
    neighbors = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        (1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1)
    ]
    region_size = 1  # initial size of the region - one voxel
    neighbor_points_list = []  # contains the locations of the neighbor points
    intensity_diff = 0  # stopping criteria is intensity diff
    neighbor_intensity_list = []  # contains the intensities of the neighbor points
    points_in_region = []  # contains the indices of all the points in the located region
    # initial mean of the region
    region_mean = image[seed]

    # input image parameters
    height, width, depth = image.shape
    image_size = height * width * depth
    # time
    start_time = time.time()
    # grow the region until intensity difference is greater than threshold
    while (intensity_diff < threshold) & (region_size < image_size) & (time.time() < start_time + 30):
        # loop through the neighbors
        for i in range(26):
            # get position of neighbor pixel
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]
            z_new = seed[2] + neighbors[i][2]
            # print("x_new: {} y_new: {} z_new: {}".format(x_new, y_new, z_new))

            # check if coordinates are in the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (z_new >= 0) & (x_new < height) & (y_new < width) & (
                    z_new < depth)

            if (check_inside):
                # print("output[{},{}] = {}".format(x_new, y_new, output[x_new,y_new]))
                if (x_new, y_new, z_new) not in points_in_region:
                    neighbor_points_list.append([x_new, y_new, z_new])
                    neighbor_intensity_list.append(image[x_new, y_new, z_new])
                    points_in_region.append((x_new, y_new, z_new))

        # add pixel with intensity closest to mean of region to region
        distance = abs(neighbor_intensity_list - region_mean)
        # print(distance)
        pixel_distance = min(distance)
        index = np.where(distance == pixel_distance)[0][0]
        # print(neighbor_intensity_list[index])
        points_in_region.append((seed[0], seed[1], seed[2]))
        intensity_diff = abs(image[seed[0], seed[1], seed[2]] - region_mean)
        region_size += 1
        if region_size > 4000:
            return None
        # update region mean
        region_mean = ((region_mean * region_size) + neighbor_intensity_list[index]) / (region_size + 1)
        # print("region mean: {}".format(region_mean))
        # update seed point
        seed = neighbor_points_list[index]
        # remove value from neighborhood list
        neighbor_intensity_list[index] = neighbor_intensity_list[-1]
        neighbor_points_list[index] = neighbor_points_list[-1]
    if region_size < 200:
        return None
    return points_in_region


def connected_component_analysis(binary_img):
    count = 0
    labels = measure.label(binary_img, connectivity=2, background=0)
    mask = np.zeros(binary_img.shape, dtype="uint8")

    # loop over unique components
    for label in np.unique(labels):
        if label == 0:
            continue

        # construct label mask and count number of pixels
        labelMask = np.zeros(binary_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        # print(np.unique(labelMask))
        numPixels = cv.countNonZero(labelMask)
        # print(numPixels)
        if (numPixels > 50) & (numPixels < 2000):
            count += 1
            mask = cv.add(mask, labelMask)
    return mask, count


def generate_random_seed(img):
    height, width = img.shape
    seed = (randint(0, height - 1), randint(0, width - 1), 80)
    while img[seed[0], seed[1]] == 0:
        seed = (randint(0, height - 1), randint(0, width - 1), 80)
    return seed

def find_seed(im_arr, max_intensity, center=(120, 135, 80)):
    start = center
    neighbors = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
        (0, 0, 1), (0, 0, -1), (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1),
        (0, -1, 1), (0, -1, -1), (1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1), (1, -1, 1),
        (1, -1, -1), (-1, 1, 1), (-1, 1, -1)
    ]
    v_not_found = True  # im_arr[center] # voxel intensity value
    height, width, depth = im_arr.shape
    if im_arr[center] < max_intensity:
        return center
    else:
        while v_not_found:
            pixels_viewed = []
            neighbor_intensities = []
            neighbor_pixel_list = []
            for i in range(26):
                # get position of neighbor pixel
                x_new = start[0] + neighbors[i][0]
                y_new = start[1] + neighbors[i][1]
                z_new = start[2] + neighbors[i][2]

                # check if coordinates are in the image
                check_inside = (x_new >= 0) & (y_new >= 0) & (z_new >= 0) & (x_new < height) & (y_new < width) & (
                        z_new < depth)
                if check_inside:
                    if im_arr[x_new, y_new, z_new] < max_intensity:
                        v_not_found = False
                        start = (x_new, y_new, z_new)
                        break
                    else:
                        neighbor_intensities.append(im_arr[x_new, y_new, z_new])
                        neighbor_pixel_list.append((x_new, y_new, z_new))
            temp = np.argmin(neighbor_intensities)
            # print(temp)
            if v_not_found:
                pixels_viewed.append(start)
                if pixels_viewed.count(start) > 3:
                    print('Balderdash!')
                    return None
                start = neighbor_pixel_list[temp]

    print(im_arr[start])
    return start


def sample():
    test = normalize_arr(load_image_array(0, 'data/train'))
    height, width, depth = test.shape
    output = np.zeros((height, width, depth))
    binary = convert_to_binary(test)
    mask, v_count = connected_component_analysis(binary[:, :, (int(depth / 2))])
    regions = []
    print(v_count)
    for x in range(v_count):
        seed = generate_random_seed(mask)
        region = grow_region(test, seed)
        while region is None:
            seed = generate_random_seed(mask)
            region = grow_region(test, seed)
        print(seed)
        if region not in regions:
            regions.append(region)
            for point in region:
                output[point[0], point[1], point[2]] = 255
        else:
            x -= 1
    return output


def fourplus(input_file):
    height, width, depth = input_file.shape
    output = []
    mask_list = []
    binary = convert_to_binary(input_file)
    for x in range(depth - 1):
        mask, count = connected_component_analysis(binary[:, :, x])
        if count >= 4:
            output.append(x)
            mask_list.append(mask)
    return output, mask_list


def find_center(points):
    """
    a method to find a good seed point for region growing given a set of coordinates (that constitute a blob)
    :param points: set of coordinates that constitute a ventricle in the image
    :return: center of the largest circle that can be formed within the ventricle - seed point
    """

    def closest_node(node, nodes):
        nodes = np.asarray(nodes)
        deltas = nodes - node
        dist = np.linalg.norm(deltas, axis=1)
        min_idx = np.argmin(dist)
        return nodes[min_idx], dist[min_idx], deltas[min_idx][1] / deltas[min_idx][0]

    points = np.array(points)

    vor = Voronoi(points)
    max_d = 0
    max_v = None
    for v in vor.vertices:
        _, d, _ = closest_node(v, points)
        if d > max_d:
            max_d = d
            max_v = v
    return tuple((int(max_v[0]), int(max_v[1])))


def find_centroid(point_set):
    sum_x = sum_y = 0
    length = len(point_set)
    for point in point_set:
        # if hull.find_simplex(point) >= 0:
        sum_x += point[0]
        sum_y += point[1]
    return int(sum_x / length), int(sum_y / length)


def blacklist(seed):
    if ((seed[1] in range(120 - 7, 120 + 7)) and (seed[0] in range(89 - 7, 89 + 7))) \
            or ((seed[1] in range(146 - 18, 146 + 18)) and (seed[0] in range(89 - 9, 89 + 9))):
        return True
    return False