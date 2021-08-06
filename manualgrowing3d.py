import numpy as np
import matplotlib.pyplot as plt
from extract_patches import normalize_arr, load_image_array
import time

'''
seed: (112, 135, 80)
for automatic seed selection:
    find low intensity points in the center of the image
'''
'''
(66, 134, 80)
(67, 132, 80)
(85, 91, 80)
(113, 133, 80)
test those seeds manually
'''
def region_growing_3d(image, seed, output_img, region_threshold=5):
    '''
    creating a new, segmented image with region growing algorithm
    :param image: input image
    :param seed: seed point for a region
    :return: segmented image
    '''

    '''here we set some parameters for the region growing algorithm
    neighbor points are adjacent to pixel/region'''
    neighbors = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        (1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1)
    ]
    region_threshold = region_threshold  # comparison between pixel values we use to decide whether to include in region or not
    region_size = 1  # initial size of the region - one voxel
    intensity_diff = 0  # difference in intensity between voxel
    neighbor_points_list = []  # contains the locations of the neighbor points
    neighbor_intensity_list = []  # contains the intensities of the neighbor points

    # initial mean of the region
    region_mean = image[seed]

    # input image parameters
    height, width, depth = image.shape
    image_size = height * width * depth

    # initialize output image as all black
    output = output_img
    timeout = time.time() + 30

    # grow the region until intensity difference is greater than threshold
    while (intensity_diff < region_threshold) & (region_size < image_size) & (time.time() < timeout):
        # loop through the neighbors
        for i in range(26):
            # get position of neighbor pixel
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]
            z_new = seed[2] + neighbors[i][2]
            #print("x_new: {} y_new: {} z_new: {}".format(x_new, y_new, z_new))

            # check if coordinates are in the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (z_new >= 0) & (x_new < height) & (y_new < width) & (z_new < depth)

            if (check_inside):
                #print("output[{},{}] = {}".format(x_new, y_new, output[x_new,y_new]))
                if (output[x_new, y_new, z_new] == 0):
                    neighbor_points_list.append([x_new, y_new, z_new])
                    neighbor_intensity_list.append(image[x_new, y_new, z_new])
                    output[x_new, y_new, z_new] = 255

        # add pixel with intensity closest to mean of region to region
        distance = abs(neighbor_intensity_list - region_mean)
        # print(distance)
        if len(distance) == 0:
            return output
        pixel_distance = min(distance)
        index = np.where(distance == pixel_distance)[0][0]
        #print(neighbor_intensity_list[index])
        output[seed[0], seed[1], seed[2]] = 255
        intensity_diff = abs(image[seed[0],seed[1], seed[2]] - region_mean)
        region_size += 1

        # update region mean
        region_mean = ((region_mean * region_size) + neighbor_intensity_list[index]) / (region_size + 1)
        #print("region mean: {}".format(region_mean))
        # update seed point
        seed = neighbor_points_list[index]
        # remove value from neighborhood list
        neighbor_intensity_list[index] = neighbor_intensity_list[-1]
        neighbor_points_list[index] = neighbor_points_list[-1]
    print(region_size)
    return output


#img = normalize_arr(load_image_array(0, 'data/train'))
#output_img = np.zeros_like(img)
#output = region_growing_3d(image=img, seed=(89, 154, 75), output_img=output_img)
'''output = region_growing_3d(image=img, seed=(113, 133, 80), output_img=output_img)
output = region_growing_3d(image=img, seed=(66, 134, 80), output_img=output_img)
output = region_growing_3d(image=img, seed=(85, 91, 80), output_img=output_img)
output = region_growing_3d(image=img, seed=(102, 65, 80), output_img=output_img)'''
#plt.imshow(img[:,:,80])
#plt.show()