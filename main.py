import math
from geojson import Polygon
from autoseed import fourplus, blacklist
from compute_vertices import hull_vertices
from extract_patches import normalize_arr, load_image_array
import numpy as np
from skimage import measure
import itertools
import nibabel as nib
from manualgrowing3d import region_growing_3d
import matplotlib.pyplot as plt
from polylabel import polylabel

def main():
    test = normalize_arr(load_image_array(0, 'data/train'))
    slices, masks = fourplus(test)
    outputs = []
    print(slices)
    for ind, mask in enumerate(masks):
        output = np.zeros_like(test)
        labels, num = measure.label(mask, connectivity=2, background=0, return_num=True)
        v_labels = (np.unique(labels))[1:]
        print(v_labels)
        for label in v_labels:
            rows, cols = np.where(labels == label)
            point_set = np.asarray(list(itertools.zip_longest(rows, cols)), dtype=float)
            vertices = [set(hull_vertices(point_set))]
            ccw_vertices = (sorted(vertices[0], key=lambda p: math.atan2(p[1], p[0])))
            ccw_vertices = list(ccw_vertices)
            polygon = Polygon([ccw_vertices])
            verts = polygon["coordinates"]
            centroid = polylabel(polygon=verts, precision=5.0, debug=True, with_distance=False)
            print('centroid: {} on slice {}'.format(centroid, slices[ind]))
            centroid = (int(centroid[0]), int(centroid[1]), int(slices[ind]))
            if not blacklist(centroid):
                output = region_growing_3d(image=test, seed=centroid, output_img=output)
            else:
                print("Centroid blacklisted")
            print(label)
        outputs.append(output)
    #return outputs
    final_img = outputs[5]
    #plt.imshow(final_img[:,:,80])
    #plt.show()

    test_img = nib.load('data/train/0402/training04_02_flair_pp.nii.gz')
    final_img = (nib.Nifti1Image(final_img, test_img.affine))
    # print(type(output_img))
    nib.save(final_img, 'flair_segmentation.nii')


if __name__ == "__main__":
    main()

#outputs[5][:,:,80]