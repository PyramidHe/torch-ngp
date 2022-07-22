import yaml
import os
import json
import numpy as np
import argparse
import glob
import cv2


def change_images_name(folder_in, folder_out, downscale=1.0, in_format='png', out_format='jpg'):
    imgs_to_read = sorted(glob.glob(os.path.join(os.path.join(folder_in, '*.'+in_format))))
    for i in range(len(imgs_to_read)):
        img = cv2.imread(imgs_to_read[i])
        width = int(img.shape[1] / downscale)
        height = int(img.shape[0] / downscale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        out_img_folder = os.path.join(folder_out, "images")
        if not os.path.exists(out_img_folder):
            # Create a new directory because it does not exist
            os.makedirs(out_img_folder)
        img_name = os.path.join(out_img_folder, "{:04d}.".format(i)+out_format)
        cv2.imwrite(img_name, img)


def scanner_to_tranforms(yaml_file, json_file):
    in_matrices = []
    ex_matrices = []
    resolutions = []
    in_matrix = np.zeros((3, 3), dtype=float)
    ex_matrix = np.zeros((4, 4), dtype=float)

    with open(yaml_file, 'r') as file:
        camera_params = yaml.load(file, Loader=yaml.Loader)
        num_cam = len(camera_params["intrinsics"])
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
    dict = {}

    for i in range(num_cam):
        if i == 0:
            width = camera_params["intrinsics"][i]["img_width"]
            height = camera_params["intrinsics"][i]["img_height"]
            fx = camera_params["intrinsics"][i]["fx"]
            fy = camera_params["intrinsics"][i]["fy"]
            dict["camera_angle_x"] = 2*np.arctan(0.5*width/fx)
            dict["camera_angle_y"] = 2*np.arctan(0.5*height/fy)
            dict["fl_x"] = fx
            dict["fl_y"] = fy
            dict["k1"] = camera_params["intrinsics"][i]["dist_k1"]
            dict["k2"] = camera_params["intrinsics"][i]["dist_k2"]
            dict["p1"] = camera_params["intrinsics"][i]["dist_px"]
            dict["p2"] = camera_params["intrinsics"][i]["dist_py"]

            
            dict["cx"] = camera_params["intrinsics"][i]["cx"]
            dict["cy"] = camera_params["intrinsics"][i]["cy"]
            dict["w"] = width
            dict["h"] = height
            dict["aabb_scale"] = 1
            dict["frames"] = []

        exdict = {}

        rot = camera_params["extrinsics"][i]["rotation"]["data"]
        translation = camera_params["extrinsics"][i]["translation"]["data"]
        for j in range(9):
            row_index = np.floor(j / 3).astype(int)
            col_index = np.floor(j % 3).astype(int)
            ex_matrix[row_index, col_index] = rot[j]

        for j in range(3):
            ex_matrix[j, 3] = translation[j]

        ex_matrix[3, :] = np.array([0., 0., 0., 1.])
        ex_matrix = np.linalg.inv(ex_matrix)
        transf = np.diag([1., -1., -1., 1.])
        #transf = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ex_matrix = ex_matrix @ transf
        exdict["file_path"] = "images/{:04d}.jpg".format(i)
        exdict["sharpness"] = .10
        exdict["transform_matrix"] = ex_matrix.tolist()
        dict["frames"].append(exdict)
        with open(json_file, "w") as outfile:
            json.dump(dict, outfile, indent=2, separators=(',', ': '))

    return in_matrices, ex_matrices, resolutions


parser = argparse.ArgumentParser()
parser.add_argument('--in_config', type=str)
parser.add_argument('--out_folder', type=str)
parser.add_argument('--img_folder', type=str)
opt = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(opt.out_folder):
        # Create a new directory because it does not exist
        os.makedirs(opt.out_folder)
        print("The new directory is created!")
    in_matrices, ex_matrices, _ = scanner_to_tranforms(opt.in_config, os.path.join(opt.out_folder, "transforms.json"))
    change_images_name(opt.img_folder, opt.out_folder, downscale=1.0)