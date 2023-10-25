from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import vamtoolbox as vam
import torch
import torch.nn as nn
import os
import io



def scale_to_fit(image_normalized, desired_height, desired_width):
    # Calculate the aspect ratio of the original image
    original_height, original_width = image_normalized.shape
    original_aspect_ratio = original_width / original_height
    desired_aspect_ratio = desired_width / desired_height

    if original_aspect_ratio > desired_aspect_ratio:
        # scale image_normalized so that the width = desired_width
        scaled_width = desired_width
        scaled_height = int(desired_width / original_aspect_ratio)
        scaled_image = cv2.resize(image_normalized, (scaled_width, scaled_height))
        buffer_image = np.zeros((desired_height, desired_width))
        start_y = (desired_height - scaled_height) // 2
        buffer_image[start_y:start_y+scaled_height, :] = scaled_image
    else:
        # scale image_normalized so that the height = desired_height
        scaled_height = desired_height
        scaled_width = int(desired_height * original_aspect_ratio)
        scaled_image = cv2.resize(image_normalized, (scaled_width, scaled_height))
        buffer_image = np.zeros((desired_height, desired_width))
        start_x = (desired_width - scaled_width) // 2
        buffer_image[:, start_x:start_x+scaled_width] = scaled_image
    return buffer_image


def saveProjections(b, dimension, save_dir: str, image_prefix: str = "image", image_type: str = ".png"):
    for k in range(len(b[0, :, 0])):
        save_path = os.path.join(save_dir, f"{image_prefix}{str(k).zfill(4)}{image_type}")
        print(f"save path = {save_path}")
        image = b[:, k, :]

        # Normalize the pixel values to the range [0, 255]
        image_normalized = (image - image.min()) * (255.0 / (image.max() - image.min()))

        # Calculate the desired dimensions with black buffer
        desired_height = dimension[0]
        desired_width = dimension[1]

        buffer_image = scale_to_fit(image_normalized, desired_height, desired_width)

        # Convert to integer type and create the image
        buffer_image = buffer_image.astype(np.uint8)
        im = Image.fromarray(buffer_image)

        im.save(save_path, subsampling=0, quality=100)
        # print(f"Saving image {str(k).zfill(4)}/{str(len(b[0])).zfill(4)}")

class Options:

    __default_FBP = {"offset":False}
    __default_CAL = {"learning_rate":0.01,"momentum":0,"positivity":0,"sigmoid":0.01}
    __default_PM = {"rho_1":1,"rho_2":1,"p":1}
    __default_OSMO = {"inhibition":0}

    def __init__(self,method : str ='CAL',n_iter : int = 50,d_h : float = 0.8,d_l : float = 0.7,filter : str ='ram-lak',units:str='normalized',**kwargs):
        """
        Parameters
        ----------

        method : str
            Type of VAM method
                - "FBP"
                - "CAL"
                - "PM"
                - "OSMO"
        
        n_iter : int
            number of iterations to perform

        d_h : float
            in-target dose constraint

        d_l : float
            out-of-target dose constraint

        filter : str
            filter for initialization ("ram-lak", "shepp-logan", "cosine", "hamming", "hanning", None)

        learning_rate : float, optional (CAL)
            step size in approximate gradient descent
        
        momentum : float, optional (CAL)
            descent momentum for faster convergence

        positivity : float, optional (CAL)
            positivity constraint enforced at each iteration
        
        sigmoid : float, optional (CAL)
            sigmoid thresholding strength
        
        rho_1 : float, optional (PM)

        rho_2 : float, optional (PM)

        p : int, optional (PM)

        inhibition : float, optional (OSMO)




        """
        self.method = method
        self.n_iter = n_iter
        self.d_h = d_h
        self.d_l = d_l
        self.filter = filter
        self.units = units
        self.__default_FBP.update(kwargs)
        self.__default_CAL.update(kwargs)
        self.__default_PM.update(kwargs)
        self.__default_OSMO.update(kwargs)
        self.__dict__.update(kwargs)  # Store all the extra variables

        self.verbose = self.__dict__.get('verbose',False)
        self.bit_depth = self.__dict__.get('bit_depth',None)
        self.exit_param = self.__dict__.get('exit_param',None)

        if method == "FBP":
            self.offset = self.__default_FBP["offset"]

        if method == "CAL":
            self.learning_rate = self.__default_CAL["learning_rate"]
            self.momentum = self.__default_CAL["momentum"]
            self.positivity = self.__default_CAL["positivity"]
            self.sigmoid = self.__default_CAL["sigmoid"]

        if method == "PM":
            self.rho_1 = self.__default_PM["rho_1"]
            self.rho_2 = self.__default_PM["rho_2"]    
            self.p = self.__default_PM["p"]        

        if method == "OSMO":
            self.inhibition = self.__default_OSMO["inhibition"]

def returnPreOptimize(target_geo : vam.geometry.TargetGeometry,proj_geo : vam.geometry.ProjectionGeometry,options:Options):
    if options.units != "normalized" or proj_geo.absorption_coeff is not None:
        proj_geo.calcAbsorptionMask(target_geo)

    if options.method == "OSMO":
        A = vam.projectorconstructor.projectorconstructor(target_geo,proj_geo)
        
        # the first model is just the target
        x_model = np.copy(target_geo.array) 



        target_filtered = vam.util.data.filterTargetOSMO(target_geo.array,options.filter)
        x_model = np.real(target_filtered)

        # the initial sinogram is just the forward projection of the model
        b = A.forward(x_model)
        b = np.clip(b,0,None)
        # x = A.backward(b)
        # x = x/np.amax(x)
        return b
    
def generatedata(dir_list, resolution, angles, optimizer_params, dimension, OutputFolder, PreProjectionFolder):
    for i in range(len(dir_list)):
        print("optimizing: " + dir_list[i])
        targetgeo = vam.geometry.TargetGeometry(stlfilename=dir_list[i], resolution=resolution)
        
        # targetgeo.show()

        # Set output name to match the STL input file
        OutputPrefix = os.path.splitext(os.path.split(dir_list[i])[1])[0]
        print(os.path.split(dir_list[i]))

        # Can we put this in calculated settings?
        proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type='parallel', CUDA=True)

        b = returnPreOptimize(targetgeo, proj_geo, optimizer_params)
        opt_sino, opt_recon, error = vam.optimize.optimize(targetgeo, proj_geo, optimizer_params)

        opt_sino.show()
        opt_recon.show()
        slice_2d = b[:, 180, :].T  # Selecting the first 100 rows from the second dimension
        plt.imshow(slice_2d)
        plt.colorbar()  # Optional: Add a colorbar to the plot
        plt.show()

        # Resize opt_sino.array with black buffer
        # desired_height = dimension[0]
        # desired_width = dimension[1]
        
        # Save images
        print("output folder:", OutputFolder)
        print("output prefix:", OutputPrefix)
        saveProjections(b, dimension, save_dir=OutputFolder, image_prefix=OutputPrefix)
        saveProjections(opt_sino.array, dimension, save_dir=PreProjectionFolder, image_prefix=OutputPrefix)
        # print("sino size:" + str(opt_sino.array.shape))
        del targetgeo
        del proj_geo
        del b
        del opt_sino
        del opt_recon
        del slice_2d

########################################################################
# AI part
########################################################################



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.middle_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.decoder = nn.Sequential(
            
            nn.Conv2d(192, 64, kernel_size=3, padding=1),  # Concatenating feature maps from the encoder
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        encoder_output1 = self.encoder(x)
        encoder_output2 = self.middle_conv(encoder_output1)
        decoder_input = torch.cat([encoder_output2, encoder_output1], dim=1)  # Concatenate feature maps
        decoder_output = self.decoder(decoder_input)
        return decoder_output
  
    




# Function to load grayscale images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

def generatedata_ML(model_path, pre_optimization_folder, OutputFolder, batch_size, use_cpu = False):

    model_buffer = io.BytesIO(open(model_path, 'rb').read())
    model = UNet()
    if use_cpu:
        model.load_state_dict(torch.load(model_buffer, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_buffer))
    model.eval()
    
    # Load pre-optimization grayscale images
    pre_optimization_images = load_images_from_folder(pre_optimization_folder)

    # Convert image lists to numpy arrays
    pre_optimization_images = np.array(pre_optimization_images)

    # Normalize the image data
    pre_optimization_images = pre_optimization_images / 255.0

    # Convert numpy arrays to PyTorch tensors
    x_train = torch.from_numpy(pre_optimization_images).unsqueeze(1).float()

    
    device_count = torch.cuda.device_count()
    if device_count > 0:
        print("Available GPUs:")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available. PyTorch is using CPU.")

    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model = model.to(device)
        print("PyTorch is using GPU.")
    else:
        print("PyTorch is using CPU.")

    # Convert the data to the appropriate device
    x_train = x_train.to(device)

    # Create a directory to save the output images
    if not os.path.exists(OutputFolder):
        os.makedirs(OutputFolder)

    # Function to save grayscale images
    def save_images(images, path, prefix="output_", file_extension=".png"):
        for i, image in enumerate(images):
            filename = os.path.join(path, f"{prefix}{i:04d}{file_extension}")
            cv2.imwrite(filename, (image * 255).astype(np.uint8))

    # Create an empty list to store the output images
    output_images = []

    # Iterate through your input images in batches
    for i in range(0, len(x_train), batch_size):
        batch = x_train[i:i + batch_size]
        
        # Make predictions using the U-Net model
        with torch.no_grad():
            predicted_images = model(batch)

        # Convert the predicted images back to NumPy arrays
        predicted_images = predicted_images.squeeze(1).cpu().numpy()

        # Append the predicted images to the list
        output_images.extend(predicted_images)

    # Save the output images
    save_images(output_images, OutputFolder)





    
################################################################
# Accuracy measurenment
################################################################
from scipy.ndimage import zoom

def load_png_to_numpy_array(folder_path, rotate=False):

    print(f"loading images from folder {folder_path}")

    image_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder_path, filename))
            image_list.append(np.array(img))



    images_array = np.array(image_list)

    if rotate:
        # Rotate every image in the array by 90 degrees
        rotated_images_array = np.array([np.rot90(img) for img in images_array])


    plt.imshow(rotated_images_array[0])
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()
    print("showing first loaded image")
    return rotated_images_array


def expand_3d_numpy_array(original_array, target_shape):
    

    # Assuming your original 3D array is called original_array with shape (115, 115, 128)
    original_shape = original_array.shape

    

    # Calculate the scaling factors for each dimension
    scaling_factors = (
        target_shape[0] / original_shape[0],
        target_shape[1] / original_shape[1],
        target_shape[2] / original_shape[2]
    )

    # Use the zoom function to interpolate the original array to the target shape
    interpolated_array = zoom(original_array, zoom=scaling_factors, order=1)
    return interpolated_array

def inverse_radon(projections, STLfile_name, resolution):
    num_angles = 360
    angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
    proj_geo = vam.geometry.ProjectionGeometry(angles,ray_type='parallel',CUDA=True)
    target_geo = vam.geometry.TargetGeometry(stlfilename=STLfile_name, resolution=resolution)

    # Define the target shape (128, 128, 128)
    dims = projections.shape
    target_shape = (dims[1], dims[1], dims[2])
    target_geo.array = expand_3d_numpy_array(target_geo.array, target_shape)
    
    target_geo.nX = target_shape[0]
    target_geo.nY = target_shape[1]


    from vamtoolbox.projector.Projector3DParallelCUDA import Projector3DParallelCUDAAstra
    A = Projector3DParallelCUDAAstra(target_geo,proj_geo)
    tmp_projections = np.transpose(projections,(2,0,1))

    x = A.backward(tmp_projections)
    x = x/np.amax(x) # normalize

    # print(f"x shape : {x.shape}")

    # _error = vam.metrics.calcVER(target_geo,x)
    return x

# DEFINE ERROR FUNCTIONS HERE

d_l = 0.6
d_h = 0.85

def squared_difference(array1, array2):
    return (array1 - array2) ** 2

def MSE(array1, array2):
    squared_diff = squared_difference(array1, array2)
    mse = squared_diff.mean()
    return mse

def round_3d_array(array, round_down_threshold, round_up_threshold):
    rounded_array = [[[0 if value <= round_down_threshold else 1 if value >= round_up_threshold else 0.5 for value in row] for row in plane] for plane in array]
    return rounded_array

def L1_dif(array1, array2):
    # Calculate the L1 norm (sum of absolute differences)
    l1_norm = np.sum(np.abs(array1 - array2))
    return l1_norm

def count_above_dif_threshold(array1, array2, threshold):
    # Calculate absolute differences between the two arrays
    abs_diff = np.abs(array1 - array2)
    
    # Create a mask where values are above the threshold
    above_threshold_mask = abs_diff > threshold
    
    # Use the mask to filter the absolute differences and then sum them
    return np.sum(above_threshold_mask)


# dif = squared_difference(reconstruction_ML, ground_truth)

def compare(array1, array2):
    
    mse = MSE(array1, array2)
    l1 = L1_dif(array1, array2)

    threshold = 0.1
    count = count_above_dif_threshold(array1, array2, threshold)

    array2_descretized = np.array(round_3d_array(array2, round_down_threshold=d_l, round_up_threshold=d_h))

    solidified_voxel_differences = count_above_dif_threshold(array1, array2, threshold)

    # calcualte percentage eerors assuming array1 an darray2 have same number of voxels
    total_pixels = array1.shape[0] * array1.shape[1] * array1.shape[2]

    print(f"MSE: {mse}, L1: {l1}, % of different voxels above {threshold}: {count/total_pixels} solidified voxel difference to STL: {solidified_voxel_differences}")



import matplotlib.pyplot as plt
from tqdm import tqdm
# Calculate optimal rotations

import vedo
    
def plot(data, d_l=0.1, d_h=1):
    vol = vedo.Volume(data).legosurface(vmin=d_l,vmax=d_h)
    vol.show(viewup="x")

def calculate_optimal_rotations(ground_truth, predicted, error_function, max_iterations=5, initial_range=[0, 4], resolution=6, name="undefined"):
    best_error = float('inf')
    best_rotations = 0
    error = []
    rotations_number = []

    for iteration in range(max_iterations):
        # Sample 'resolution' number of points in the interval 'initial_range'
        step_size = (initial_range[1] - initial_range[0]) / (resolution - 1)
        print(f"checking {initial_range[0]} to {initial_range[1]}")
        for i in range(resolution):
            temp = (initial_range[0] + i * step_size) * predicted
            error.append(error_function(ground_truth, temp))
            rotations_number.append((initial_range[0] + i * step_size))

        # Find the index of the minimum error in the current window
        min_error_idx = error.index(min(error))

        # Update the best error and rotations
        if error[min_error_idx] < best_error:
            best_error = error[min_error_idx]
            best_rotations = rotations_number[min_error_idx]

        # Update the search window for the next iteration
        # Ensure the range is centered around the minimum error point
        center = rotations_number[min_error_idx]
        width = (initial_range[1] - initial_range[0]) / 4
        initial_range = [center - width, center + width]

    # Create a figure and plot all data points
    sorted_lists = zip(rotations_number, error)
    sorted_lists = sorted(sorted_lists, key=lambda x: x[0])
    rotations_number, error = zip(*sorted_lists)
    plt.figure(figsize=(10, 6))
    plt.plot(rotations_number, error, marker='o')
    plt.title(f'Error vs. Rotation Factor for {name}')
    plt.xlabel('Rotation Factor')
    plt.ylabel('Error')
    plt.grid(True)

    plt.axvline(x=best_rotations, color='r', linestyle='--', label='Minimum Error')

    plt.legend()
    plt.show()

    return best_rotations


