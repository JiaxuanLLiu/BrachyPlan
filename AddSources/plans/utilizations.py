import SimpleITK as sitk
import numpy as np
from . import geometry
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import scipy
import torch
import torch.optim as optim
from . import fitting_model
from . import visualizer
import math
import vtk
import os
import copy


def create_folder_if_not_exists(folder_path):
    """
    Create a folder if it does not already exist.

    Parameters:
        folder_path (str): The path of the folder to be created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")
        
        
def normalize_dose_image(image, window_min, window_max, output_min, output_max): 
    """
    Normalize the pixel values of an image to a specified range.

    Parameters:
        image (SimpleITK.Image): The input image to be normalized.
        min_val (float): The minimum value of the normalized range.
        max_val (float): The maximum value of the normalized range.

    Returns:
        SimpleITK.Image: The normalized image.
    """
    ww_filter = sitk.IntensityWindowingImageFilter()
    ww_filter.SetWindowMinimum(window_min)
    ww_filter.SetWindowMaximum(window_max)
    ww_filter.SetOutputMinimum(output_min)
    ww_filter.SetOutputMaximum(output_max)
    processed_image = ww_filter.Execute(image)
    return processed_image


def normalize_dose_array(array, min_val, max_val, scale=255): 
    """
    Normalize a numerical array to a specified range.

    This function normalizes the values of a numerical array based on a given minimum 
    and maximum range, and scales them to a target range (default is [0, 255]).

    Parameters:
        array (array-like): 
            Input numerical array (e.g., NumPy array or list) to be normalized.
        min_val (float): 
            The minimum value of the input range.
        max_val (float): 
            The maximum value of the input range.
        scale (float, optional): 
            The scaling factor for normalization. Default is 255.

    Returns:
        np.ndarray: 
            A normalized NumPy array with values scaled to the specified range.

    Example:
        >>> import numpy as np
        >>> data = np.array([10, 20, 30, 40, 50])
        >>> normalized_data = normalize_dose_array(data, 10, 50, scale=1)
        >>> print(normalized_data)
        [0.   0.25 0.5  0.75 1.  ]
    """
    return (np.array(array) - min_val) * (scale / (max_val - min_val))


def position_transform(image, coords):
    """
    Transform voxel (array) coordinates to physical (world) coordinates for a position.
    
    This function serves as a wrapper around `voxel_to_world` to specifically 
    handle the transformation of **position** coordinates from voxel space to physical space.
    
    Parameters:
        - image: SimpleITK.Image object  
            The medical image containing metadata for spacing, origin, and direction.
        - coords: list or numpy array  
            The voxel coordinates representing a position. Can be a single point (shape: (3,)) 
            or multiple points (shape: (n, 3)).
    
    Returns:
        - world_coords: numpy array  
            The transformed physical coordinates corresponding to the input voxel coordinates.
    
    Notes:
        - Position transformation considers both spacing and origin.
        - It assumes that `voxel_to_world` correctly handles coordinate order and direction.
    """
    # Transform voxel coordinates into physical coordinates, including origin offset.
    return geometry.voxel_to_world(image, coords, True)


def direction_transform(image, direc):
    """
    Transform voxel direction vectors to physical direction vectors.
    
    This function serves as a wrapper around `voxel_to_world` to specifically 
    handle the transformation of **direction** vectors from voxel space to physical space.
    
    Parameters:
        - image: SimpleITK.Image object  
            The medical image containing metadata for spacing, origin, and direction.
        - direc: list or numpy array  
            The voxel coordinates representing a direction vector. Can be a single vector (shape: (3,)) 
            or multiple vectors (shape: (n, 3)).
    
    Returns:
        - world_coords: numpy array 
            The transformed physical direction vectors corresponding to the input voxel coordinates.
    
    Notes:
        - Direction transformation involves scaling by spacing but **does not include the origin offset**.
        - Ensure that `voxel_to_world` correctly distinguishes direction vectors from position coordinates.
    """
    # Transform the voxel direction vector(s) into physical space using a coordinate transformation.
    direction = geometry.voxel_to_world(image, direc, False)
    
    # Normalize the resulting direction vector to ensure it has a unit length.
    direction = direction / np.linalg.norm(direction)
    
    # Return the normalized physical direction vector.
    return direction


def read_nii_image(path):
    """
    Reads a medical image volume from the given file path.
    
    Parameters:
        path (str): The file path to the medical image (e.g., .nii, .nii.gz, etc.).
        
    Returns:
        SimpleITK.Image: The loaded medical image volume as a SimpleITK Image object.
    """
    volume = ImageResample_size(sitk.ReadImage(path))  # Reads the image volume from the specified path
    return volume


def get_reference_direction(radiation_array, target_value):
    """
    Calculate the reference direction for a given radiation array using Principal Component Analysis (PCA).

    This function identifies the primary direction of a specified target area within a radiation pattern, 
    denoted by the `target_value`. It performs PCA on the coordinates of this target area to determine 
    the principal component direction.

    Parameters:
    radiation_array (array-like): The input array representing radiation data.
    target_value (int or float): The value in the array representing the target area of interest.

    Returns:
    np.ndarray: A unit vector representing the primary principal direction of the target area.

    Raises:
    AssertionError: If the target value does not exist within the radiation array.

    Note:
    Ensure that the input array and target value are correctly specified. 
    The function assumes that the target areas are contiguous or constitute a meaningful region for directional analysis.
    """

    # Check for presence of the target value within the array
    assert np.any(radiation_array == target_value), 'No target area in your input array'
    
    # Obtain the coordinates where the radiation array equals the target value
    coordinates = np.argwhere(radiation_array == target_value)

    # Perform PCA to determine the principal direction vector
    pca = PCA()
    pca.fit(coordinates)
    
    # Extract and normalize the primary principal component
    direction_vector = pca.components_[0]
    return direction_vector / np.linalg.norm(direction_vector)


def volume2array(path):
    """
    Converts a medical image volume to a NumPy array.
    
    Parameters:
        path (str): The file path to the medical image.
        
    Returns:
        numpy.ndarray: The medical image volume converted into a NumPy array format.
    """
    return sitk.GetArrayFromImage(ImageResample_size(read_nii_image(path), is_label=True))  # Converts the volume to a NumPy array


def get_planning_volume_array(target_volume_path, target_value, obstacle_value, background_value):
    """
    Generate a radiation planning volume array by processing the target volume.

    This function loads a target volume, applies thresholding, and assigns specific values to 
    target, obstacle, and background regions based on predefined criteria. The final array 
    represents the radiation planning volume.

    Args:
        target_volume_path (str): 
            Path to the target volume file (e.g., NIfTI or other supported formats).
        target_value (int or float): 
            Value assigned to voxels identified as target regions.
        obstacle_value (int or float): 
            Value assigned to voxels identified as obstacle regions.
        background_value (int or float): 
            Value assigned to voxels identified as background regions.

    Returns:
        ndarray: 
            A NumPy array representing the radiation planning volume, with target, obstacle, 
            and background regions labeled according to the specified values.
    """
    
    # Step 1: Load the target volume into a NumPy array
    # tv_array = volume2array(target_volume_path)
    tv_array = sitk.GetArrayFromImage(target_volume_path)
    # commented out for official use
    # tv_array[tv_array>0.5] = target_value
    # tv_array[tv_array<0.5] = background_value
    
    # Step 2: Classify regions in the target volume based on their values:
    # - Voxels matching the target value are kept as target regions.
    # - Voxels matching the obstacle value are kept as obstacle regions.
    # - All other voxels are classified as background regions.
    tv_array[tv_array>obstacle_value] = obstacle_value
    tv_array[(tv_array != target_value) & (tv_array != obstacle_value)] = background_value
    # tv_array[(tv_array != target_value) & (tv_array != background_value)] = obstacle_value
    # new_tv = sitk.GetImageFromArray(tv_array)
    # sitk.WriteImage(new_tv, 'new_tv.nii')
    # Step 3: Return the processed radiation planning volume array
    return tv_array


def ImageResample_size(sitk_image, new_size=[128, 128, 128], is_label=False):
    """
    Resample a SimpleITK image to a new size.

    This function resamples a given SimpleITK image to a specified new size, adjusting the spacing accordingly.
    It can handle both label images and regular images by using different interpolation methods.

    Parameters:
    - sitk_image (SimpleITK.Image): The input image to be resampled.
    - new_size (list of int): The desired size of the output image [x, y, z].
    - is_label (bool): If True, use nearest neighbor interpolation (suitable for label images).
                       If False, use linear interpolation (suitable for regular images).

    Returns:
    - SimpleITK.Image: The resampled image with the specified new size.
    """
    size = np.array(sitk_image.GetSize())  # Original size of the image
    spacing = np.array(sitk_image.GetSpacing())  # Original spacing of the image
    new_size = np.array(new_size)  # Desired new size

    # Calculate the new spacing based on the new size
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    # Set up the resample filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    # Choose the appropriate interpolator
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    # Execute the resampling and return the new image
    newimage = resample.Execute(sitk_image)

    # Save the resampled image as a .nii.gz file
    # sitk.WriteImage(newimage, 'resized_img.nii.gz')

    return newimage


def cal_next_seed_pos_direc(mask_volume, cur_radiation, in_lowest_dose, single_seed_radiations, seed_sigma, seed_avr_dose, direc_res):
    """
    Calculate the next seed position and its orientation direction for placement in a 3D radiation volume.

    The function determines the next seed's location by identifying under-irradiated regions 
    based on the current radiation profile. It then calculates the optimal orientation direction 
    for the seed placement.

    Parameters:
        mask_volume (numpy.ndarray): A 3D binary array representing the target volume for radiation. 
                                      Values of 1 indicate regions to be irradiated, and 0 indicates non-target regions.
        cur_radiation (numpy.ndarray): A 3D array representing the current radiation coverage.
                                       Values represent radiation levels, with 0 indicating no radiation.
        in_lowest_dose (float): The threshold dose used to identify under-irradiated regions (regions with radiation < this value).
        single_seed_radiations (list of ndarray): A list of radiation fields contributed by previously placed seeds.
        seed_sigma (float): Standard deviation of the radiation spread (Gaussian distribution) for each seed.
        seed_avr_dose (float): The average dose delivered by a single seed.
        direc_res (tuple): A resolution parameter that defines the discretization of candidate directions (e.g., radial, azimuthal, and polar angles).

    Returns:
        tuple: A tuple containing:
            - pos (tuple): Normalized coordinates (x, y, z) of the next seed position within the volume.
                           If no radiation has been applied, returns the center of the largest unirradiated region.
                           Otherwise, returns the position with the lowest radiation coverage in the masked target region.
            - direc (numpy.ndarray): A unit vector representing the optimal orientation direction for placing the seed.
            - cur_seed_radiation (numpy.ndarray): The radiation field contributed by the newly placed seed.
            - updated_radiation (numpy.ndarray): The updated cumulative radiation coverage after placing the seed.
            - cur_DVH_rate (float): The Dose-Volume Histogram (DVH) rate for the current radiation distribution.
    """
    
    # Step 1: Calculate the next seed position based on the current radiation coverage and the target volume mask.
    # This function identifies regions with insufficient radiation and returns the position with the lowest radiation coverage.
    pos = cal_next_seed_pos(mask_volume, cur_radiation, in_lowest_dose)
    
    # Step 2: Calculate the optimal orientation direction for the seed based on the current radiation and target mask.
    # This function returns the best direction for placing the seed at the calculated position.
    direc = cal_next_seed_direc(mask_volume, cur_radiation, pos)
    
    # Step 3: Normalize the seed position to the dimensions of the target volume (mask_volume).
    # The position is normalized to the range [0, 1] based on the size of the mask_volume.
    pos = np.array([pos[0]/mask_volume.shape[0], pos[1]/mask_volume.shape[1], pos[2]/mask_volume.shape[2]])
    
    # Step 4: Place the seed at the calculated position and evaluate the resulting radiation coverage.
    # The function tests multiple possible orientations and selects the best direction for maximum DVH rate.
    best_direc, cur_seed_radiation, cur_radiation, cur_DVH_rate = place_and_evaluate_seed(
            pos, direc, cur_radiation, mask_volume, in_lowest_dose,
            single_seed_radiations, seed_sigma, seed_avr_dose, direc_res)

    # Step 5: Return the normalized seed position, the best orientation direction, and updated radiation information.
    return pos, best_direc, cur_seed_radiation, cur_radiation, cur_DVH_rate


def cal_next_seed_pos_direc_v2(dose_image, dose_cal_model, mask_volume, cur_radiation, in_lowest_dose, single_seed_radiations, direc_res):
    """
    Calculate the next seed position and its orientation direction for placement in a 3D radiation volume.

    The function determines the next seed's location by identifying under-irradiated regions 
    based on the current radiation profile. It then calculates the optimal orientation direction 
    for the seed placement.

    Parameters:
        dose_image (numpy.ndarray): A 3D array representing the dose image, containing radiation dose information.
        dose_cal_model (object): The model used to calculate the dose distribution from a seed.
        mask_volume (numpy.ndarray): A 3D binary array representing the target volume for radiation. 
                                      Values of 1 indicate regions to be irradiated, and 0 indicates non-target regions.
        cur_radiation (numpy.ndarray): A 3D array representing the current radiation coverage.
                                       Values represent radiation levels, with 0 indicating no radiation.
        in_lowest_dose (float): The threshold dose used to identify under-irradiated regions (regions with radiation < this value).
        single_seed_radiations (list of numpy.ndarray): A list of radiation fields contributed by previously placed seeds.
        direc_res (tuple): A resolution parameter that defines the discretization of candidate directions (e.g., radial, azimuthal, and polar angles).

    Returns:
        tuple: A tuple containing:
            - pos (tuple): Normalized coordinates (x, y, z) of the next seed position within the volume.
                           If no radiation has been applied, returns the center of the largest unirradiated region.
                           Otherwise, returns the position with the lowest radiation coverage in the masked target region.
            - direc (numpy.ndarray): A unit vector representing the optimal orientation direction for placing the seed.
            - cur_seed_radiation (numpy.ndarray): The radiation field contributed by the newly placed seed.
            - updated_radiation (numpy.ndarray): The updated cumulative radiation coverage after placing the seed.
            - cur_DVH_rate (float): The Dose-Volume Histogram (DVH) rate for the current radiation distribution.
    """
    
    # Step 1: Calculate the next seed position based on the current radiation coverage and the target volume mask.
    # This step identifies regions that have not received sufficient radiation (below the threshold).
    # The function `cal_next_seed_pos` returns the position with the lowest radiation coverage, 
    # or the largest unirradiated region.
    pos = cal_next_seed_pos(mask_volume, cur_radiation, in_lowest_dose)
    
    # Step 2: Calculate the optimal orientation direction for the seed based on the current radiation and target mask.
    # The `cal_next_seed_direc` function determines the best direction for the seed placement at the calculated position.
    direc = cal_next_seed_direc(mask_volume, cur_radiation, pos)
    
    # Step 3: Normalize the seed position to the dimensions of the target volume (mask_volume).
    # Normalize the position so that it fits into the range [0, 1] according to the dimensions of the mask volume.
    pos = np.array([pos[0]/mask_volume.shape[0], pos[1]/mask_volume.shape[1], pos[2]/mask_volume.shape[2]])
    
    # Step 4: Place the seed at the calculated position and evaluate the resulting radiation coverage.
    # This function, `place_and_evaluate_seed_v2`, tests multiple possible orientations and selects 
    # the best direction that maximizes the Dose-Volume Histogram (DVH) rate.
    best_direc, cur_seed_radiation, cur_radiation, cur_DVH_rate = place_and_evaluate_seed_v2(
        dose_image, dose_cal_model, pos, direc, cur_radiation, mask_volume, in_lowest_dose, single_seed_radiations, direc_res,
    )

    # Step 5: Return the normalized seed position, the best orientation direction, and updated radiation information.
    # The return values include the normalized seed position, the optimal orientation direction, 
    # the radiation field contributed by the newly placed seed, the updated radiation field, and the DVH rate.
    return pos, best_direc, cur_seed_radiation, cur_radiation, cur_DVH_rate


def cal_next_seed_pos(radiation_volume, cur_radiation, lowest_dose):
    """
    Calculate the next seed position in a 3D radiation volume based on the current radiation coverage.

    Parameters:
        radiation_volume (numpy.ndarray): 3D array representing the target volume for radiation, 
                                          where 1 indicates regions to be irradiated and 0 indicates non-target regions.
        cur_radiation (numpy.ndarray): 3D array representing the current radiation coverage, 
                                       where values represent radiation levels, and 0 indicates no radiation.
        lowest_dose (float): Threshold dose to identify areas with insufficient radiation coverage.

    Returns:
        tuple: The coordinates (x, y, z) of the next seed position. 
               If `cur_radiation` is entirely zeros, it returns the center of the largest unirradiated region (an isolated area).
               Otherwise, it returns the coordinate of the minimum value in the masked radiation coverage.
    """
    
    # Create a mask volume that marks target areas in `radiation_volume` with 1, others with 0
    mask_volume = (radiation_volume == 1).astype(float)

    # Case 1: If there is no existing radiation coverage
    if np.all(cur_radiation == 0):
        # Use the mask to find the center of the largest unirradiated region
        mask_radiation = mask_volume
        # Return the center of the largest "island" of unirradiated region based on adaptive sigma (from an external function)
        return geometry.find_island_center_adaptive_sigma(mask_radiation)[1]
    
    # Case 2: If there is some radiation coverage
    else:
        # Apply the radiation mask to the target volume, retaining only target regions
        # Set areas that have sufficient radiation (above `lowest_dose`) or non-target areas to a high placeholder value (1e5)
        mask_radiation = mask_volume * cur_radiation
        mask_radiation[(mask_radiation > lowest_dose) | (mask_radiation == 0)] = 1e5
        
        # Find the coordinates of the minimum value in `mask_radiation`, indicating the location with the least radiation
        # within the target area
        return np.unravel_index(np.argmin(mask_radiation), mask_radiation.shape)


def cal_next_seed_direc(radiation_volume, cur_radiation, pos):
    """
    Calculate the direction vector for the next radiation seed placement based on the current radiation
    coverage and target volume.

    Parameters:
        radiation_volume (numpy.ndarray): 3D array representing the target area for radiation.
                                          Values are 1 for target areas, 0 otherwise.
        cur_radiation (numpy.ndarray): 3D array of the same shape as radiation_volume showing current
                                       radiation coverage (0 for no radiation, >0 for irradiated areas).
        pos (tuple): Current position in the 3D array where a new seed is being considered.
        lowest_dose (float): Threshold dose for minimal acceptable radiation coverage.

    Returns:
        numpy.ndarray: Normalized direction vector for the next radiation seed.
    """

    # Create a binary mask of the target volume
    mask_volume = (radiation_volume == 1).astype(float)

    # Case 1: If there is no radiation coverage at all
    if np.all(cur_radiation == 0):
        # Find all coordinates in the target volume
        coordinates = np.argwhere(mask_volume > 0)

        # Perform PCA to determine the principal direction vector
        pca = PCA()
        pca.fit(coordinates)
        
        # Extract and normalize the primary principal component
        direction_vector = pca.components_[0]
        return direction_vector / np.linalg.norm(direction_vector)

    # Case 2: If there is existing radiation coverage
    else:
        # Apply the current radiation mask, setting non-target areas or those above lowest_dose to a high placeholder
        mask_radiation = mask_volume * cur_radiation
        direc = geometry.find_min_absolute_gradient_direction(mask_radiation, pos)
        direc = direc / np.linalg.norm(direc)
        return direc
      
        # # Apply the current radiation mask, setting non-target areas or those above lowest_dose to a high placeholder
        # mask_radiation = mask_volume * cur_radiation
        
        # # Find coordinates within the target volume with radiation below the lowest dose threshold
        # coordinates = np.argwhere((mask_volume > 0) & (mask_radiation <= lowest_dose))
        
        # # Ensure the specified position `pos` is within the under-radiated area
        # target_index = np.where((coordinates == pos).all(axis=1))[0]
        # assert target_index.size > 0, "Position not found in the specified region."

        # # Cluster under-radiated points using DBSCAN and find the cluster for the target position
        # y_pred = DBSCAN(eps=1).fit_predict(coordinates)
        # target_cluster_indices = np.where(y_pred == y_pred[target_index[0]])[0]
        # target_cluster = coordinates[target_cluster_indices]

        # # Perform PCA on the points within the target cluster to find the main direction vector
        # pca = PCA()
        # pca.fit(target_cluster)

        # # Extract and normalize the primary principal component
        # direction_vector = pca.components_[0]
        # return direction_vector / np.linalg.norm(direction_vector)


def get_cone(dire, angle, r_resolution, c_resolution):
    """
    Generates a cone of unit vectors around a specified central direction vector.
    
    Parameters:
        dire (numpy.ndarray): The central direction vector (3D) around which the cone is generated.
        angle (float): The angle of the cone in radians.
        r_resolution (int): Radial resolution - the number of points to sample from the center to the cone edge.
        c_resolution (int): Circumferential resolution - the number of rotations around the central direction.

    Returns:
        list of numpy.ndarray: A list of 3D unit vectors representing points on the cone.
    """
    # Generate an orthogonal direction to `dire` to form the initial radial vector
    orth_dir = geometry.perpendicular_vector(dire)
    
    # Calculate the radius based on the specified cone angle
    radius = np.tan(np.deg2rad(angle))
    
    # Rotation matrix to rotate around `dire` in `c_resolution` steps
    rot_mtx = scipy.linalg.expm(np.cross(np.eye(3), dire / np.linalg.norm(dire) * 2 * np.pi / c_resolution))
    
    # Start cone with the central direction
    cone = [dire]
    
    # Generate the cone's vectors
    for _ in range(c_resolution):
        # Rotate the orthogonal direction around `dire`
        orth_dir = np.dot(rot_mtx, orth_dir)
        
        # Add vectors moving radially from `dire` outwards to the edge of the cone
        for j in range(1, r_resolution + 1):
            n_dir = dire + orth_dir * radius * (j / r_resolution)
            n_dir = n_dir / np.linalg.norm(n_dir)  # Normalize to keep it a unit vector
            cone.append(n_dir)
    
    return cone


def simple_single_dose_calculation(shape, pos, direc, seed_sigma, seed_avr_dose):
    """
    Calculate a simple single-dose distribution based on a 3D Gaussian model.

    Parameters:
        shape (tuple): The shape of the output 3D array, typically specified as (depth, height, width).
        pos (tuple): The center position of the dose distribution in 3D space (z, y, x).
        seed_sigma (tuple): Standard deviations for the Gaussian distribution along each axis (z, y, x), 
                            representing the spread or "size" of the dose.
        direc (tuple): A 3D direction vector specifying the orientation of the dose distribution.

    Returns:
        numpy.ndarray: A 3D array representing the single-dose distribution with the specified orientation and spread.
    """
    pos = np.array(pos).reshape(-1)*np.array(shape).reshape(-1)
    
    return seed_avr_dose*geometry.generate_oriented_3d_gaussian(shape, pos, direc, seed_sigma)


def single_dose_calculation_v2(pos, direc, dose_image, dose_cal_model):
    """
    Calculate a simple single-dose distribution based on a 3D Gaussian model.

    Parameters:
        shape (tuple): The shape of the output 3D array, typically specified as (depth, height, width).
        pos (tuple): The center position of the dose distribution in 3D space (z, y, x).
        seed_sigma (tuple): Standard deviations for the Gaussian distribution along each axis (z, y, x), 
                            representing the spread or "size" of the dose.
        direc (tuple): A 3D direction vector specifying the orientation of the dose distribution.

    Returns:
        numpy.ndarray: A 3D array representing the single-dose distribution with the specified orientation and spread.
    """
    image_array = sitk.GetArrayFromImage(dose_image)
    image_shape, image_spascing, image_origin = image_array.shape, dose_image.GetSpacing(), dose_image.GetOrigin()

    pos = np.array(pos).reshape(-1)*np.array(image_shape).reshape(-1) + np.array(image_origin).reshape(-1)
    
    soft_treatment_plan =  position_soft_method(pos, image_origin, image_shape, image_spascing)
    x_a, y_a = geometry.get_x_y_angle(direc)
    points_pos_angle = np.array([pos[0], pos[1], pos[2], x_a, y_a])
    line_map = line_source_map(points_pos_angle, image_origin, image_shape, image_spascing)
    device = next(dose_cal_model.parameters()).device
    
    train_image = torch.FloatTensor(image_array).unsqueeze(0).to(device).unsqueeze(0)
    train_label = torch.FloatTensor(soft_treatment_plan).unsqueeze(0).to(device).unsqueeze(0)
    train_map = torch.FloatTensor(line_map).unsqueeze(0).to(device).unsqueeze(0)
    train_input = torch.cat((train_image, train_label, train_map), dim=1)
    pred_labels = dose_cal_model(train_input)   
    return pred_labels.squeeze(0).squeeze(0).detach().cpu().numpy()


def single_seed_dose_calculation_dl(pos, direc, dose_image, dose_cal_model, infer_image_size, seed_info, image_normalize_min, image_normalize_max, image_normalize_scale):
    """
    Calculate the radiation dose distribution for a single seed using a deep learning model.

    This function predicts the radiation dose distribution based on the seed's spatial position, orientation,
    and physical properties. It prepares the input tensors, processes them using a deep learning model,
    and returns the resulting dose map.

    Parameters:
        pos (tuple):
            The (z, y, x) coordinates of the seed position in voxel space.
        direc (tuple):
            A direction vector (dx, dy, dz) indicating the orientation of the radiation seed.
        dose_image (SimpleITK.Image):
            A medical image representing the dose grid, containing spatial metadata (size, spacing, origin).
        dose_cal_model (torch.nn.Module):
            A pre-trained deep learning model for predicting radiation dose distributions.
        infer_image_size (tuple):
            The size of the cropped region used for dose inference.
        seed_info (dict):
            Dictionary containing seed-specific parameters:
                - 'length' (float): Effective length of the radiation seed.
        image_normalize_min (float):
            Minimum value for image normalization.
        image_normalize_max (float):
            Maximum value for image normalization.
        image_normalize_scale (float):
            Scaling factor applied during image normalization.

    Returns:
        numpy.ndarray:
            A 3D NumPy array representing the predicted radiation dose distribution.

    Workflow:
        1. Normalize and crop the dose image around the seed position.
        2. Transform the seed position into physical coordinates.
        3. Generate a spatial dose map (soft_treatment_plan) based on seed position.
        4. Create a line source map using seed position, orientation, and length.
        5. Prepare input tensors for the deep learning model.
        6. Pass tensors through the model to predict the dose distribution.
        7. Convert the model output into a SimpleITK image and restore metadata.
        8. Return the predicted dose map as a NumPy array.
    """
    # Disable gradient computations during inference
    with torch.no_grad():
        # Extract image metadata
        _, image_spacing, image_origin = dose_image.GetSize(), dose_image.GetSpacing(), dose_image.GetOrigin()
        
        # Crop and normalize the dose image
        crop_img = crop_from_pos(pos[::-1], normalize_dose_image(dose_image, image_normalize_min, image_normalize_max, image_normalize_min, image_normalize_max), infer_image_size)
        
        # Transform seed position to physical space
        physical_pos = position_transform(dose_image, pos)[0]
        
        # Generate spatial dose map
        soft_treatment_plan = position_soft_method(physical_pos, image_origin, infer_image_size, image_spacing)
        
        # Create line source map based on seed orientation and length
        line_map = line_source_map(
            physical_pos,
            direction_transform(dose_image, direc)[0],
            image_origin,
            infer_image_size,
            image_spacing,
            seed_info['length']
        )
        
        # Prepare model input tensors
        device = next(dose_cal_model.parameters()).device
        train_image = torch.FloatTensor(normalize_dose_array(sitk.GetArrayFromImage(crop_img), image_normalize_min, image_normalize_max, image_normalize_scale)).unsqueeze(0).to(device).unsqueeze(0)
        train_label = torch.FloatTensor(soft_treatment_plan).unsqueeze(0).to(device).unsqueeze(0)
        train_map = torch.FloatTensor(line_map).unsqueeze(0).to(device).unsqueeze(0)
        
        # Concatenate tensors into model input
        train_input = torch.cat((train_image, train_label, train_map), dim=1).to(device)
        
        # Predict dose distribution
        pred_label = dose_cal_model(train_input)
        output = pred_label.squeeze(0).squeeze(0).detach().cpu().numpy()
    
    # Convert output to SimpleITK image and restore metadata
    pred_label_image = sitk.GetImageFromArray(output)
    pred_label_image.CopyInformation(crop_img)
    pred_label_image = pad_to_original_size(pred_label_image, dose_image)
    
    # Return dose map as NumPy array
    return sitk.GetArrayFromImage(pred_label_image) #.transpose(2, 1, 0)


def get_lowest_pos_index(planned_seeds, radiation, index_num=1):
    """
    Find the indices of the seeds in `planned_seeds` that have the lowest radiation doses.

    Parameters:
        planned_seeds (list of tuples): A list where each element is a tuple representing a position in the 3D space (x, y, z).
        radiation (numpy.ndarray): A 3D numpy array representing radiation doses at each spatial position.
        index_num (int, optional): Number of seeds with the lowest doses to find. Default is 1 (the single lowest dose).

    Returns:
        list of int: A list of indices of the seeds in `planned_seeds` that correspond to the lowest radiation doses.
    """
    # Initialize lists to store the lowest radiation doses and their corresponding indices
    lowest_doses = [float('inf')] * index_num  # Start with positive infinity
    lowest_indices = [-1] * index_num  # Placeholder for indices

    # Iterate through each seed in `planned_seeds`
    for i, (pos, _) in enumerate(planned_seeds):
        # Map the seed's normalized position to the radiation grid indices
        pos_indices = np.array([
            min(int(pos[0] * radiation.shape[0]), radiation.shape[0]-1),
            min(int(pos[1] * radiation.shape[1]), radiation.shape[1]-1),
            min(int(pos[2] * radiation.shape[2]), radiation.shape[2]-1)
        ])
        # Retrieve the radiation dose at the mapped position
        cur_dose = radiation[tuple(pos_indices)]

        # Insert the current dose into the list of lowest doses if applicable
        for j in range(index_num):
            if cur_dose < lowest_doses[j]:
                # Shift higher-ranked doses and indices up the list
                lowest_doses[j+1:] = lowest_doses[j:index_num-1]
                lowest_indices[j+1:] = lowest_indices[j:index_num-1]
                
                # Update the current rank with the new dose and index
                lowest_doses[j] = cur_dose
                lowest_indices[j] = i
                break  # Exit the loop once the dose is placed

    return lowest_indices  # Return the indices of the seeds with the lowest radiation doses


def get_highest_pos_index(planned_seeds, radiation, index_num=1):
    """
    Find the indices of the seeds in `planned_seeds` that have the highest radiation doses.

    Parameters:
        planned_seeds (list of tuples): A list where each element is a tuple representing a position in the 3D space (x, y, z).
        radiation (numpy.ndarray): A 3D numpy array representing radiation doses at each spatial position.
        index_num (int, optional): Number of seeds with the highest doses to find. Default is 1 (the single highest dose).

    Returns:
        list of int: A list of indices of the seeds in `planned_seeds` that correspond to the highest radiation doses.
    """
    # Initialize lists to store the highest radiation doses and their corresponding indices
    highest_doses = [-float('inf')] * index_num  # Start with negative infinity
    highest_indices = [-1] * index_num  # Placeholder for indices

    # Iterate through each seed in `planned_seeds`
    for i, (pos, _) in enumerate(planned_seeds):
        # Map the seed's normalized position to the radiation grid indices
        pos_indices = np.array([
            min(int(pos[0] * radiation.shape[0]), radiation.shape[0] - 1),
            min(int(pos[1] * radiation.shape[1]), radiation.shape[1] - 1),
            min(int(pos[2] * radiation.shape[2]), radiation.shape[2] - 1)
        ])
        # Retrieve the radiation dose at the mapped position
        cur_dose = radiation[tuple(pos_indices)]

        # Insert the current dose into the list of highest doses if applicable
        for j in range(index_num):
            if cur_dose > highest_doses[j]:
                # Shift lower-ranked doses and indices down the list
                highest_doses[j+1:] = highest_doses[j:index_num-1]
                highest_indices[j+1:] = highest_indices[j:index_num-1]
                
                # Update the current rank with the new dose and index
                highest_doses[j] = cur_dose
                highest_indices[j] = i
                break  # Exit the loop once the dose is placed

    return highest_indices


def remove_elements_by_indices(my_list, indices_to_remove):
    """
    Remove elements from the list at the specified indices.
    
    Parameters:
        my_list (list): The original list from which elements will be removed.
        indices_to_remove (list): The list of indices of the elements to be removed.
    
    Returns:
        list: The updated list with the specified elements removed.
    """
    # Convert indices_to_remove to a set for O(1) lookups
    indices_to_remove = set(indices_to_remove)

    # Use list comprehension to keep elements that are not in the indices_to_remove
    return [item for idx, item in enumerate(my_list) if idx not in indices_to_remove]


def shrink_mask(mask, shrink_factor):
    """
    Shrink a mask by a specified factor.

    Parameters:
        mask (numpy.ndarray): The mask to be shrunk.
        shrink_factor (float): The factor by which to shrink the mask.

    Returns:
        numpy.ndarray: The shrunk mask.
    """
    return geometry.shrink_island_by_distance(mask, shrink_factor)


def objective_function(x, dose_volume, radiation_volume, seed_sigma, lowest_dose, DVH_rate, seed_avr_dose):
    radiation = np.zeros(radiation_volume.shape)
    for i in range(len(x)//6):  # Parallelize this loop
        cur_effective_radiation = simple_single_dose_calculation(radiation_volume.shape, 
                                                                 [x[6*i], x[6*i+1], x[6*i+2]], 
                                                                 [x[6*i+3], x[6*i+4], x[6*i+5]], 
                                                                 seed_sigma, 
                                                                 seed_avr_dose)
        radiation += cur_effective_radiation
    effective_radiation = radiation * radiation_volume
    cur_DVH_rate = np.sum(effective_radiation > lowest_dose) / np.sum(radiation_volume==1)
    if cur_DVH_rate <= DVH_rate:
        return - cur_DVH_rate
    else:
        return - DVH_rate - np.sum(effective_radiation) / np.sum(radiation)
    
    
def constraint_direc(x):
    """
    Calculate the sum of squared differences from 1 for each 3D direction vector in seeds, representing
    a constraint that each direction vector should ideally be a unit vector.

    Parameters:
        x (list or ndarray): Flattened array of seed data, where each seed has a position (x, y, z) and 
                              a direction (sigma_x, sigma_y, sigma_z) in the array.
                              The direction vector components are at indices (6*i+3, 6*i+4, 6*i+5).

    Returns:
        float: The cumulative "score" which sums the deviations of each direction vector's squared magnitude from 1.
               A score close to 0 indicates that all direction vectors are close to being unit vectors.
    """
    score = 0
    # Loop through each seed in the flattened array, each seed having 6 parameters
    for i in range(len(x)//6):
        # Calculate the squared sum of the components of the direction vector
        deviation = x[6*i+3]**2 + x[6*i+4]**2 + x[6*i+5]**2 - 1
        # Accumulate the absolute deviation from 1
        score += np.abs(deviation)
        
    return score


def constraint_bounds(x):
    """
    Constraint function to ensure that each coordinate of all seeds lies within the range [0, 1].

    This function calculates a cumulative "violation score" for coordinates that fall outside of the [0, 1] range.
    A penalty is applied to each coordinate that is either less than 0 or greater than 1.

    Parameters:
        x (list or ndarray): Flattened array of seed data where positions are interleaved.
                             Coordinates are located at indices i % 6 < 3, representing (x, y, z).

    Returns:
        float: A cumulative "violation score" indicating the total deviation of coordinates outside [0, 1].
               A score of 0 means all coordinates are within bounds.
    """
    x_arr = np.array(x)  # Ensure x is a numpy array (if it's a list or other type)
    x_pos = x_arr.reshape(-1, 3)[::2]  # Select position coordinates (x, y, z) for each seed
    x_pos = x_pos.reshape(-1, 1)  # Reshape the coordinates to a column vector (though this is unnecessary)

    # Return the sum of violations, counting coordinates that are out of bounds
    return np.sum((x_pos < 0) & (x_pos > 1))  # Return the sum of the violation condition

    
def from_seeds_to_x(seeds):
    """
    Converts a list of seeds, each defined by a position and direction, into a flattened list
    for use in optimization or other computations.

    Args:
        seeds (list): A list of seeds where each seed is represented as a tuple:
                      seed[0]: A numpy array or list of length 3, representing the position [x, y, z].
                      seed[1]: A numpy array or list of length 3, representing the direction [dx, dy, dz].

    Returns:
        array: A flattened array of seed positions and directions in the following order:
              [x1, y1, z1, dx1, dy1, dz1, x2, y2, z2, dx2, dy2, dz2, ...].
    """
    x = []  # Initialize an empty list to store the flattened seed attributes.
    for seed in seeds:
        # Append the 3D position coordinates [x, y, z] to the list.
        x.append(seed[0][0])  # x-coordinate of the position
        x.append(seed[0][1])  # y-coordinate of the position
        x.append(seed[0][2])  # z-coordinate of the position
        
        # Append the 3D direction components [dx, dy, dz] to the list.
        x.append(seed[1][0])  # x-component of the direction
        x.append(seed[1][1])  # y-component of the direction
        x.append(seed[1][2])  # z-component of the direction
    return np.array(x)  # Return the flattened list containing all seed positions and directions.


def update_seeds(single_seed_radiations, planned_seeds):
    """
    Updates the seed configurations and calculates the total radiation field.

    Args:
        single_seed_radiations (list of ndarray): A list of individual radiation fields, 
            where each element is a 3D array representing the radiation distribution 
            contributed by a single seed.
        planned_seeds (list of tuples): A list of planned seed configurations, where 
            each seed is represented as a tuple containing its position and direction.

    Returns:
        tuple:
            - new_single_seed_radiations (list of ndarray): The updated list of individual seed radiation fields.
            - new_planned_seeds (list of tuples): The updated list of planned seed configurations.
            - new_radiation (ndarray): The cumulative radiation field, obtained by summing all single-seed radiation fields.

    Steps:
        1. Assign the provided single-seed radiation fields to `new_single_seed_radiations`.
        2. Assign the provided planned seeds to `new_planned_seeds`.
        3. Calculate the cumulative radiation field by summing all single-seed radiation fields.
        4. Return the updated single-seed radiation fields, planned seeds, and cumulative radiation field.
    """
    # Step 1: Use the provided single-seed radiation fields
    new_single_seed_radiations = single_seed_radiations

    # Step 2: Use the provided planned seeds
    new_planned_seeds = planned_seeds

    # Step 3: Compute the cumulative radiation field
    new_radiation = np.sum(np.asarray(new_single_seed_radiations), axis=0)

    # Step 4: Return the updated configurations and the cumulative radiation field
    return new_single_seed_radiations, new_planned_seeds, new_radiation


def calculate_tmp_DVH_rate(pos, direc, cur_radiation, mask_volume, lowest_dose, single_seed_radiations, seed_sigma, seed_avr_dose):
    """
    Calculate the Dose-Volume Histogram (DVH) rate after placing a new seed, considering its radiation contribution.

    Args:
        pos (tuple or array-like): The position of the seed in the 3D space (x, y, z).
        direc (tuple or array-like): The direction of the seed's radiation.
        cur_radiation (ndarray): The current radiation distribution in the 3D grid.
        mask_volume (ndarray): A binary mask representing the target volume, such as the tumor region (1 for target, 0 for non-target).
        lowest_dose (float): The minimum dose threshold to consider in the DVH calculation.
        single_seed_radiations (list of ndarray): A list of radiation distributions from previously placed seeds.
        seed_sigma (float): The standard deviation for the spread of radiation from a seed (Gaussian model parameter).
        seed_avr_dose (float): The average dose delivered by a single seed (Gaussian model parameter).

    Returns:
        tuple:
            - tmp_DVH_rate (float): The DVH rate after placing the current seed, indicating the fraction of target volume receiving a dose above the threshold.
            - tmp_seed_radiation (ndarray): The radiation contribution from the current seed.
            - tmp_radiation (ndarray): The updated cumulative radiation field after adding the current seed's contribution.
            - direc (tuple or array-like): The direction vector (unchanged).

    Steps:
        1. Calculate the radiation distribution for the current seed based on its position and direction.
        2. Add the new seed's radiation contribution to the list of individual seed contributions.
        3. Compute the updated total radiation field by summing all individual contributions.
        4. Apply the target volume mask to isolate the target regions in the radiation field.
        5. Calculate the DVH rate as the fraction of the target volume receiving a dose greater than the specified threshold.
        6. Return the DVH rate, the new seed's radiation contribution, the updated total radiation field, and the unchanged direction.
    """
    
    # Step 1: Calculate the radiation distribution from the current seed using a Gaussian model
    tmp_seed_radiation = simple_single_dose_calculation(
        cur_radiation.shape, pos, direc, seed_sigma, seed_avr_dose
    )

    # Step 2: Append the current seed's radiation contribution to the list of previous seed radiations
    tmp_single_seed_radiations = single_seed_radiations.copy()
    tmp_single_seed_radiations.append(tmp_seed_radiation)

    # Step 3: Compute the updated total radiation field by summing the contributions from all seeds
    tmp_radiation = np.sum(np.asarray(tmp_single_seed_radiations), axis=0) * mask_volume

    # Step 4: Compute the DVH rate for the target region
    tmp_DVH_rate = np.sum(tmp_radiation > lowest_dose) / np.sum(mask_volume)

    # Step 5: Return the computed values
    return tmp_DVH_rate, tmp_seed_radiation, tmp_radiation, direc


def calculate_tmp_DVH_rate_v2(pos, direc, dose_image, dose_cal_model, mask_volume, lowest_dose, single_seed_radiations):
    """
    Calculate the Dose-Volume Histogram (DVH) rate after placing a new seed, considering its radiation contribution.

    Args:
        pos (tuple or array-like): The 3D position of the seed (x, y, z).
        direc (tuple or array-like): The direction vector of the seed's radiation.
        dose_image (ndarray): The dose distribution image for the radiation grid.
        dose_cal_model (model): Model for calculating dose distribution from a seed.
        mask_volume (ndarray): A binary mask of the target region (1 for target, 0 for non-target).
        lowest_dose (float): The minimum dose threshold for DVH calculation.
        single_seed_radiations (list of ndarray): List of radiation fields from previously placed seeds.

    Returns:
        tuple:
            - tmp_DVH_rate (float): The DVH rate, the fraction of the target volume receiving a dose above `lowest_dose`.
            - tmp_seed_radiation (ndarray): Radiation distribution from the current seed.
            - tmp_radiation (ndarray): The updated cumulative radiation field after adding the current seed's contribution.
            - direc (tuple or array-like): The unchanged direction vector.
    
    Steps:
        1. Calculate the radiation distribution from the current seed using a Gaussian dose model.
        2. Add the new seed's radiation to the list of previous seed radiation fields.
        3. Compute the cumulative radiation field by summing all individual seed contributions.
        4. Apply the mask to isolate the target region in the radiation field.
        5. Calculate the DVH rate as the fraction of the target region receiving a dose greater than `lowest_dose`.
        6. Return the DVH rate, the current seed's radiation, the cumulative radiation field, and the unchanged direction vector.
    """
    
    # Step 1: Calculate the radiation distribution from the current seed using a dose calculation model
    tmp_seed_radiation = single_dose_calculation_v2(pos, direc, dose_image, dose_cal_model)
    # This computes the radiation distribution from the current seed based on its position (pos) and direction (direc)

    # Step 2: Add the current seed's radiation contribution to the list of previous seed radiations
    tmp_single_seed_radiations = single_seed_radiations.copy()  # Make a copy of the previously placed seeds' radiation fields
    tmp_single_seed_radiations.append(tmp_seed_radiation)  # Append the current seed's radiation to the list

    # Step 3: Calculate the cumulative radiation field from all seeds
    tmp_radiation = np.sum(np.asarray(tmp_single_seed_radiations), axis=0) * mask_volume
    # The cumulative radiation field is the sum of all seed contributions, scaled by the mask_volume
    # This ensures that the radiation is only calculated in the target regions defined by mask_volume

    # Step 4: Compute the DVH rate (fraction of the target volume receiving dose > lowest_dose)
    tmp_DVH_rate = np.sum(tmp_radiation > lowest_dose) / np.sum(mask_volume)
    # This calculates the fraction of the target region that has received a dose greater than the specified threshold (`lowest_dose`)
    # The result is the Dose-Volume Histogram (DVH) rate, representing how much of the target region has been adequately irradiated

    # Step 5: Return the DVH rate, current seed's radiation, updated radiation field, and direction
    return tmp_DVH_rate, tmp_seed_radiation, tmp_radiation, direc


def process_best_x(best_x, cur_radiation, mask_volume, in_lowest_dose, volume, seed_sigma, seed_avr_dose):
    """
    Processes the deep learning model's output (`best_x`) to generate a list of optimized seed placements 
    and their corresponding radiation distributions. Additionally, it computes the Dose-Volume Histogram (DVH) rate.

    Parameters:
        best_x (torch.Tensor): Tensor containing the positions and directions of the seeds. The tensor has a shape 
                               of (N, 6), where each seed is represented by a 6-dimensional vector:
                               - First 3 elements: Seed position (x, y, z).
                               - Last 3 elements: Seed direction (dx, dy, dz).
        cur_radiation (numpy.ndarray): Current 3D radiation dose distribution map.
        mask_volume (numpy.ndarray): 3D binary mask representing the target regions (1 = target, 0 = non-target).
        in_lowest_dose (float): Minimum acceptable dose for a region to be considered adequately treated.
        volume (float): Total volume of the target region to be irradiated.
        seed_sigma (tuple): Tuple representing the radiation spread (sigma) for the seed:
                            - (length, radial_x, radial_y).
        seed_avr_dose (float): Average dose delivered by a single seed.

    Returns:
        tuple:
            - best_planned_seeds (list): List of optimized seed placements, where each seed is represented as 
                                         [position, direction]. Position is a 3D array (x, y, z), and direction 
                                         is a normalized 3D array (dx, dy, dz).
            - best_single_seed_radiations (list): List of radiation distributions (3D arrays) for each seed.
            - best_DVH_rate (float): The DVH rate, calculated as the percentage of the target volume that meets or 
                                     exceeds the minimum dose requirement (`in_lowest_dose`).
    """
    best_planned_seeds = []  # List to store optimized seed placements
    best_single_seed_radiations = []  # List to store radiation distributions of individual seeds
    
    # Disable gradient tracking for efficiency
    with torch.no_grad():
        # Convert `best_x` tensor from GPU (if applicable) to a NumPy array
        best_x = best_x.detach().cpu().numpy()
        
        # Loop through each seed in `best_x` (6 values per seed: 3 for position, 3 for direction)
        for i in range(best_x.shape[0] // 6):
            pos = best_x[6 * i:6 * i + 3]  # Extract position (x, y, z)
            direc = best_x[6 * i + 3:6 * i + 6]  # Extract direction (dx, dy, dz)
            
            # Normalize the direction vector
            norm = np.linalg.norm(direc)
            if norm != 0:
                direc = direc / norm
            else:
                # Handle zero-norm direction vectors by skipping normalization
                print("Warning: Direction vector has zero norm, skipping normalization.")
            
            # Store the seed's position and direction
            seed = [pos.reshape(-1), direc.reshape(-1)]
            best_planned_seeds.append(seed)
            
            # Calculate the radiation distribution for the current seed
            best_single_seed_radiations.append(
                simple_single_dose_calculation(cur_radiation.shape, seed[0], seed[1], seed_sigma, seed_avr_dose)
            )
        
        # Combine the radiation distributions of all seeds to compute the overall radiation map
        best_radiation = np.sum(np.asarray(best_single_seed_radiations), axis=0)
        
        # Compute the DVH rate as the percentage of the target volume receiving sufficient dose
        best_DVH_rate = np.sum(best_radiation * mask_volume > in_lowest_dose) / volume
    
    # Return the optimized seed placements, their radiation distributions, and the DVH rate
    return best_planned_seeds, best_single_seed_radiations, best_DVH_rate


def from_x_to_seeds(x):
    """
    Convert a flattened tensor of seed parameters into a structured list of seeds.

    Args:
        x (torch.Tensor): A 1D tensor with seed data. Each seed is represented by
                          6 consecutive values: 
                          - First 3 values: Position (x, y, z)
                          - Next 3 values: Direction (dx, dy, dz)

    Returns:
        list: A list of seeds, where each seed is a list containing:
              - Position: [x, y, z]
              - Normalized Direction: [dx, dy, dz]
    """
    # Detach the tensor from computation graph, move to CPU, and convert to numpy array
    x = x.detach().cpu().numpy()
    
    seeds = []  # Initialize a list to store seed data
    
    # Loop through the tensor data, 6 values at a time (1 seed = 6 values)
    for i in range(x.shape[0] // 6):
        # Extract position (first 3 values)
        pos = x[6 * i:6 * i + 3]
        
        # Extract direction (next 3 values)
        direc = x[6 * i + 3:6 * i + 6]
        
        # Normalize the direction vector to unit length
        norm = np.linalg.norm(direc)
        if norm != 0:
            direc = direc / norm  # Normalize if norm is non-zero
        else:
            # Handle zero-norm direction vectors (log a warning)
            print("Warning: Direction vector has zero norm, skipping normalization.")
        
        # Store the seed as a list: [position, normalized direction]
        seed = [pos.tolist(), direc.tolist()]
        seeds.append(seed)
    
    return seeds


# def position_soft_method(seed, image_origin, image_size, image_spacing):
#     """
#     Generate a spatial influence map based on a seed point's position.
    
#     This function creates a soft treatment plan by modeling a spherical region of influence 
#     around the seed point. The influence diminishes as the distance from the seed increases.

#     Parameters:
#     ----------
#     seed (tuple or list): 
#         Coordinates of the seed point in physical space.
#     image_origin (tuple or list): 
#         Physical coordinates of the image origin.
#     image_size (tuple or list): 
#         Size of the 3D image grid (number of voxels in each dimension).
#     image_spacing (tuple or list): 
#         Spacing between voxels in each dimension (physical distance per voxel).

#     Returns:
#     -------
#     soft_treatment_plan (numpy array): 
#         A 3D array representing the normalized spatial influence map of the seed point.
#     """
#     sphere_radius = 4  # Radius of the sphere of influence
#     sphere_volume = (4/3) * np.pi * sphere_radius**3  # Volume of the sphere

#     # Initialize a 3D grid for the treatment plan
#     grid_shape = tuple(image_size)
#     soft_treatment_plan = np.zeros(grid_shape)
#     grid_spacing = np.array(image_spacing)

#     # Iterate through all grid points
#     for x in range(grid_shape[0]):
#         for y in range(grid_shape[1]):
#             for z in range(grid_shape[2]):
#                 # Calculate the voxel's physical position
#                 voxel_center = np.array([x, y, z]) * grid_spacing + np.array(image_origin)
#                 distance = np.linalg.norm(voxel_center - np.array(seed)) 
                
#                 # voxel_center = np.array([x, y, z]) * grid_spacing + np.array(image_origin)
#                 # distance = np.linalg.norm(voxel_center - np.array(seed))  # Distance from the seed

#                 # Check if the voxel is within the sphere of influence
#                 if distance <= sphere_radius:
#                     # Calculate overlapping volume for partial coverage
#                     overlapping_volume = (4/3) * np.pi * ((sphere_radius - distance) ** 3)
#                     normalized_volume = overlapping_volume / sphere_volume
#                     soft_treatment_plan[x, y, z] += normalized_volume

#     # Normalize the influence map so that it sums up to 1
#     soft_treatment_plan[soft_treatment_plan > 0] /= np.sum(soft_treatment_plan[soft_treatment_plan > 0])
    
#     return soft_treatment_plan


def position_soft_method(seed, image_origin, image_size, image_spacing):
    """
    Generate a soft treatment plan based on a spherical distribution around a given seed point.

    Parameters:
    seed (np.ndarray): The seed point in 3D space, represented as [x, y, z].
    image_origin (np.ndarray): The physical origin of the image grid in 3D space, represented as [x, y, z].
    image_size (tuple): The size of the 3D image grid, represented as (x_dim, y_dim, z_dim).
    image_spacing (np.ndarray): The spacing between voxels in each dimension, represented as [x_spacing, y_spacing, z_spacing].

    Returns:
    np.ndarray: A 3D array representing the soft treatment plan, where each voxel's value corresponds 
                to the normalized overlap volume of a sphere centered at the seed point.

    Steps:
    1. Define a sphere with a fixed radius and compute its theoretical volume.
    2. Create a 3D grid of voxel centers based on the image size and spacing.
    3. Calculate the Euclidean distance from each voxel center to the seed point.
    4. Determine overlapping voxels within the sphere radius and compute their contribution to the sphere volume.
    5. Normalize the contribution of overlapping voxels to form the soft treatment plan.
    6. Transpose the resulting array to match the expected output orientation.

    Notes:
    - This function assumes the sphere radius is 4 units and calculates volumes in physical space based on 
      the provided grid spacing and origin.
    - Ensure that the input parameters are consistent in their coordinate systems and units.

    Example Usage:
    seed = np.array([10, 10, 10])
    image_origin = np.array([0, 0, 0])
    image_size = (50, 50, 50)
    image_spacing = np.array([1, 1, 1])
    result = position_soft_method(seed, image_origin, image_size, image_spacing)
    """
    sphere_radius = 4  # Radius of the sphere in physical units
    sphere_volume = (4 / 3) * np.pi * sphere_radius**3  # Volume of the sphere

    # Create a 3D grid for the image based on its size
    grid_shape = tuple(image_size)
    soft_treatment_plan = np.zeros(grid_shape)  # Initialize the treatment plan array
    grid_spacing = np.array(image_spacing)  # Spacing between grid points

    # Generate voxel center coordinates
    x_grid, y_grid, z_grid = np.meshgrid(
        np.arange(grid_shape[0]),
        np.arange(grid_shape[1]),
        np.arange(grid_shape[2]),
        indexing='ij'
    )
    voxel_centers = np.stack([x_grid, y_grid, z_grid], axis=-1) * grid_spacing + image_origin

    # Calculate the Euclidean distance from each voxel center to the seed point
    distances = np.linalg.norm(voxel_centers - seed[0:3], axis=-1)

    # Compute overlapping volumes for voxels within the sphere radius
    overlap_mask = distances <= sphere_radius
    overlapping_volume = overlap_mask * (4 / 3) * np.pi * ((sphere_radius - distances) ** 3)

    # Normalize the overlapping volume by the sphere volume
    normalized_volume = overlapping_volume / sphere_volume
    soft_treatment_plan += normalized_volume

    # Normalize the treatment plan to ensure sum of non-zero values equals 1
    non_zero_values = soft_treatment_plan[soft_treatment_plan > 0]
    if np.sum(non_zero_values) > 0:
        soft_treatment_plan[soft_treatment_plan > 0] /= np.sum(non_zero_values)

    # Transpose to match the desired output orientation
    soft_treatment_plan = np.transpose(soft_treatment_plan, (2, 1, 0))
    return soft_treatment_plan


def crop_from_pos(center_point_index, image, patch_size=(32, 32, 32)):
    """
    Extract a sub-region (patch) from a 3D image centered at a specified point.
    
    Args:
        center_point_index (tuple): Coordinates of the center point (x, y, z).
        image (SimpleITK.Image): Input 3D image in SimpleITK format.
        patch_size (tuple): Size of the patch to extract, default is (32, 32, 32).
    
    Returns:
        SimpleITK.Image: Extracted sub-region (patch) of the image.
    """
    # Convert center point coordinates to integers
    center_x, center_y, center_z = map(int, center_point_index)
    
    # Calculate half the patch size for each dimension
    half_size = [size // 2 for size in patch_size]
    
    # Calculate the starting coordinates of the patch
    start_x = center_x - half_size[0]
    start_y = center_y - half_size[1]
    start_z = center_z - half_size[2]
    
    # Get the original image dimensions
    image_size = image.GetSize()
    
    # Ensure the starting coordinates are within image boundaries
    start_x = max(0, min(start_x, image_size[0] - patch_size[0]))
    start_y = max(0, min(start_y, image_size[1] - patch_size[1]))
    start_z = max(0, min(start_z, image_size[2] - patch_size[2]))
    
    # Extract the patch using SimpleITK's Extract function
    cropped_image = sitk.Extract(
        image,
        size=patch_size,
        index=[start_x, start_y, start_z]
    )
    
    return cropped_image


import SimpleITK as sitk

def pad_to_original_size(cropped_image: sitk.Image, 
                         original_image: sitk.Image) -> sitk.Image:
    """
    Pad a cropped image back to the size of the original image while preserving spatial information.
    
    Args:
        cropped_image (SimpleITK.Image): The cropped sub-region image.
        original_image (SimpleITK.Image): The original reference image.
    
    Returns:
        SimpleITK.Image: A padded image matching the original image's size, spacing, and origin.
    """
    # Step 1: Create a blank image with the same size as the original image
    padded_image = sitk.Image(
        original_image.GetSize(), 
        cropped_image.GetPixelID(),
        cropped_image.GetNumberOfComponentsPerPixel()
    )
    padded_image.CopyInformation(original_image)
    
    # Step 2: Get index and size information
    crop_origin = cropped_image.GetOrigin()
    start_index = original_image.TransformPhysicalPointToIndex(crop_origin)
    cropped_size = cropped_image.GetSize()
    
    # Step 3: Ensure the index is within valid range
    image_size = original_image.GetSize()
    
    start_index = [max(0, min(start_index[d], image_size[d] - 1)) for d in range(3)]
    end_index = [start_index[d] + cropped_size[d] for d in range(3)]
    end_index = [min(end_index[d], image_size[d]) for d in range(3)]
    
    # Step 4: Convert images to NumPy arrays for padding
    cropped_array = sitk.GetArrayFromImage(cropped_image)
    padded_array = sitk.GetArrayFromImage(padded_image)
    
    # Step 5: Compute NumPy indices (note that SimpleITK and NumPy have reversed coordinate order)
    z_start, y_start, x_start = start_index[::-1]
    z_end, y_end, x_end = end_index[::-1]
    
    # Ensure the cropped image size matches the valid region
    cropped_z, cropped_y, cropped_x = cropped_array.shape
    valid_z = min(cropped_z, z_end - z_start)
    valid_y = min(cropped_y, y_end - y_start)
    valid_x = min(cropped_x, x_end - x_start)
    
    # Step 6: Insert the cropped image into the padded image
    padded_array[z_start:z_start+valid_z, y_start:y_start+valid_y, x_start:x_start+valid_x] = \
        cropped_array[:valid_z, :valid_y, :valid_x]
    
    # Step 7: Convert back to SimpleITK image and set metadata
    padded_image = sitk.GetImageFromArray(padded_array)
    padded_image.CopyInformation(original_image)
    
    return padded_image


# def pad_to_original_size(cropped_image: sitk.Image, 
#                          original_image: sitk.Image) -> sitk.Image:
#     """
#     Pad a cropped image back to the size of the original image while preserving spatial information.
    
#     Args:
#         cropped_image (SimpleITK.Image): The cropped sub-region image.
#         original_image (SimpleITK.Image): The original reference image.
    
#     Returns:
#         SimpleITK.Image: A padded image matching the original image's size, spacing, and origin.
#     """
#     # Create an empty image with the same size and metadata as the original image
#     padded_image = sitk.Image(
#         original_image.GetSize(), 
#         cropped_image.GetPixelID(),
#         cropped_image.GetNumberOfComponentsPerPixel()
#     )
#     padded_image.SetOrigin(original_image.GetOrigin())
#     padded_image.SetSpacing(original_image.GetSpacing())
#     padded_image.SetDirection(original_image.GetDirection())
    
#     # Initialize the padded image with zeros
#     padded_array = sitk.GetArrayFromImage(padded_image)
#     padded_array.fill(0)
#     padded_image = sitk.GetImageFromArray(padded_array)
#     padded_image.CopyInformation(original_image)
    
#     # Calculate the starting index of the cropped image in the original image space
#     crop_origin = cropped_image.GetOrigin()
#     start_index = original_image.TransformPhysicalPointToIndex(crop_origin)
    
#     # Get the size of the cropped image
#     patch_size = cropped_image.GetSize()
    
#     # Copy pixel values from the cropped image to the corresponding position in the padded image
#     for z in range(patch_size[2]):
#         for y in range(patch_size[1]):
#             for x in range(patch_size[0]):
#                 orig_x = start_index[0] + x
#                 orig_y = start_index[1] + y
#                 orig_z = start_index[2] + z
                
#                 # Ensure the indices are within bounds of the original image
#                 if (0 <= orig_x < original_image.GetSize()[0] and 
#                     0 <= orig_y < original_image.GetSize()[1] and 
#                     0 <= orig_z < original_image.GetSize()[2]):
#                     padded_image[orig_x, orig_y, orig_z] = cropped_image[x, y, z]
    
#     return padded_image


def line_source_map(seed, direction, image_origin, image_size, image_spacing, seed_length):
    """
    Generate a radiation dose distribution map from a line source defined by a seed and direction.

    This function models the radiation dose from a line source by calculating the influence 
    of the source along a specified direction in a 3D grid.

    Parameters:
    ----------
    seed (tuple or list): 
        Coordinates of the seed point in physical space.
    direction (tuple or list): 
        Direction vector indicating the orientation of the line source.
    image_origin (tuple or list): 
        Physical coordinates of the image origin.
    image_size (tuple or list): 
        Size of the 3D image grid (number of voxels in each dimension).
    image_spacing (tuple or list): 
        Spacing between voxels in each dimension (physical distance per voxel).

    Returns:
    -------
    line_map (SimpleITK Image): 
        A 3D image representing the radiation distribution map from the line source.
    """
    # Create physical coordinate grids
    x = image_origin[0] + np.arange(image_size[0]) * image_spacing[0]
    y = image_origin[1] + np.arange(image_size[1]) * image_spacing[1]
    z = image_origin[2] + np.arange(image_size[2]) * image_spacing[2]

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    Vx = X - seed[0]
    Vy = Y - seed[1]
    Vz = Z - seed[2]

    # Compute squared distances and avoid division by zero
    distance_squared = Vx**2 + Vy**2 + Vz**2
    distance_squared[np.where(distance_squared < 0.01)] = 0.01

    # Define line source length and direction normalization
    norm_direction_vector = direction / np.linalg.norm(direction[:3])
    norm_direction_vector = np.array([norm_direction_vector[0], norm_direction_vector[1], -norm_direction_vector[2]])

    # Calculate start and end points of the seed
    A_prime = seed[0:3] - seed_length/2 * norm_direction_vector
    B_prime = seed[0:3] + seed_length/2 * norm_direction_vector

    # Compute vectors to the endpoints of the line source
    V_PA = np.array([X - A_prime[0], Y - A_prime[1], Z - A_prime[2]])
    V_PB = np.array([X - B_prime[0], Y - B_prime[1], Z - B_prime[2]])

    # Calculate vector magnitudes
    V_PA_magnitude = np.sqrt(np.sum(V_PA**2, axis=0))
    V_PB_magnitude = np.sqrt(np.sum(V_PB**2, axis=0))

    # Calculate the angle between vectors
    dot_product = np.sum(V_PA * V_PB, axis=0)
    cos_beta = np.abs(dot_product / (V_PA_magnitude * V_PB_magnitude))
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))

    # Compute the angle to the direction vector
    vectors_to_mid_point = np.stack((X - seed[0], Y - seed[1], Z - seed[2]), axis=-1)
    norm_vectors_to_mid_point = vectors_to_mid_point / (np.linalg.norm(vectors_to_mid_point, axis=-1, keepdims=True) + 1e-5)
    cos_theta = np.dot(norm_vectors_to_mid_point, norm_direction_vector)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Generate the dose distribution map
    line_map = beta / (np.sin(theta) * (distance_squared)**(-1) + 1e-5)
    line_map = np.transpose(line_map, (2, 1, 0))

    # Create a SimpleITK image from the dose distribution map
    # line_map = sitk.GetImageFromArray(line_map)
    # line_map.SetOrigin(image_origin)
    # line_map.SetSpacing(image_spacing)
    
    return line_map


def train_model(epochs, x, BrachyPlanNet, optimizer, criterion, fitting_loss, early_stop, seeds_variation):
    """
    Train the BrachyPlanNet model for radiotherapy seed optimization over multiple epochs.

    This function performs the training loop where the model learns to predict optimal seed positions and 
    directions based on the provided input data. During each epoch, the model performs forward propagation, 
    calculates the loss, and applies backpropagation to update the model parameters using the specified optimizer. 
    The training loop will stop early if the early stopping criteria are met, based on the loss history.

    Parameters:
        epochs (int): 
            The number of epochs (iterations) to train the model.
            
        x (torch.Tensor): 
            A tensor containing the input data for seed positions and directions. The shape of the tensor should be 
            [N, 6], where N is the number of seeds. Each seed is represented by a 6-dimensional vector: 
            [x, y, z, dx, dy, dz] (position + direction).
            
        BrachyPlanNet (torch.nn.Module): 
            The neural network model that takes the seed data (position and direction) as input and predicts optimal 
            seed positions and directions. The model is trained to minimize the error between predicted and desired values.
            
        optimizer (torch.optim.Optimizer): 
            The optimization algorithm (e.g., Adam, AdamW) used to update the model's weights. 
            It adjusts the model parameters based on the gradients computed during backpropagation.
            
        criterion (torch.nn.Module): 
            The loss function used to compute the error between the predicted seed positions and directions 
            and the target ground truth. This is used to guide the model's learning process.
        
        fitting_loss (list): 
            A list to record the loss values during each epoch of training. This can be used to monitor the model's 
            convergence and performance over time.
        
        early_stop (function): 
            A function that checks whether early stopping criteria are met based on the loss history. It should return 
            a boolean indicating whether training should stop early to prevent overfitting or unnecessary computation.
            
        seeds_variation (list): 
            A list that stores the seed positions and directions after each epoch, in their transformed format. 
            This is useful for tracking the optimization process over time.

    Returns:
        tuple: 
            - best_x (torch.Tensor or None): 
                The best predicted seed positions and directions found during training. Returns `None` if training 
                stops early due to a critical error or early stopping criteria.
                
            - fitting_loss (list): 
                A list of loss values recorded at each epoch. This helps track the progress and performance of training.
                
            - seeds_variation (list): 
                A list containing seed variations (positions and directions) at each epoch, useful for understanding 
                how the model's predictions evolve over time.

    The function includes early stopping functionality, which halts the training if the model's performance stagnates, 
    preventing overfitting. It also handles exceptions that might occur during training and ensures that the best model 
    parameters are returned, even if training encounters an error.
    """
    best_x = x.clone()  # Initialize best_x to store the best prediction if early stopping occurs
    BrachyPlanNet.train()  # Set the model to training mode
    
    # Loop over the specified number of epochs
    for epoch in range(epochs):
        print("\nEpoch {}/{}".format(epoch + 1, epochs), end=", ")  # Print the current epoch number
        
        try:
            # Perform a forward pass to get the model's predictions (seed positions and directions)
            x_pre = BrachyPlanNet(x)
            
            # Zero out gradients before the backward pass to avoid accumulation
            optimizer.zero_grad()
            
            # Calculate the loss between predicted and actual values
            loss = criterion(x_pre)
            
            # Perform backpropagation to compute gradients
            loss.backward()
            
            # Apply the gradients to the model's parameters using the optimizer
            optimizer.step()
            
            # Store the loss for this epoch in the fitting_loss list
            fitting_loss.append(loss.item())
            
            # Check if early stopping conditions are met
            if not early_stop(fitting_loss[-1]):
                # If early stopping criteria are not met, update the best model prediction
                best_x = x_pre.clone()
                seeds_variation.append(from_x_to_seeds(best_x))  # Record seed variation for tracking
            
            # If early stopping criteria are triggered, print a message and stop training
            if early_stop.early_stop:
                print('Early stopping triggered')
                return best_x, fitting_loss, seeds_variation
        
        except Exception as e:
            # Catch and handle any exceptions (e.g., RuntimeError, ValueError)
            print("Critical error encountered during training. Stopping early.")
            return best_x, fitting_loss, seeds_variation  # Return the best prediction found so far

    # Return the best model's predictions (best_x) and training history (fitting_loss) after all epochs
    return best_x, fitting_loss, seeds_variation


def place_and_evaluate_seed(pos, direc, cur_radiation, mask_volume, in_lowest_dose, single_seed_radiations, seed_sigma, seed_avr_dose, direc_res):
    """
    Place a seed at a specific position in the 3D grid and evaluate its effect on radiation coverage.
    The function tests multiple candidate directions for seed placement and selects the one that maximizes
    the Dose-Volume Histogram (DVH) rate for the target region.

    Parameters:
        pos (tuple): Position of the seed in the 3D grid (x, y, z).
        direc (numpy.ndarray): The initial direction of the seed (represented as a vector).
        cur_radiation (numpy.ndarray): Current radiation distribution map (3D grid).
        mask_volume (numpy.ndarray): Mask indicating the target region (1 for target, 0 for non-target).
        in_lowest_dose (float): The minimum dose required for the target to be treated.
        single_seed_radiations (list): List of radiation distributions from previously placed seeds.
        seed_sigma (tuple): Spread of radiation from the seed (standard deviation in x, y, z directions).
        seed_avr_dose (float): The average dose delivered by a single seed.
        direc_res (tuple): Resolution of candidate directions (radial, azimuthal, and polar angles).

    Returns:
        best_direc (numpy.ndarray): The best direction for placing the seed to maximize the DVH rate.
        cur_seed_radiation (numpy.ndarray): Radiation distribution contributed by the current seed.
        cur_radiation (numpy.ndarray): Updated radiation distribution after placing the seed.
        cur_DVH_rate (float): The Dose-Volume Histogram (DVH) rate for the current radiation distribution.
    """
    
    # Step 1: Generate candidate directions for placing the seed
    candidate_direcs = get_cone(direc, direc_res[0], direc_res[1], direc_res[2])  # Get all possible directions based on current direction
    
    best_direc = direc.copy()  # Initially set the best direction to the input direction
    cur_seed_radiation = simple_single_dose_calculation(
        cur_radiation.shape, pos, direc, seed_sigma, seed_avr_dose)  # Radiation distribution from current seed in the initial direction
    
    cur_DVH_rate = 0  # Initialize DVH rate (before any radiation contribution)

    # Step 2: Evaluate each candidate direction
    for candidate_direc in candidate_direcs:
        # Calculate DVH rate and radiation from the current seed at the candidate direction
        tmp_DVH_rate, tmp_seed_radiation, tmp_radiation, _ = calculate_tmp_DVH_rate(
            pos, candidate_direc, cur_radiation, mask_volume, in_lowest_dose, single_seed_radiations, seed_sigma, seed_avr_dose)
        
        # Step 3: Update the best direction if the candidate direction results in a higher DVH rate
        if tmp_DVH_rate > cur_DVH_rate:
            cur_DVH_rate = tmp_DVH_rate  # Update DVH rate
            best_direc = candidate_direc.copy()  # Update the best direction
            cur_seed_radiation = tmp_seed_radiation  # Update seed's radiation distribution
            cur_radiation = tmp_radiation  # Update the total radiation map
    
    # Step 4: Return the best direction, radiation from the current seed, and the updated radiation map
    return best_direc, cur_seed_radiation, cur_radiation, cur_DVH_rate


def place_and_evaluate_seed_v2(dose_image, dose_cal_model, pos, direc, cur_radiation, mask_volume, in_lowest_dose, single_seed_radiations, direc_res):
    """
    Place a seed at a specific position in the 3D grid and evaluate its effect on radiation coverage.
    The function tests multiple candidate directions for seed placement and selects the one that maximizes
    the Dose-Volume Histogram (DVH) rate for the target region.

    Parameters:
        dose_image (numpy.ndarray): A 3D array representing the dose image, containing radiation dose information.
        dose_cal_model (object): The model used to calculate the dose distribution from a seed.
        pos (tuple): Position of the seed in the 3D grid (x, y, z).
        direc (numpy.ndarray): The initial direction of the seed (represented as a vector).
        cur_radiation (numpy.ndarray): Current radiation distribution map (3D grid).
        mask_volume (numpy.ndarray): Mask indicating the target region (1 for target, 0 for non-target).
        in_lowest_dose (float): The minimum dose required for the target to be treated.
        single_seed_radiations (list): List of radiation distributions from previously placed seeds.
        direc_res (tuple): Resolution of candidate directions (radial, azimuthal, and polar angles).

    Returns:
        tuple: A tuple containing:
            - best_direc (numpy.ndarray): The best direction for placing the seed to maximize the DVH rate.
            - cur_seed_radiation (numpy.ndarray): Radiation distribution contributed by the current seed.
            - cur_radiation (numpy.ndarray): Updated radiation distribution after placing the seed.
            - cur_DVH_rate (float): The Dose-Volume Histogram (DVH) rate for the current radiation distribution.
    """
    
    # Step 1: Generate candidate directions for placing the seed
    # Generate a list of candidate directions by discretizing the spherical angles around the given initial direction
    candidate_direcs = get_cone(direc, direc_res[0], direc_res[1], direc_res[2])  # Get all possible directions based on current direction
    
    best_direc = direc.copy()  # Initially set the best direction to the input direction
    cur_seed_radiation = single_dose_calculation_v2(
        pos, direc, dose_image, dose_cal_model)  # Radiation distribution from the current seed in the initial direction
    
    cur_DVH_rate = 0  # Initialize DVH rate (before any radiation contribution)

    # Step 2: Evaluate each candidate direction
    # Loop over each candidate direction to evaluate its effectiveness in improving the DVH rate
    for candidate_direc in candidate_direcs:
        # Calculate the DVH rate and radiation from the current seed at the candidate direction
        tmp_DVH_rate, tmp_seed_radiation, tmp_radiation, _ = calculate_tmp_DVH_rate_v2(
            pos, candidate_direc, dose_image, dose_cal_model, mask_volume, in_lowest_dose, single_seed_radiations
        )
        
        # Step 3: Update the best direction if the candidate direction results in a higher DVH rate
        # Compare the DVH rate of the current candidate with the best one so far
        if tmp_DVH_rate > cur_DVH_rate:
            cur_DVH_rate = tmp_DVH_rate  # Update the DVH rate with the new, better value
            best_direc = candidate_direc.copy()  # Update the best direction
            cur_seed_radiation = tmp_seed_radiation  # Update seed's radiation distribution in the new direction
            cur_radiation = tmp_radiation  # Update the total radiation distribution map
    
    # Step 4: Return the best direction, radiation from the current seed, and the updated radiation map
    # After evaluating all candidate directions, return the best direction that maximized the DVH rate
    return best_direc, cur_seed_radiation, cur_radiation, cur_DVH_rate


def deep_learning_optimization(planned_seeds, radiation_volume, cur_radiation, mask_volume, in_lowest_dose, out_highest_dose, DVH_rate, volume, seed_sigma, seed_avr_dose, dl_params, seeds_variation):
    """
    Optimize the placement of seeds using a deep learning-based model. The function uses a neural network
    to find the best configuration of seeds that maximizes the Dose-Volume Histogram (DVH) rate while meeting 
    the dose constraints for the target region.

    Parameters:
        planned_seeds (list): List of seed positions and directions, which are the initial guess for the seed placements.
        radiation_volume (numpy.ndarray): The radiation map of the current environment.
        cur_radiation (numpy.ndarray): The current radiation distribution generated by previously placed seeds.
        mask_volume (numpy.ndarray): A binary mask of the target region (1 for the target area, 0 for non-target).
        in_lowest_dose (float): The minimum dose required to treat the target region.
        out_highest_dose (float): The maximum dose allowed in the surrounding area to avoid excessive radiation exposure.
        DVH_rate (float): The target Dose-Volume Histogram rate that needs to be achieved.
        volume (float): The total volume of the target region (the number of voxels that belong to the target).
        seed_sigma (tuple): Radiation spread in the x, y, and z directions from each seed.
        seed_avr_dose (float): The average dose delivered by each seed.
        dl_params (dict): A dictionary containing deep learning parameters such as learning rate, epochs, 
                          patience, and loss weights. Keys include:
                            - 'lr': Learning rate for optimization
                            - 'epochs': Number of training epochs
                            - 'patience': Patience for early stopping
                            - 'verbose': Verbosity level for training
                            - 'device': The device (e.g., CPU or GPU) for model training
                            - 'loss_weights': Weights for different parts of the loss function

    Returns:
        best_planned_seeds (list): The optimized seed positions and directions that maximize the DVH rate.
        best_single_seed_radiations (list): The radiation distribution for each optimized seed.
        best_DVH_rate (float): The achieved Dose-Volume Histogram rate with the optimized seed placements.
    """
    
    # Step 1: Extract deep learning parameters from the dictionary
    lr = dl_params['lr']
    epochs = dl_params['epochs']
    patience = dl_params['patience']
    verbose = dl_params['verbose']
    device = dl_params['device']
    loss_weights = dl_params['loss_weights']
    
    # Step 2: Initialize the deep learning model for seed placement optimization
    BrachyPlanNet = fitting_model.BrachyPlanNet(dl_params['search_region']).to(device)  # Neural network model
    
    # Optimizer for the neural network
    optimizer = optim.AdamW(BrachyPlanNet.parameters(), lr=lr)  
    # optimizer = optim.Adadelta(BrachyPlanNet.parameters(), lr=lr, rho=0.9, eps=1e-6)
    # optimizer = optim.NAdam(BrachyPlanNet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    # optimizer = optim.SGD(BrachyPlanNet.parameters(), lr=lr, momentum=0.9)

    
    criterion = fitting_model.DoseOptimizationLoss(
        seed_sigma, radiation_volume, in_lowest_dose, out_highest_dose, DVH_rate, seed_avr_dose, device, loss_weights
    ).to(device)  # Loss function for dose optimization, with the required constraints

    # Step 3: Prepare the initial input for the model
    x = from_seeds_to_x(planned_seeds)  # Convert planned seeds into a suitable input format for the model
    # x = torch.ones(x.shape, dtype=torch.float32).to(device) * 0.5
    x = torch.tensor(x, dtype=torch.float32).to(device)  # Convert to a tensor and move to the specified device (CPU/GPU)

    # Step 4: Train the model and find the optimized seed configuration
    best_x, _, seeds_variation = train_model(epochs, x, BrachyPlanNet, optimizer, criterion, [], fitting_model.early_stop(patience, verbose, dl_params['delta']), seeds_variation)
    # `train_model` trains the neural network and returns the best seed placement configuration (best_x)

    # Step 5: Process the output of the trained model to obtain the best seed positions, radiation contributions, and DVH rate
    best_planned_seeds, best_single_seed_radiations, best_DVH_rate = process_best_x(best_x, cur_radiation, mask_volume, in_lowest_dose, volume, seed_sigma, seed_avr_dose)
    # `process_best_x` converts the model's output into actual seed placements, radiation contributions, and DVH rates

    # Step 6: Return the optimized results
    return best_planned_seeds, best_single_seed_radiations, best_DVH_rate, seeds_variation


def generate_dense_rays_from_radiation_volume(radiation_volume, target_val, obs_val, back_val, angle_range, ref_direc, sigma):
    """
    Generate dense ray trajectories from a specified radiation volume using ray tracing.

    This function leverages the `get_rays_from_img_array` method from the `geometry` module to trace
    rays within a 3D volume. It identifies valid paths that originate from the surface of the target 
    region, steer clear of obstacles, and terminate in the background region.

    Args:
        radiation_volume (ndarray): A 3D NumPy array representing the volume of interest:
            - `target_val` marks target regions where rays originate,
            - `obs_val` indicates obstacles that rays must circumvent,
            - `back_val` marks the background where valid rays can end.

        target_val (int or float): The value representing target areas in the volume.

        obs_val (int or float): The value representing obstacles within the volume.

        back_val (int or float): The value marking the background regions within the volume. Rays 
                                 that reach these areas are considered valid.

        angle_range (float): The maximum deviation angle in degrees allowed for rays relative to 
                             the `ref_direc` vector.

        ref_direc (array-like): A reference direction vector. Rays aligned within the specified 
                                `angle_range` from this vector are included for further processing.

    Returns:
        list: A list of valid ray paths, each represented as a sequence of 3D coordinates. These 
              rays are derived from the `get_rays_from_img_array` function based on their behavior 
              within the `radiation_volume`.
    """
    return geometry.get_rays_from_img_array(radiation_volume, target_val, obs_val, back_val, angle_range, ref_direc, sigma)


def draw_radiations(radiation_volume, single_seed_radiations, target_value, threshold=1, sample_fraction=1, save_path=''):
    """
    Visualize and save radiation distributions from multiple seeds.

    This function accumulates radiation distributions from individual seeds, visualizes them
    in 3D, and optionally saves the visualizations as image files.

    Args:
        radiation_volume (numpy.ndarray): 
            A 3D array representing the spatial structure of the target volume.
        single_seed_radiations (list of numpy.ndarray): 
            A list of 3D arrays, where each array represents the radiation distribution 
            from a single seed.
        target_value (int or float): 
            The value representing the target regions in the radiation_volume array.
        threshold (float, optional): 
            Minimum radiation intensity to be visualized. Defaults to 1.
        sample_fraction (float, optional): 
            Fraction of data points to sample for visualization. Defaults to 1 (no sampling).
        save_path (str, optional): 
            Directory path to save the visualization images. Defaults to an empty string.

    Returns:
        None
    """
    
    # Step 1: Initialize an empty array for cumulative radiation distribution
    total_radiation = np.zeros_like(radiation_volume).astype(float)
    
    # Step 2: Iterate through each seed's radiation contribution
    for i, seed_radiation in enumerate(single_seed_radiations):
        # Accumulate the radiation from the current seed into the total distribution
        total_radiation += seed_radiation
        
        # Generate and save a 3D visualization for the current cumulative radiation
        visualizer.get_radiation_3d(
            total_radiation, 
            radiation_volume, 
            target_value, 
            threshold, 
            sample_fraction, 
            f'{save_path}/seed_{i}.png'
        )



def get_close_points(dose_image, radiation_array, ref_direc, target_value, extract_angle):
    """
    Identify points in a radiation array that are close to a specified direction 
    while also matching a given target radiation value within a defined angular range.

    Parameters:
    radiation_array (np.ndarray): A 3D array representing the distribution of radiation values.
    ref_direc (np.ndarray): A unit vector indicating the reference direction for filtering.
    target_value (float): The target radiation value used to filter points in the radiation array.
    extract_angle (float): The maximum angular deviation (in radians) allowed for points to be 
                           considered close to the reference direction.

    Returns:
    tuple: A tuple containing two elements:
        - np.ndarray: An array of coordinates that meet the criteria of proximity to 
          the target value and alignment with the reference direction within the 
          specified angular range.
        - float: The `length` variable, which represents the projection length from the 
          filtered points to the reference direction.

    Steps:
    1. Locate all coordinates in the radiation array that match the specified target value.
    2. Calculate the geometric center of these coordinates.
    3. Define a light source position along the reference direction, positioned far from the center 
       to ensure accurate angular filtering.
    4. Use the 'get_backlit_points' function to filter coordinates that lie within 
       the specified angular range relative to the light source.

    Notes:
    - This function assumes the presence of a 'geometry' module with a 'get_backlit_points' 
      method for filtering coordinates based on the defined angle.
    - The light source is purposely set far away to guarantee precise angular filtering.
    """  
    coordinates = np.argwhere(radiation_array == target_value)
    trans_coordinates = position_transform(dose_image, coordinates)
    # print('trans_coordinates', trans_coordinates)
    length = geometry.projection_length(trans_coordinates, ref_direc)
    coord_center = np.mean(trans_coordinates, axis=0)
    light_source = coord_center + 5 * length * direction_transform(dose_image, ref_direc)
    _, indices = geometry.get_backlit_points(trans_coordinates, light_source, extract_angle)
    close_coordinates = coordinates[indices]

    append_filter = vtk.vtkAppendPolyData()

    for p in close_coordinates:
        # Create a sphere
        # print('physical coordinates', p)
        p = position_transform(dose_image, p)[0]
        # print('physical coordinates', p)
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(p.tolist())
        sphere.SetRadius(1)
        sphere.SetThetaResolution(16)  # Control sphere resolution
        sphere.SetPhiResolution(16)

        sphere.Update()
        append_filter.AddInputData(sphere.GetOutput())

    p = light_source[0]
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(p.tolist())
    sphere.SetRadius(1)
    sphere.SetThetaResolution(16)  # Control sphere resolution
    sphere.SetPhiResolution(16)

    sphere.Update()
    append_filter.AddInputData(sphere.GetOutput())

    p = coord_center
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(p.tolist())
    sphere.SetRadius(1)
    sphere.SetThetaResolution(16)  # Control sphere resolution
    sphere.SetPhiResolution(16)

    sphere.Update()
    append_filter.AddInputData(sphere.GetOutput())
    # Merge all spheres
    append_filter.Update()

    # Write to STL file
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName('close_points.stl')
    stl_writer.SetInputData(append_filter.GetOutput())
    stl_writer.Write()
    
    return close_coordinates, length


def voxel_grid_downsampling(points, voxel_size = 1):
    """
    Perform Voxel Grid Downsampling on a point cloud.

    Parameters:
        points (ndarray): Nx3 array of 3D points.
        voxel_size (float): The size of each voxel grid cell.

    Returns:
        ndarray: Downsampled points after voxel grid processing.
    """
    if points.shape[1] != 3:
        raise ValueError("Input points should have shape (N, 3)")
    
    # Step 1: Quantize point coordinates to voxel grid
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Step 2: Create a dictionary to group points by voxel index
    voxel_dict = {}
    for idx, point in zip(voxel_indices, points):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(point)
    
    # Step 3: Calculate the centroid for each voxel
    downsampled_points = np.array([
        np.mean(voxel_dict[key], axis=0) for key in voxel_dict
    ])
    
    return downsampled_points


def get_depthInfo_from_point(point, array, direc, target_value, background_value, obstacle_value):
    """
    Calculate the depth along a specified direction from a given point in a 3D array.
    
    Parameters:
    - point (tuple or np.ndarray): Coordinates where depth calculation begins in the 3D array.
    - array (np.ndarray): A 3D data array, such as one representing radiation levels or intensity.
    - direc (np.ndarray): A unit vector indicating the direction for depth calculation.
    - target_value: The value representing the target area within the array.
    - background_value: The value representing the background within the array.
    - obstacle_value: The value representing obstacles within the array.

    Returns:
    - tuple: Consisting of a boolean marking the presence of an obstacle and 
             lists of lengths for contiguous segments of target and background values.

    Notes:
    - It employs the `geometry.get_trajectory_info` method to evaluate depth.
    - Ensure `direc` is a unit vector to guarantee accurate path integration.
    - The function integrates values starting from `point`, proceeding along `direc`.

    Example Usage:
    point = (10, 20, 30)
    array = np.random.random((100, 100, 100))
    direc = np.array([1, 0, 0])
    depth = get_depth_from_point(point, array, direc, target_value, background_value, obstacle_value)
    print(f"Depth: {depth}")
    """
    return geometry.get_trajectory_info(point, array, direc, target_value, background_value, obstacle_value)


def init_trajectories_with_depth(close_points, radiation_array, ref_direc, target_value, background_value, obstacle_value, min_depth, max_length):
    """
    Compute the depth of potential trajectories from specified points in a 3D radiation array along a given direction.

    Parameters:
    - close_points (list of np.ndarray): A list of coordinate arrays indicating points in the radiation array.
    - radiation_array (np.ndarray): A 3D array containing data on radiation intensity values.
    - ref_direc (np.ndarray): A unit vector defining the direction for calculating depth.
    - target_value: The numeric representation of target materials or areas within the array.
    - background_value: The numeric representation of non-target areas within the array.
    - obstacle_value: The numeric representation of obstacles that stop or hinder depth calculations.
    - min_depth (float): The minimum accumulated target depth required for a trajectory to be considered valid.

    Returns:
    - list of tuples: A list where each element is a tuple containing:
        - point: The coordinates of the point within the radiation array.
        - ref_direc: The direction used for depth calculations.
        - target_depths: A list of segment lengths corresponding to contiguous target areas encountered.
        - background_depths: A list of segment lengths corresponding to contiguous background areas encountered.
        - max_length: The maximum length of target segments along the reference direction at the point.

    Process:
    1. Iterate over each point in `close_points`.
    2. For each point, use `get_depthInfo_from_point()` to obtain depth information along the defined direction.
    3. Append a tuple for each valid point (with target depth ≥ min_depth and no obstacle detected) to the result list.

    Notes:
    - The `get_depthInfo_from_point()` function handles the integration of values in `radiation_array` in the specified direction,
      effectively accounting for targets, backgrounds, and obstacles.
    - Ensure `ref_direc` is normalized to avoid inaccuracies in calculations.

    Example Usage:
    close_points = [(10, 20, 30), (15, 25, 35)]
    radiation_array = np.random.random((100, 100, 100))
    ref_direc = np.array([0, 0, 1])
    min_depth = 5.0
    result = init_trajectories_with_depth(
        close_points, radiation_array, ref_direc, target_value, background_value, obstacle_value, min_depth
    )
    """
    res = []
    for c_p in close_points:
        obs_sign, target_depths, background_depths = get_depthInfo_from_point(
            c_p, radiation_array, ref_direc, target_value, background_value, obstacle_value
        )
        if not obs_sign:
            target_depth = sum(target_depths)
            # print(target_depth)
            # print('min depth', min_depth)
            if target_depth >= min_depth or max_length < min_depth:
                res.append((c_p, ref_direc, target_depths, background_depths, target_depth))
    return res


def sort_candidate_trajectories_by_depth(trajectories):
    """
    Sorts a list of candidate trajectories based on their depth or score in descending order.

    Parameters:
    trajectories (list): A list of trajectories, where each trajectory is represented as a sublist or tuple. 
                         The last element of each sublist/tuple (i.e., x[-1]) represents the depth or score 
                         that will be used for sorting purposes.

    Returns:
    None: This function sorts the `trajectories` list in-place and does not return anything. 
          As a result, the original list is modified to reflect the new order.

    Notes:
    - The sorting operation is performed in-place, meaning the original `trajectories` list will be reordered 
      rather than a new list being created.
    - The `sort` method of the list is used with a lambda function (`key=lambda x: x[-1]`) as the sorting key. 
      This sorts the trajectories based on the last element which is assumed to be the depth or score.
    - The `reverse=True` parameter ensures the list is sorted in descending order, where trajectories 
      with higher depth or score appear first.
    """
    return sorted(trajectories, key=lambda x: x[-1], reverse=True)


def get_candidate_traj_distance(planned_trajectories, candidate_trajectories, dose_image):
    """
    Calculate the minimum distance from each candidate trajectory to a set of planned trajectories.

    Parameters:
    planned_trajectories -- A list of planned trajectories, where each trajectory consists of a point and a direction vector.
    candidate_trajectories -- A list of candidate trajectories, where each trajectory consists of a point and a direction vector.
    dose_image -- A 3D dose image representing the radiation distribution in the environment.

    Returns:
    A list of minimum distances from each candidate trajectory to the set of planned trajectories.
    """
    # Scale the first element of each sublist in planned_trajectories by spacing.
    # Transform each trajectory's first element (point) by the scaling factor.
    # planned_lines stores the scaled version of planned_trajectories
    planned_lines = []

    # Iterate over each pair of start_point and direction in planned_trajectories
    for (start_point, direction, _, _, _) in planned_trajectories:
        # Reshape start_point to a 1D array and multiply by spacing, then append the result with direction
        planned_lines.append([position_transform(dose_image, np.array(start_point).reshape(-1))[0], direction_transform(dose_image, np.array(direction).reshape(-1))[0]])
    
    # Initialize distances with zeros; one for each candidate trajectory.
    distance = [0] * len(candidate_trajectories)
    
    # Iterate over each candidate trajectory
    for i, candidate_trajectory in enumerate(candidate_trajectories):
        # Calculate and append the minimum distance for each candidate trajectory to the planned lines.
        # The candidate's point is scaled by the spacing factor before calculation.
        distance[i] = geometry.min_distance_to_lines(
            position_transform(dose_image, candidate_trajectory[0])[0],
            direction_transform(dose_image, candidate_trajectory[1])[0],
            planned_lines
        )
    
    # Return the list of minimum distances
    print('distance',distance)
    return distance


def get_candidate_traj_radiation_by_point_count(trajectories, radiation, in_lowest_dose, rate, seed_info, dose_image, distance_map):
    """
    Calculate the effective radiation for each trajectory.

    Parameters:
    trajectories: list of tuples
        Each tuple contains:
        - point: Initial position as a list or array.
        - direction: Direction vector as a list or array.
        - target_depths: List of target depths.
        - background_depths: List of background depths.
    radiation: ndarray
        A multidimensional array representing radiation levels at each point in space.
    in_lowest_dose: bool
        A flag indicating whether the trajectory is within the lowest dose region.
    rate: float
        A scaling factor for the radiation calculation.
    seed_info: tuple
        A tuple containing seed information (seed_id, seed_position, seed_direction, seed_energy).
    dose_image: ndarray
        A 3D dose image representing the radiation distribution in the environment.

    Returns:
    list
        A list of effective radiation values for each trajectory.
    """
    res = []
    for trajectory in trajectories:
        point = np.array(trajectory[0]).reshape(-1)
        direction = np.array(trajectory[1]).reshape(-1)
        
        # Find the index of the direction component with the largest absolute value
        max_index = np.argmax(np.abs(direction))
        
        # Normalize the direction using the largest component for scaling
        update_direction = direction / np.abs(direction[max_index])
        
        target_depths = trajectory[2]
        background_depths = trajectory[3]
        
        # Ensure the lengths of target and background depths are valid
        assert (len(target_depths) >= len(background_depths) - 1) and (len(target_depths) <= len(background_depths) + 1), \
            'Trajectories error, lengths of target_depths and background_depths are not valid'
        
        effective_range = get_available_position(trajectory, [], seed_info, dose_image, distance_map)
        # # Calculate the effective range of steps to consider for each trajectory
        # for i in range(len(target_depths)):
        #     effective_range += list(range(1 + sum(background_depths[:i]) + sum(target_depths[:i]), 
        #                                   target_depths[i] + 1 + sum(background_depths[:i]) + sum(target_depths[:i])))
        
        # Initialize the effective radiation for the current trajectory
        effect_score = 0
        
        # Sum the radiation values at each step in the effective range
        for step in effective_range:
            update_point = point + update_direction * step
            int_coords = tuple(update_point.astype(int))  # Convert to integer coordinates
            effect_score += int(radiation[int_coords] <= in_lowest_dose / rate)
        
        # Append the calculated effective radiation to the results
        res.append(effect_score)
            
    return res


def get_candidate_traj_radiation(trajectories, radiation, in_lowest_dose, seed_info, dose_image, distance_map):
    """
    Calculate the effective radiation for each trajectory.

    Parameters:
    trajectories: list of tuples
        Each tuple contains:
        - point: Initial position as a list or array.
        - direction: Direction vector as a list or array.
        - target_depths: List of target depths.
        - background_depths: List of background depths.
    radiation: ndarray
        A multidimensional array representing radiation levels at each point in space.
    seed_info: list
        A list of seed information, each element is a tuple containing:
        - point: Initial position as a list or array.
        - direction: Direction vector as a list or array.
        - target_depths: List of target depths.
        - background_depths: List of background depths.
    dose_image: SimplrITk image
        A 3D image representing the radiation distribution in the environment.
    distance_map: ndarray
        A multidimensional array representing distance map.

    Returns:
    list
        A list of effective radiation values for each trajectory.
    """
    # rates = [1, 2, 3]
    # for rate in rates:
    #     res = get_candidate_traj_radiation_by_point_count(trajectories, radiation, in_lowest_dose, rate)
    #     if not all(x == 0 for x in res):
    #         return res
    return get_candidate_traj_radiation_by_point_count(trajectories, radiation, in_lowest_dose, 1, seed_info, dose_image, distance_map)



def get_candidate_traj_dir_score(candidate_trajectories, planned_trajectories):
    """
    Calculate the directional alignment score for each candidate trajectory with respect to planned trajectories.

    Parameters:
    candidate_trajectories: list of tuples
        Each tuple contains:
        - point: Initial position as a list or array.
        - direction: Direction vector as a list or array.
    planned_trajectories: list of tuples
        Each tuple contains:
        - point: Initial position as a list or array.
        - direction: Direction vector as a list or array.

    Returns:
    list
        A list of directional alignment scores for each candidate trajectory. The score represents the highest
        cosine similarity between the direction of the candidate trajectory and any of the planned trajectories.
    """
    if len(planned_trajectories) == 0:
        return [1] * len(candidate_trajectories)
    res = []
    for candidate_trajectory in candidate_trajectories:
        direction = np.array(candidate_trajectory[1]).reshape(-1)
        direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
        best_score = 0
        for planned_trajectory in planned_trajectories:
            planned_dire = np.array(planned_trajectory[1]).reshape(-1)
            planned_dire = planned_dire / np.linalg.norm(planned_dire)  # Normalize the planned direction vector
            tmp_score = np.dot(direction, planned_dire)  # Calculate the cosine similarity
            if tmp_score > best_score:
                best_score = tmp_score  # Update the best score if the current score is higher
        res.append(best_score**4)  # Append the best score for the current candidate trajectory
    return res



def get_candidate_traj_edge_distance(trajectories, distance_map):
    """
    Calculate the effective radiation for each trajectory.

    Parameters:
    trajectories: list of tuples
        Each tuple contains:
        - point: Initial position as a list or array.
        - direction: Direction vector as a list or array.
        - target_depths: List of target depths.
        - background_depths: List of background depths.
    radiation: ndarray
        A multidimensional array representing radiation levels at each point in space.

    Returns:
    list
        A list of effective radiation values for each trajectory.
    """
    res = []
    for trajectory in trajectories:
        point = np.array(trajectory[0]).reshape(-1)
        direction = np.array(trajectory[1]).reshape(-1)
        
        # Find the index of the direction component with the largest absolute value
        max_index = np.argmax(np.abs(direction))
        
        # Normalize the direction using the largest component for scaling
        update_direction = direction / np.abs(direction[max_index])
        
        target_depths = trajectory[2]
        background_depths = trajectory[3]
        
        # Ensure the lengths of target and background depths are valid
        assert (len(target_depths) >= len(background_depths) - 1) and (len(target_depths) <= len(background_depths) + 1), \
            'Trajectories error, lengths of target_depths and background_depths are not valid'
        
        effective_range = []
        # Calculate the effective range of steps to consider for each trajectory
        for i in range(len(target_depths)):
            effective_range += list(range(1 + sum(background_depths[:i]) + sum(target_depths[:i]), 
                                          target_depths[i] + 1 + sum(background_depths[:i]) + sum(target_depths[:i])))
        
        # Initialize the effective radiation for the current trajectory
        edge_distance = 0
        min_edge_distance = np.inf
        # Sum the radiation values at each step in the effective range
        for step in effective_range[1:-1]:
            update_point = point + update_direction * step
            int_coords = tuple(update_point.astype(int))  # Convert to integer coordinates
            # edge_distance += distance_map[int_coords]  # Accumulate radiation value
            tmp_edge_distance = distance_map[int_coords]
            if tmp_edge_distance < min_edge_distance:
                min_edge_distance = tmp_edge_distance
        # Append the calculated effective radiation to the results
        if min_edge_distance <= 1:
            min_edge_distance = 0
        res.append(min_edge_distance*len(effective_range))
    
    return res


def adjust_margin_scores(candidate_traj_margin):
    """
    Adjust margin scores by adding the maximum value in the list to each element.

    Parameters:
        candidate_traj_margin (list): 
            A list of margin scores for each trajectory.

    Returns:
        list: 
            A new list where each element is increased by the maximum value in the original list.
    """
    # Step 1: Find the maximum value in the margin scores list
    max_value = max(candidate_traj_margin)
    
    # Step 2: Add the maximum value to each element in the list
    adjusted_margin = [value + max_value for value in candidate_traj_margin]
    
    return adjusted_margin


def get_candidate_traj_weights(candidate_trajectories, planned_trajectories, dose_image, lower_bound, upper_bound, distance_rate):
    """
    Calculate the weights for candidate trajectories based on their distances 
    from a set of planned trajectories. This function uses a smooth distance 
    filter to assign weights that decrease as the distance of a candidate 
    trajectory increases, within the specified bounds.

    Parameters:
    - candidate_trajectories: list
        A list of candidate trajectories that need to be evaluated.
    - planned_trajectories: list
        A list of planned trajectories. If this list is empty, 
        all candidate trajectories will receive equal weights of 1.
    - dose_image: nii image
        A 3D image representing the radiation distribution in the environment.
    - lower_bound: float
        The minimum distance for the distance filter. Below this value, 
        the weight for a candidate trajectory is close to the maximum value.
    - upper_bound: float
        The maximum distance for the distance filter. Beyond this value, 
        the weight for a candidate trajectory approaches a minimum value (e.g., 0).
    - distance_rate: float
        Controls the steepness of the transition in the distance filter. 
        Higher values lead to a sharper drop in weights as the distance increases.

    Returns:
    - candidate_traj_weights: list
        A list of weights corresponding to the candidate trajectories. 
        The weights are calculated based on the distance between each candidate 
        trajectory and the set of planned trajectories, processed through the 
        distance filter.

    Function behavior:
    - If `planned_trajectories` is empty, all candidate trajectories receive 
      equal weights (default weight of 1).
    - If `planned_trajectories` is provided, the Euclidean distances between 
      each candidate trajectory and the planned trajectories are calculated 
      using the helper function `get_candidate_traj_distance`. These distances 
      are then passed through the `geometry.distance_filter` function to 
      compute the weights.
    """
    
    # If no planned trajectories are provided, assign equal weights to all candidates
    if len(planned_trajectories) == 0:
        return [1] * len(candidate_trajectories)
    else:
        # Calculate distances between candidate and planned trajectories
        candidate_distances = get_candidate_traj_distance(planned_trajectories, candidate_trajectories, dose_image)
        
        # Initialize list for candidate trajectory weights
        candidate_traj_weights = []
        
        # Compute weights using the distance filter for each candidate distance
        for candidate_distance in candidate_distances:
            weight = geometry.distance_filter(candidate_distance, lower_bound, upper_bound, distance_rate)
            candidate_traj_weights.append(weight)
        
        # Return the list of trajectory weights
        return candidate_traj_weights
        
        
def select_optimal_trajectory(candidate_trajectories, planned_trajectories, radiation, dose_image, lower_bound, upper_bound, distance_rate, in_lowest_dose, distance_map, seed_info, selected_indices):
    """
    Select the optimal trajectory from a list of candidate trajectories using a multi-factor scoring system.

    Parameters:
        candidate_trajectories (list): 
            A list of candidate trajectories to be evaluated for optimality.
        
        planned_trajectories (list): 
            A list of previously planned trajectories used as a reference for distance-based weighting.
        
        radiation (numpy.ndarray): 
            A 3D array representing radiation distribution, used to calculate dose-related trajectory scores.
        
        dose_image (SimpleITK.Image):
            A SimpleITK image object containing dose distribution data, including metadata like spacing and origin. 
        
        lower_bound (float): 
            Minimum distance threshold for distance-based weighting. Trajectories closer than this threshold receive higher weights.
        
        upper_bound (float): 
            Maximum distance threshold for distance-based weighting. Trajectories farther than this threshold receive lower weights.
        
        distance_rate (float): 
            A smoothing parameter controlling the transition between high and low weights based on distance.
        
        in_lowest_dose (bool): 
            A flag indicating whether to focus on trajectories within the lowest radiation dose regions.
        
        distance_map (numpy.ndarray): 
            A 3D array representing distances to boundaries or obstacles, used for margin-based scoring.
        
        seed_info (dict): 
            Information about the seeds used in the trajectory planning.
        
        selected_indices (list):
            A list of indices of already selected trajectories to avoid re-selection.

    Returns:
        tuple: 
            The optimal trajectory selected from the candidate trajectories based on the highest combined score and its index.

    Description:
        1. **Distance-Based Weight Calculation**: 
           Trajectories are assigned weights based on their distance to planned trajectories using a smooth distance-based filter.
        
        2. **Radiation Scoring**: 
           Evaluate radiation dose contribution for each trajectory to ensure appropriate dose coverage.
        
        3. **Margin Scoring**: 
           Evaluate the edge distance of each trajectory to measure its safety margin based on the `distance_map`.
        
        4. **Score Combination**: 
           Combine weights, radiation scores, and margin scores via element-wise multiplication to calculate the final score for each trajectory.
        
        5. **Optimal Trajectory Selection**: 
           Select the trajectory with the highest combined score.
    """
    # Step 1: Calculate weights based on distances to planned trajectories
    candidate_traj_weights = get_candidate_traj_weights(
        candidate_trajectories, 
        planned_trajectories, 
        dose_image, 
        lower_bound, 
        upper_bound, 
        distance_rate
    ) 
    # print('weights',candidate_traj_weights)
    # Step 2: Calculate radiation scores for each candidate trajectory
    candidate_traj_radiation = get_candidate_traj_radiation(
        candidate_trajectories, 
        radiation, 
        in_lowest_dose,
        seed_info,
        dose_image,
        distance_map
    )
    # print('radiation',candidate_traj_radiation)
    # Step 3: Calculate margin scores based on edge distances
    candidate_traj_margin = get_candidate_traj_edge_distance(
        candidate_trajectories, 
        distance_map
    )
    # print('margin',candidate_traj_margin)
    # Step 4: Calculate direction scores based on planned trajectories
    candidate_direction_score = get_candidate_traj_dir_score(
        candidate_trajectories, 
        planned_trajectories
    )
    # print('direct score',candidate_direction_score)
    adjusted_candidate_traj_margin = candidate_traj_margin
    
    # Step 5: Combine scores using element-wise multiplication
    candidate_traj_scores = (
        np.array(candidate_traj_weights).reshape(-1) * 
        np.array(candidate_traj_radiation).reshape(-1) * 
        np.array(candidate_direction_score).reshape(-1) * 
        np.array(adjusted_candidate_traj_margin).reshape(-1)
    )
    
    # print(candidate_traj_scores)
    
    if np.max(candidate_traj_scores) == 0:
        candidate_traj_scores = (
            np.array(candidate_traj_weights).reshape(-1) * 
            np.array(candidate_traj_radiation).reshape(-1) * 
            np.array(candidate_direction_score).reshape(-1)
        )
        
        if np.max(candidate_traj_scores) == 0:
            candidate_traj_scores = (
                np.array(candidate_traj_weights).reshape(-1) * 
                np.array(candidate_direction_score).reshape(-1)
            )

            if np.max(candidate_traj_scores) == 0:
                candidate_traj_scores = (
                    np.array(candidate_traj_weights).reshape(-1)
                )
                if np.max(candidate_traj_scores) == 0:
                    return None, None
    
    for i, candidate_trajectory in enumerate(candidate_trajectories):
        if len(get_available_position(candidate_trajectory, [], seed_info, dose_image, distance_map)) == 0 or i in selected_indices:
            candidate_traj_scores[i] = 0 

    # Step 6: Select and return the trajectory with the highest score
    return candidate_trajectories[np.argmax(candidate_traj_scores)], np.argmax(candidate_traj_scores)


def put_seeds(radiation_volume, dose_image, dose_cal_model, infer_img_size, radiation, target_value, in_lowest_dose, trajectory, seed_info, DVH_rate, distance_map, image_normalize_min, image_normalize_max, image_normalize_scale):
    """
    Optimize and place radioactive seeds along a predefined trajectory to achieve a target Dose Volume Histogram (DVH) rate.

    This function strategically places radioactive seeds within a treatment volume to ensure a desired dose distribution, 
    leveraging a deep learning model for dose calculation. It iteratively evaluates seed placement positions 
    to maximize radiation coverage while adhering to constraints.

    Parameters:
    ----------
    radiation_volume : numpy.ndarray
        A 3D array representing the treatment area, where specific voxel values mark target regions.

    dose_image : SimpleITK.Image
        A SimpleITK image object containing dose distribution data, including metadata like spacing and origin.

    dose_cal_model : torch.nn.Module
        A pre-trained deep learning model used to predict radiation dose distributions.

    infer_img_size : tuple
        The size of the image to be passed into the deep learning model for inference.

    radiation : numpy.ndarray
        A 3D array representing the current accumulated radiation dose distribution in the treatment area.

    target_value : float
        Voxel intensity value identifying target regions within the `radiation_volume`.

    in_lowest_dose : float
        Minimum acceptable radiation dose threshold for voxels in the target region.

    trajectory : list
        A two-element list defining the seed placement trajectory:
            - trajectory[0]: Starting point (3D coordinates).
            - trajectory[1]: Direction vector for seed placement.

    seed_info : dict
        Dictionary containing parameters of the radiation seed:
            - 'radius' (float): Effective radius of radiation influence for each seed.
            - 'length' (float): Length of the radiation source in the seed.

    DVH_rate : float
        Target Dose Volume Histogram (DVH) rate, representing the fraction of target voxels receiving at least `in_lowest_dose`.

    distance_map : numpy.ndarray
        A 3D array defining spatial constraints, indicating valid regions for seed placement.

    image_normalize_min : float
        Minimum value used for normalizing the dose image before model input.

    image_normalize_max : float
        Maximum value used for normalizing the dose image before model input.

    image_normalize_scale : float
        Scaling factor applied during normalization.

    Returns:
    -------
    tuple:
        seeds : list
            A list of placed seeds, where each seed is represented as a tuple:
            - Position (3D coordinates) of the seed.
            - Direction vector of the seed.

        cur_DVH_rate : float
            The final achieved DVH rate after placing all seeds.

        single_seed_radiations : list
            A list of 3D arrays, each representing the radiation dose contribution from an individual seed.

    Workflow:
    --------
        1. Normalize the trajectory direction.
        2. Create a binary mask for the target region.
        3. Identify valid seed placement positions.
        4. Iterate through valid positions and evaluate DVH improvement.
        5. Select optimal seed positions and update radiation distribution.
        6. Repeat until the target DVH rate is achieved or valid positions are exhausted.
        7. Return seed positions, final DVH rate, and individual seed radiation maps.

    Example:
    --------
        >>> seeds, final_dvh, seed_radiations = put_seeds(
                radiation_volume=volume,
                dose_image=dose_img,
                dose_cal_model=model,
                infer_img_size=(128, 128, 128),
                radiation=current_radiation,
                target_value=1.0,
                in_lowest_dose=0.8,
                trajectory=[(0, 0, 0), (1, 0, 0)],
                seed_info={'radius': 2.0, 'length': 10.0},
                DVH_rate=0.95,
                distance_map=distance_constraints,
                image_normalize_min=0,
                image_normalize_max=255,
                image_normalize_scale=1.0
            )
        >>> print(len(seeds), final_dvh)
        5, 0.96
    """
    # Step 1: Normalize the trajectory direction
    point = np.array(trajectory[0]).reshape(-1)  # Starting point of the trajectory
    direction = np.array(trajectory[1]).reshape(-1)  # Direction vector of the trajectory
    spacing = dose_image.GetSpacing()

    # Identify the axis with the largest direction component
    max_index = np.argmax(np.abs(direction))
    update_direction = direction / np.abs(direction[max_index])

    # Step 2: Create a binary mask for the target region
    mask_volume = (radiation_volume == target_value).astype(float)

    # Step 3: Initialize storage for seed placements and dose contributions
    seeds = []
    single_seed_radiations = []

    # Step 4: Get valid positions for seed placement
    effective_range = get_available_position(trajectory, seeds, seed_info, dose_image, distance_map)
    target_v = np.sum(mask_volume)
    cur_DVH_rate = np.sum(radiation * mask_volume > in_lowest_dose) / target_v

    # Step 5: Iteratively place seeds until the target DVH rate is met
    while len(effective_range) > 0 and cur_DVH_rate < DVH_rate:
        cur_point = copy.deepcopy(point)
        cur_seed_radiation = 0
        cur_radiation = copy.deepcopy(radiation)
        
        # for length in effective_range:
        #     updated_point = point + length * update_direction
        #     tmp_seed_radiation = single_seed_dose_calculation_dl(
        #         np.array(updated_point).astype(int).reshape(-1),
        #         direction,
        #         dose_image,
        #         dose_cal_model,
        #         infer_img_size,
        #         seed_info,
        #         image_normalize_min,
        #         image_normalize_max,
        #         image_normalize_scale
        #     )
        #     tmp_radiation = radiation + tmp_seed_radiation
        #     tmp_DVH_rate = np.sum(tmp_radiation * mask_volume > in_lowest_dose) / target_v

        #     if tmp_DVH_rate >= cur_DVH_rate:
        #         cur_DVH_rate = tmp_DVH_rate
        #         cur_seed_radiation = tmp_seed_radiation
        #         cur_point = copy.deepcopy(updated_point)
        #         cur_radiation = tmp_radiation

        #         if cur_DVH_rate >= DVH_rate:
        #             break

        
        updated_point = point + effective_range[0] * update_direction
        cur_seed_radiation = single_seed_dose_calculation_dl(
            np.array(updated_point).astype(int).reshape(-1),
            direction,
            dose_image,
            dose_cal_model,
            infer_img_size,
            seed_info,
            image_normalize_min,
            image_normalize_max,
            image_normalize_scale
        )
        cur_radiation = radiation + cur_seed_radiation
        cur_DVH_rate = np.sum(cur_radiation * mask_volume > in_lowest_dose) / target_v
        cur_point = copy.deepcopy(updated_point)
        # if tmp_DVH_rate >= cur_DVH_rate:
        #     cur_DVH_rate = tmp_DVH_rate
        #     cur_seed_radiation = tmp_seed_radiation
        #     cur_point = copy.deepcopy(updated_point)
        #     cur_radiation = tmp_radiation

        seeds.append((cur_point, direction))
        single_seed_radiations.append(cur_seed_radiation)
        radiation = cur_radiation
        if cur_DVH_rate >= DVH_rate:
            break
        effective_range = get_available_position(trajectory, seeds, seed_info, dose_image, distance_map)

    return seeds, cur_DVH_rate, single_seed_radiations


def get_available_position(trajectory, seeds, seed_info, dose_image, distance_map, distance_margin = 0):
    """
    Compute the valid positions along a trajectory for seed placement, 
    excluding positions within the influence zone of already-placed seeds.

    Parameters:
        trajectory (list): Information about the placement trajectory:
            - trajectory[0]: The starting point of the trajectory (as a NumPy array).
            - trajectory[1]: The direction vector of the trajectory (as a NumPy array).
            - trajectory[2]: List of target depths (valid regions for seed placement).
            - trajectory[3]: List of background depths (non-target regions between target depths).
        seeds (list): A list of already-placed seeds, where each seed contains spatial position information.
        seed_info (list or tuple): Information about the seed properties:
            - seed_info['length']: The influence range of the seed (radius of the seed's radiation effect).
            - seed_info['num_of_seeds']: Tuple indicating thresholds for adjusting exclusion rate.
        dose_image (simplritk.Image): The dose image used for calculating seed radiation effects.
        distance_map (ndarray): A map indicating distance constraints for valid placement.

    Returns:
        list: A list of valid positions along the trajectory (in the allowed range) 
              with positions influenced by existing seeds excluded.
    """
    spacing = np.array(dose_image.GetSpacing()).reshape(-1)
    # Convert the starting point of the trajectory into a flat NumPy array.
    point = np.array(trajectory[0]).reshape(-1)
    world_p = position_transform(dose_image, point)[0]
    
    # Normalize the trajectory direction vector to obtain a unit vector.
    direction = np.array(trajectory[1]).reshape(-1)
    direction = direction / np.linalg.norm(direction)
    
    # Identify the index of the component in the direction vector with the largest absolute value.
    max_index = np.argmax(np.abs(direction))
    update_direction = direction / np.abs(direction[max_index])

    # Convert the seed's influence radius into the number of voxels along the trajectory direction.
    seed_volume_length = seed_info['length']/spacing[2-max_index] #  spacing[max_index]

    # Extract target depths and background depths from the trajectory.
    target_depths = trajectory[2]
    background_depths = trajectory[3]

    # Initialize the effective range (valid positions).
    effective_range = []
    
    # Compute the valid positions (effective range) based on the target and background depths.
    for i in range(len(target_depths)):
        effective_range += list(range(
            1 + sum(background_depths[:i]) + sum(target_depths[:i]),
            target_depths[i] + 1 + sum(background_depths[:i]) + sum(target_depths[:i])
        ))
    
    # Exclude positions near the boundary of the seed influence zone.
    total_depth = sum(background_depths) + sum(target_depths) 
    rate = seed_info['margin_rate']
    # if len(effective_range) > seed_info['num_of_seeds'][0]:
    #     rate *= 2
    # if len(effective_range) > seed_info['num_of_seeds'][1]:
    #     rate *= 2
    effective_range = [
        x for x in effective_range 
        if np.linalg.norm(position_transform(dose_image, np.array(update_direction * x + point))[0] - world_p) > rate * seed_info['length'] / 2 \
            and np.linalg.norm(position_transform(dose_image, np.array(update_direction * x + point))[0] - world_p) < np.linalg.norm(position_transform(dose_image, np.array(update_direction * total_depth + point)) - world_p) - rate * seed_info['length'] / 2 \
                and distance_map[tuple(np.array(update_direction * x + point).astype(int))] > distance_margin*seed_volume_length
    ]

    # Adjust the effective range to exclude positions influenced by each already-placed seed.
    for seed in seeds:
        # Compute the vector difference between the existing seed and the trajectory's starting point.
        diff = position_transform(dose_image, np.array(seed[0]).reshape(-1))[0] - world_p
        # Calculate the Euclidean distance from the trajectory starting point to the seed.
        distance = np.linalg.norm(diff) # np.abs(diff[max_index])
        # Compute the exclusion range (start and end of the influence zone) for this seed.
        start = distance - seed_info['length']
        end = distance + seed_info['length']
        effective_range = [
            x for x in effective_range 
            if not (start < np.linalg.norm(position_transform(dose_image, np.array(update_direction * x + point))[0] - world_p) < end)
        ]

    # Return the final list of valid positions in the effective range.
    return effective_range

    
# def remove_unproper_seed(traj_seed_radiations, radiation_volume, radiation, out_highest_dose, target_value, background_value, obstacle_value):  
#     """
#     Removes an improper seed from a trajectory to minimize dangerous radiation areas.
    
#     Parameters:
#         traj_seed_radiations (list): A list of tuples, each containing trajectory, seeds, and their radiations.
#         radiation_volume (ndarray): 3D volume representing the radiation values.
#         radiation (ndarray): The current radiation field in the volume.
#         out_highest_dose (float): Threshold for the maximum allowable dangerous radiation dose.
#         target_value (float): Value indicating the target region in the volume.
#         background_value (float): Value indicating the background region in the volume.
#         obstacle_value (float): Value indicating the obstacle region in the volume.

#     Returns:
#         list: Updated trajectory-seed-radiation list with one seed removed to minimize danger.
#     """
#     # Identify dangerous regions based on the volume classifications
#     target_volume = (radiation_volume == target_value).astype(float)
#     background_volume = (radiation_volume == background_value).astype(float)
#     obstacle_volume = (radiation_volume == obstacle_value).astype(float)
#     dangerous_volume = background_volume + obstacle_volume

#     # Initialize variables to track the minimum danger
#     dangerous_num = 1e6  # Large initial value
#     rest_res = traj_seed_radiations.copy()  # Copy of input to modify and return
#     chosen_index = 0
#     chosen_traj = None
#     rest_seeds = None
#     rest_single_seed_radiations = None
#     rest_radiation = radiation.copy()

#     # Iterate through each trajectory and its associated seeds
#     for i, (traj, seeds, single_seed_radiations) in enumerate(traj_seed_radiations):
#         for j, _ in enumerate(seeds):
#             # Temporarily calculate radiation without the current seed
#             tmp_radiation = radiation - single_seed_radiations[j]

#             # Calculate the number of dangerous voxels exceeding the threshold
#             tmp_dangerous_num = np.sum((tmp_radiation * dangerous_volume >= out_highest_dose).astype(float))

#             # Update the chosen seed if this configuration is less dangerous
#             if tmp_dangerous_num < dangerous_num:
#                 dangerous_num = tmp_dangerous_num
#                 chosen_traj = traj
#                 rest_seeds = seeds.copy() 
#                 del rest_seeds[j] # Copy the seeds to modify
#                 rest_single_seed_radiations = single_seed_radiations.copy()
#                 del rest_single_seed_radiations[j]
#                 chosen_index = i
#                 rest_radiation = tmp_radiation

#     # Update the result with the modified trajectory and seeds
#     rest_res[chosen_index] = [chosen_traj, rest_seeds, rest_single_seed_radiations]
    
#     return rest_res, rest_radiation


def remove_unproper_seed(traj_seed_radiations, radiation_volume, radiation, out_highest_dose, target_value, background_value, obstacle_value):
    """
    Remove an improper seed from a trajectory to minimize radiation exposure in dangerous regions.
    
    Parameters:
        traj_seed_radiations (list):
            A list of tuples, each containing:
                - traj: Trajectory information.
                - seeds: List of seed positions.
                - single_seed_radiations: Radiation contribution from each seed.
        radiation_volume (ndarray):
            3D array representing different regions (target, background, obstacle).
        radiation (ndarray):
            3D array representing the current radiation dose distribution.
        out_highest_dose (float):
            Threshold for the maximum allowable radiation dose.
        target_value (float):
            Identifier for target regions in the radiation volume.
        background_value (float):
            Identifier for background regions in the radiation volume.
        obstacle_value (float):
            Identifier for obstacle regions in the radiation volume.
    
    Returns:
        tuple:
            - rest_res (list): Updated trajectory-seed-radiation list with one seed removed.
            - rest_radiation (ndarray): Updated radiation field after removing the selected seed.
    
    Description:
        - Identify dangerous regions in the radiation volume.
        - Evaluate each seed's contribution to dangerous radiation levels.
        - Probabilistically select and remove a seed based on its contribution to dangerous regions.
        - Update the radiation field and seed list accordingly.
    """
    # Classify regions based on volume values
    # target_volume = (radiation_volume == target_value).astype(float)
    background_volume = (radiation_volume == background_value).astype(float)
    obstacle_volume = (radiation_volume == obstacle_value).astype(float)
    dangerous_volume = background_volume + obstacle_volume

    # Initialize tracking variables
    all_dangerous_nums = []
    all_dangerous_num = 0

    # Evaluate each seed's contribution to dangerous regions
    for i, (_, seeds, single_seed_radiations) in enumerate(traj_seed_radiations):
        dangerous_nums = []
        for j, _ in enumerate(seeds):
            tmp_radiation = radiation - single_seed_radiations[j]
            tmp_dangerous_num = np.sum((tmp_radiation * dangerous_volume >= out_highest_dose).astype(float))
            all_dangerous_num += tmp_dangerous_num
            dangerous_nums.append(tmp_dangerous_num)
        all_dangerous_nums.append(dangerous_nums)
    
    # Select a seed to remove probabilistically
    chosen_dangerous_num = all_dangerous_num * np.random.uniform(0, 1)
    accumulate_num =all_dangerous_num
    chosen_i, chosen_j = 0, 0
    chosen_sign = False
    
    for i, dangerous_nums in enumerate(all_dangerous_nums):
        for j, dangerous_num in enumerate(dangerous_nums):
            accumulate_num -= dangerous_num
            if accumulate_num <= chosen_dangerous_num:
                chosen_i, chosen_j = i, j
                chosen_sign = True
                break
        if chosen_sign:
            break
    
    # Remove the selected seed and update the trajectory
    rest_seeds = copy.deepcopy(traj_seed_radiations[chosen_i][1])
    del rest_seeds[chosen_j]
    
    rest_radiation = radiation - traj_seed_radiations[chosen_i][2][chosen_j]
    rest_single_seed_radiations = copy.deepcopy(traj_seed_radiations[chosen_i][2])
    del rest_single_seed_radiations[chosen_j]
    
    rest_res = copy.deepcopy(traj_seed_radiations)
    rest_res[chosen_i] = (traj_seed_radiations[chosen_i][0], rest_seeds, rest_single_seed_radiations)
    
    return rest_res, rest_radiation



def remove_seed_sequentially(traj_seed_radiations, all_seeds, itera, radiation):
    """
    Remove a specific seed from the trajectory sequentially.

    Parameters:
        traj_seed_radiations (list): A list of tuples, each containing trajectory, seeds, and their radiations.
        all_seeds (list): A list of all seeds.
        itera (int): The index of the seed to be removed in the all_seeds list.
        radiation (ndarray): The current radiation field in the volume.

    Returns:
        tuple: Updated trajectory-seed-radiation list with the specified seed removed and the updated radiation field.
    """
    seed = all_seeds[itera]
    chosen_i, chosen_j = 0, 0
    chosen_sign = False
    
    # Find the matching seed position
    for i, (_, seeds, _) in enumerate(traj_seed_radiations):
        for j, tmp_seed in enumerate(seeds):
            if np.array_equal(tmp_seed[0], seed[0]) and np.array_equal(tmp_seed[1], seed[1]):
                chosen_i, chosen_j = i, j
                chosen_sign = True
                break
        if chosen_sign:
            break
    
    # Remove the selected seed and update the radiation
    rest_seeds = copy.deepcopy(traj_seed_radiations[chosen_i][1])
    del rest_seeds[chosen_j]
    
    rest_radiation = radiation - traj_seed_radiations[chosen_i][2][chosen_j]
    rest_single_seed_radiations = copy.deepcopy(traj_seed_radiations[chosen_i][2])
    del rest_single_seed_radiations[chosen_j]
    
    rest_res = copy.deepcopy(traj_seed_radiations)
    rest_res[chosen_i] = (traj_seed_radiations[chosen_i][0], rest_seeds, rest_single_seed_radiations)
    
    return rest_res, rest_radiation


                        
def add_proper_seed(traj_seed_radiations, radiation_volume, radiation, dose_image, dose_cal_model, infer_img_size, 
                    in_lowest_dose, out_highest_dose, target_value, background_value, obstacle_value, 
                    DVH_rate, seed_info, distance_map, image_normalize_min, image_normalize_max, image_normalize_scale):
    """
    Strategically add a radioactive seed to improve dose coverage and minimize overexposure.

    This function evaluates potential seed placements along predefined trajectories to select 
    the optimal position, enhancing the Dose Volume Histogram (DVH) coverage for target regions 
    while avoiding over-irradiation of non-target regions such as background or obstacles.

    Parameters:
    ----------
    traj_seed_radiations : list
        A list of trajectory records, each containing:
            - trajectory (tuple): (start_point, direction_vector)
            - seeds (list): List of current seed positions along the trajectory.
            - seed_radiations (list): Radiation dose contributions from each seed.

    radiation_volume : numpy.ndarray
        A 3D array indicating voxel classification:
            - target regions (target_value)
            - background regions (background_value)
            - obstacle regions (obstacle_value)

    radiation : numpy.ndarray
        A 3D array representing the current cumulative radiation dose distribution.

    dose_image : SimpleITK.Image
        A medical dose image with spatial metadata, including spacing and origin.

    dose_cal_model : torch.nn.Module
        A pre-trained deep learning model for predicting radiation dose distribution.

    infer_img_size : tuple
        Input size required by the dose calculation model.

    in_lowest_dose : float
        Minimum acceptable radiation dose for target voxels.

    out_highest_dose : float
        Maximum acceptable radiation dose for non-target voxels (e.g., background and obstacles).

    target_value : float
        Label indicating target regions in `radiation_volume`.

    background_value : float
        Label indicating background regions in `radiation_volume`.

    obstacle_value : float
        Label indicating obstacle regions in `radiation_volume`.

    DVH_rate : float
        Target Dose Volume Histogram (DVH) coverage rate, representing the fraction of target voxels 
        receiving sufficient dose.

    seed_info : dict
        Dictionary containing seed parameters:
            - 'radius' (float): Radiation influence radius.
            - 'length' (float): Physical length of the radiation source.

    distance_map : numpy.ndarray
        A 3D array defining valid spatial regions for seed placement, ensuring constraints are followed.

    image_normalize_min : float
        Minimum intensity value used for dose image normalization.

    image_normalize_max : float
        Maximum intensity value used for dose image normalization.

    image_normalize_scale : float
        Scaling factor applied during normalization.

    Returns:
    -------
    tuple:
        traj_seed_radiations : list
            Updated list of trajectory records with the newly added seed and its radiation contribution.

        final_radiation : numpy.ndarray
            Updated cumulative radiation dose distribution.

        success : bool
            `True` if a suitable seed placement was successfully found and added, `False` otherwise.

    Workflow:
    --------
    1. Identify target, background, and obstacle regions based on voxel classification.
    2. Determine valid seed placement ranges along each trajectory.
    3. Evaluate potential seed placements:
        - Calculate radiation contribution at each point.
        - Assess DVH coverage and non-target overexposure.
    4. Select the optimal seed placement minimizing non-target exposure.
    5. Update the trajectory with the new seed and its radiation contribution.
    """
    # Step 1: Define target and dangerous regions
    target_volume = (radiation_volume == target_value).astype(float)
    target_v = np.sum(target_volume)

    background_volume = (radiation_volume == background_value).astype(float)
    obstacle_volume = (radiation_volume == obstacle_value).astype(float)
    dangerous_volume = background_volume + obstacle_volume

    # Step 2: Calculate effective placement ranges for each trajectory
    effective_ranges = []
    for res in traj_seed_radiations:
        effective_ranges.append(get_available_position(res[0], res[1], seed_info, dose_image, distance_map, 2))

    # Step 3: Initialize variables for optimal seed selection
    cur_dangerous_num = float('inf')
    chosen_index = 0
    chosen_seed_radiation = None
    chosen_seed = None
    final_radiation = radiation

    # Step 4: Evaluate potential seed placements
    for i, effective_range in enumerate(effective_ranges):
        traj = traj_seed_radiations[i][0]

        if len(effective_range) != 0:
            point = np.array(traj[0]).reshape(-1)
            direction = np.array(traj[1]).reshape(-1)
            max_index = np.argmax(np.abs(direction))
            update_direction = direction / np.abs(direction[max_index])

            for length in effective_range:
                updated_point = point + length * update_direction

                tmp_seed_radiation = single_seed_dose_calculation_dl(
                    np.array(updated_point).astype(int).reshape(-1), 
                    direction, 
                    dose_image,
                    dose_cal_model,
                    infer_img_size,
                    seed_info,
                    image_normalize_min,
                    image_normalize_max,
                    image_normalize_scale
                )
                
                tmp_radiation = radiation + tmp_seed_radiation
                cur_DVH_rate = np.sum(tmp_radiation * target_volume >= in_lowest_dose) / target_v

                if cur_DVH_rate >= DVH_rate:
                    tmp_dangerous_num = np.sum((tmp_radiation * dangerous_volume >= out_highest_dose).astype(float))

                    if tmp_dangerous_num < cur_dangerous_num:
                        cur_dangerous_num = tmp_dangerous_num
                        chosen_index = i
                        chosen_seed_radiation = tmp_seed_radiation
                        chosen_seed = [updated_point, direction]
                        final_radiation = tmp_radiation

    # Step 5: Update trajectory with the selected seed
    if chosen_seed is not None:
        traj_seed_radiations[chosen_index][1].append(chosen_seed)  
        traj_seed_radiations[chosen_index][2].append(chosen_seed_radiation)  
        return traj_seed_radiations, final_radiation, True  
    else:
        return traj_seed_radiations, final_radiation, False


def replan(traj_seed_radiations, radiation_volume, radiation, dose_image, dose_cal_model, infer_img_size, in_lowest_dose, target_value, 
           background_value, obstacle_value, seed_info, distance_map, image_normalize_min, image_normalize_max, image_normalize_scale):
    """
    Optimize and replan radioactive seed placements along predefined trajectories to maximize radiation coverage 
    in target regions while minimizing exposure in non-target (dangerous) regions.

    This function evaluates potential seed placements along each trajectory, selects optimal positions based on Dose 
    Volume Histogram (DVH) improvement, and updates the cumulative radiation distribution accordingly.

    Parameters:
    ----------
    traj_seed_radiations : list
        A list of tuples representing trajectories, associated seeds, and their radiation profiles.
        Each tuple contains:
            - trajectory (tuple): (start_point, direction_vector)
            - seeds (list): List of seed positions along this trajectory.
            - seed_radiations (list): Radiation dose distributions from the placed seeds.

    radiation_volume : numpy.ndarray
        A 3D array representing the anatomical region, where voxel values indicate target, background, or obstacles.

    radiation : numpy.ndarray
        A 3D array representing the current cumulative radiation dose distribution across the region.

    dose_image : SimpleITK.Image
        Image representing the dose distribution, used for precise radiation simulation.

    dose_cal_model : torch.nn.Module
        Deep learning model for predicting radiation dose distribution from seed placement parameters.

    in_lowest_dose : float
        Minimum acceptable radiation dose threshold for effective coverage in target regions.

    target_value : float
        Label value identifying target voxels in `radiation_volume`.

    background_value : float
        Label value identifying background voxels in `radiation_volume`.

    obstacle_value : float
        Label value identifying obstacle voxels in `radiation_volume`.

    seed_info : dict
        Dictionary describing the properties of the radioactive seeds:
            - 'radius': Effective radiation influence radius per seed.
            - 'length': Physical length of the radiation source.

    distance_map : numpy.ndarray
        A 3D array defining valid spatial regions for seed placement, used to enforce geometric constraints.

    image_normalize_min : float
        Minimum value for normalizing input dose image data.

    image_normalize_max : float
        Maximum value for normalizing input dose image data.

    Returns:
    -------
    tuple:
        traj_seed_radiations : list
            Updated list of trajectories with newly added seeds and their radiation contributions.

        final_DVH_rate : float
            The final Dose Volume Histogram (DVH) coverage rate, representing the proportion of target voxels 
            receiving the minimum acceptable dose.

        final_radiation : numpy.ndarray
            Updated cumulative radiation dose distribution after optimizing seed placements.

    Workflow:
    --------
        1. **Identify Key Regions:** Define target, background, and obstacle regions based on voxel labels.
        2. **Evaluate Placement Ranges:** Calculate valid seed placement positions along each trajectory.
        3. **Simulate Placements:** Evaluate potential seed placements along trajectories using dose simulation models.
        4. **Optimize DVH Rate:** Identify seed placement that maximizes DVH rate while minimizing exposure to dangerous regions.
        5. **Update Trajectories:** Update the selected trajectory with the optimal seed placement.
        6. **Recalculate DVH:** Compute the final DVH coverage rate and cumulative radiation distribution.

    Example:
    --------
        >>> updated_trajs, final_DVH, final_radiation = replan(
                traj_seed_radiations=trajectories,
                radiation_volume=volume,
                radiation=current_radiation,
                dose_image=dose_img,
                dose_cal_model=model,
                in_lowest_dose=0.8,
                target_value=1.0,
                background_value=0.0,
                obstacle_value=-1.0,
                seed_info={'radius': 2.0, 'length': 10.0},
                spacing=(1.0, 1.0, 1.0),
                distance_map=distance_constraints,
                image_normalize_min=0,
                image_normalize_max=255
            )
        >>> print(final_DVH)
        0.95
    """
    # --- Step 1: Identify Key Regions in the Radiation Volume ---
    target_volume = (radiation_volume == target_value).astype(float)
    target_v = np.sum(target_volume)  # Total number of target voxels
    
    # background_volume = (radiation_volume == background_value).astype(float)
    # obstacle_volume = (radiation_volume == obstacle_value).astype(float)
    # dangerous_volume = background_volume + obstacle_volume  # Define dangerous regions

    # --- Step 2: Calculate Effective Seed Placement Ranges for Each Trajectory ---
    effective_ranges = []
    for res in traj_seed_radiations:
        effective_ranges.append(get_available_position(res[0], res[1], seed_info, dose_image, distance_map, 2))

    # --- Step 3: Search for the Optimal Seed Placement ---
    chosen_index = 0  # Index of the selected trajectory
    chosen_seed_radiation = None  # Selected seed radiation profile
    chosen_seed = None  # Selected seed position and direction
    final_DVH_rate = np.sum((radiation * target_volume) >= in_lowest_dose) / target_v  # Track the achieved DVH rate
    final_radiation = radiation  # Store the updated radiation distribution

    # Evaluate seed placement along each trajectory
    for i, effective_range in enumerate(effective_ranges):
        traj = traj_seed_radiations[i][0]  # Current trajectory

        if len(effective_range) != 0:
            point = np.array(traj[0]).reshape(-1)  # Start point of the trajectory
            direction = np.array(traj[1]).reshape(-1)  # Direction vector
            max_index = np.argmax(np.abs(direction))
            update_direction = direction / np.abs(direction[max_index])  # Normalize by the dominant axis

            # Evaluate potential seed placements along the trajectory
            for length in effective_range:
                updated_point = point + length * update_direction
                
                tmp_seed_radiation = single_seed_dose_calculation_dl(
                    np.array(updated_point).astype(int).reshape(-1), 
                    direction, 
                    dose_image,
                    dose_cal_model,
                    infer_img_size,
                    seed_info,
                    image_normalize_min,
                    image_normalize_max,
                    image_normalize_scale
                )
                
                tmp_radiation = radiation + tmp_seed_radiation

                # Calculate DVH rate for the target region
                cur_DVH_rate = np.sum((tmp_radiation * target_volume) >= in_lowest_dose) / target_v

                # Update optimal seed placement if DVH rate improves
                if cur_DVH_rate >= final_DVH_rate:
                    chosen_index = i
                    chosen_seed_radiation = tmp_seed_radiation
                    chosen_seed = [updated_point, direction]
                    final_DVH_rate = cur_DVH_rate
                    final_radiation = tmp_radiation

    # --- Step 4: Update Trajectory with the Chosen Seed Placement ---
    if chosen_seed is not None:
        traj_seed_radiations[chosen_index][1].append(chosen_seed)  
        traj_seed_radiations[chosen_index][2].append(chosen_seed_radiation)  
        return traj_seed_radiations, final_DVH_rate, final_radiation, True
    else:
        return traj_seed_radiations, final_DVH_rate, final_radiation, False
