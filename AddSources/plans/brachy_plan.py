import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from . import config
from . import core
from . import utilizations
import numpy as np
from . import visualizer


def brachy_plan(ctimage, ctvimage, args):
    """
    Generate a brachytherapy radiation treatment plan by optimizing seed trajectories and placements, 
    followed by 3D visualization and exporting the results as STL files.

    Parameters:
        args (Namespace): 
            Configuration object containing parameters for radiation planning, seed placement, 
            dose constraints, and file paths.

    Workflow:
        1. **Image and Radiation Data Preparation:**  
            - Load tumor image (NIfTI format) and extract voxel spacing.  
            - Generate the radiation planning volume based on specified thresholds.  
            - Determine a reference direction for trajectory initialization.  

        2. **Trajectory Initialization:**  
            - Generate candidate trajectories for seed placement based on geometry and dose constraints.  

        3. **Optimal Plan Generation:**  
            - Optimize seed placements along the selected trajectories to meet dose coverage requirements.  
            - Refine the plan iteratively to minimize radiation exposure to healthy tissues.  

        4. **3D Visualization and Export:**  
            - Convert seed positions and directions into 3D STL files for visualization and further analysis.  

    Returns:
        list:
            - plan_res (list): Final optimized seed placement plan, including trajectories, seeds, and dose information.
    
    Output Files:
        - STL files representing the 3D position and direction of each seed are saved in the `./output` folder.
    """

    # --- Stage 1: Image and Radiation Data Preparation ---
    # Load the tumor image
    dose_image = utilizations.normalize_dose_image(ctimage, args.image_normalize[0], args.image_normalize[1], args.image_normalize[0], args.image_normalize[1])
    # target_image = utilizations.read_nii_image(args.target_image_path)
    
    # Generate the radiation planning volume based on threshold values
    radiation_volume = utilizations.get_planning_volume_array(
        ctvimage,
        args.radiation_array_params['target_value'],
        args.radiation_array_params['obstacle_value'],
        args.radiation_array_params['background_value'],
    )
    
    # Determine the reference direction for trajectory planning
    # ref_direc = utilizations.get_reference_direction(
    #     radiation_volume,
    #     args.radiation_array_params['target_value']
    # )
    ref_direc = np.array([0, 1, -2.5])  # Manually set direction if needed
    
    # --- Stage 2: Trajectory Initialization ---
    init_tracjectories = core.init_plan(
        dose_image,
        radiation_volume,
        ref_direc,
        args.direc_resolution,
        args.radiation_array_params['backlit_angle'],
        args.radiation_array_params['target_value'],
        args.radiation_array_params['background_value'],
        args.radiation_array_params['obstacle_value'],
        args.seed_info['length']
    )
    
    # --- Stage 3: Optimal Plan Generation ---
    plan_res, init_res,sum_image = core.optimal_plan(
        init_tracjectories,
        radiation_volume,
        dose_image,
        args.dl_params,
        args.distance_filtter['lower_bound'],
        args.distance_filtter['upper_bound'],
        args.distance_filtter['distance_rate'],
        args.radiation_array_params['target_value'],
        args.radiation_array_params['obstacle_value'],
        args.radiation_array_params['background_value'],
        args.radiation_array_params['infer_img_size'],
        args.in_lowest_energy,
        args.out_highest_energy,
        args.DVH_rate,
        args.seed_info,
        args.iter_rate,
        args.image_normalize[0], 
        args.image_normalize[1],
        args.image_normalize[2]
    )
    
    # --- Stage 4: 3D Visualization and Export ---
    # planned_seeds = []
    # planned_seed_doses = []
    # for res in plan_res:
    #     planned_seeds.append(res[1])
    #     planned_seed_doses.append(res[2])
    
    # utilizations.create_folder_if_not_exists('./output')
    
    # for i, seeds in enumerate(planned_seeds):
    #     for j, seed in enumerate(seeds):
    #         # Transform the seed position into physical space
    #         pos = utilizations.position_transform(dose_image, seed[0])[0]
            
    #         # Transform and normalize the seed direction
    #         direction = utilizations.direction_transform(dose_image, seed[1])[0]
            
    #         # Save the seed geometry as an STL file
    #         visualizer.save_polydata_as_stl(
    #             visualizer.get_seed_polydata(
    #                 pos,
    #                 direction,
    #                 args.seed_info['length'],
    #                 args.seed_info['radius']
    #             ),
    #             f'./output/seed_{i}_{j}.stl'
    #         )
    #         visualizer.save_numpy_as_nii(planned_seed_doses[i][j], dose_image, f'./output/dose_{i}_{j}.nii.gz')
            
    
    # Uncomment the following block if initial seed plan visualization is needed
    # planned_seeds = []
    # for res in init_res:
    #     planned_seeds.append(res[1])

    # for i, seeds in enumerate(planned_seeds):
    #     for j, seed in enumerate(seeds):
    #         pos = utilizations.position_transform(dose_image, seed[0])[0]
    #         direction = utilizations.direction_transform(dose_image, seed[1])[0]
            
    #         visualizer.save_polydata_as_stl(
    #             visualizer.get_seed_polydata(
    #                 pos,
    #                 direction,
    #                 args.seed_info['length'],
    #                 args.seed_info['radius']
    #             ),
    #             f'./output/init_seed_{i}_{j}.stl'
    #         )
    
    
    # plan_res structure:
    # Example: [[trajectory1], [trajectory2], [trajectory3], ...]
    # trajectory1 = {'dire': direction, 'seeds': [[pos1, direct1], [pos2, direct2], ...]}
    
    return plan_res, sum_image, dose_image


if __name__ == '__main__':
    brachy_plan(config.setting())
