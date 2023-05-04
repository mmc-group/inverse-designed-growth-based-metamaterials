# Inverse-designed Growth-based Metamaterials

This repository contains the codes to train the deep learning framework for inverse-designing growth-based cellular metamaterials. Please follow the publication https://doi.org/10.1016/j.mechmat.2023.104668 for implementation details.

## Citation

If you use this code please cite the following publication: Sikko van ’t Sant, Prakash Thakolkaran, Jonàs Martínez, Siddhant Kumar, Inverse-designed growth-based cellular metamaterials, Mechanics of Materials, 2023, 104668, ISSN 0167-6636, https://doi.org/10.1016/j.mechmat.2023.104668.

## Data generation

The dataset used in this project can be downloaded here: https://data.4tu.nl/datasets/94939dc6-9f51-4f4a-a84b-ce660db0e7e0/1

### Design generation

To generate the growth-based cellular metamaterials designs, we used the following open-source code:
https://github.com/mfx-inria/auxeticgrowthprocess2d

The input parameter file (named `input_file.txt`) of any given design should be as follows:

```sh
name=./star_shape
radial_spans=r1,r2,r3,r4,r5,r6,r7
max_growth_length_radial_spans=R1,R2,R3,R4,R5,R6,R7
point_process=File
interpolation_type=PolarCubic
image_size=K
symmetry_type=NoSymmetry
plot_starshaped_ppm=false
plot_starshaped_pdf=true
plot_sites_pdf=false
save_porous_material_ppm=true
save_results_txt=false
plot_porous_material_sites_png=true
num_growth_plots=0
save_non_regularized=false
filename_points=./points_lattice.txt
```

Where the `r1,..,r7` correspond to the spans defining the star-shaped set $\mathcal{S}$ and `R1,...,R7` correspond to the spans defining the star-shaped set $\mathcal{S}^*$. Refer to the value ranges as described in the publication.

Moreover, the `points_lattice.txt` file contains:
```sh
0.5,0.5
```
This defines the position of the nucleus in the domain (i.e., in the middle of the domain).

To generate a black-and-white image of the unit cell, run the following (after building the `growthProcess2d` executable from the aforementioned repository):
```sh
./growthprocess2d input_file.txt
```

### FFT-based homogenization

To efficiently homogenize our metamaterial designs we utilised the following open-source code:
https://github.com/sbrisard/janus

The unit cell images have to be converted into 2D arrays containing `0`'s (void phase) and `1`'s (solid phase) before the homogenization. 

## Software requirements
- Python (tested on version 3.7.1)
- Python packages:
    - PyTorch (tested without CUDA)
    - NumPy
    - Pandas
    
## Usage

```sh
python main.py
```

## File descriptions
- main.py: main file to be executed and contains training protocols
- model.py: functions for creating neural network models
- loadDataset.py: functions for loading data from data.csv
- errorAnalysis.py: functions for post-processing and error analysis
- normalization.py: functions for normalization of features (inputs to neural networks)
- parameters.py: contains all parameters and hyper-parameters for neural network architectures and training protocols

## Outputs
After training is over, outputs will be available in the following directories:
- ./models/ : contains trained models
- ./loss-history/ : contains loss history during training
