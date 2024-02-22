# Light-driven multidirectional bending in artificial muscles

Provide a brief description of what your project does and its purpose. This introduction should give users a clear idea of the project's goals and the problems it aims to solve.

## Getting Started

These instructions will guide you through getting a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

This project requires Miniconda to manage dependencies and create an isolated environment. If you don't have Miniconda installed, follow the instructions on the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) to download and install it.

### Installation

First, clone the repository to get a local copy of the project. Then, navigate to the project directory.

```bash
git clone https://github.com/p3d2/LTCAM.git
cd LTCAM 
```

### Setting Up the Environment

This project comes with an env.yml file that specifies all the necessary dependencies and their versions. To create a Miniconda environment with these dependencies, use the following command:

```mamba env create -f env.yml -p env/```

This command creates a new Conda environment named LTCAM and installs all the required packages listed in env.yml.

Activating the Environment
Before running the scripts, ensure that the LTCAM environment is activated:

```conda activate ./env```

### Add raw data

Download the raw data files from Zenodo (link) and 

## Description of Python Files

In this project, multiple Python files are utilized to process raw data, track movements, and generate plots and videos. Below is an overview of what each script does:

### `LTCAMAnalysis.py`

- **Functionality**: This script initiates the data processing workflow by creating a "processed" folder, within which a subfolder named "LTCAM" is generated. Inside the "LTCAM" folder, it further creates subfolders named "Measurements", "Plots", and "Videos".
- **Purpose**: It analyzes raw data to make measurements and produce an infrared video, organizing the output into the respective subfolders.

### `Plots 1.py` and `Plots 2.py`

- **Functionality**: These scripts are designed to generate plots based on the measurements data obtained from running `LTCAMAnalysis.py`.
  - `Plots 1.py` should be run after `LTCAMAnalysis.py` has been executed with `c` values ranging from 0 to 5.
  - `Plots 2.py` requires `LTCAMAnalysis.py` to be run with `c` values from 6 to 13.
- **Purpose**: They store the generated plots in the "Plots" subfolder within the "LTCAM" directory, visually representing the analysis results.
- **Configuration**: `dir_path` Update this variable to point to the directory where the data files folder is located. `c` - This integer value should be set between 0 and 13, indicating the specific file to be processed for LTCAM analysis.

### `ButterflyTrack.py`

- **Functionality**: Similar to `LTCAMAnalysis.py`, this script creates a "processed" folder with a specific subfolder named "Butterfly", which then contains "Measurements", "Plots", and "Videos" subfolders.
- **Purpose**: It tracks the movement of a butterfly textile, producing a video that showcases this tracking alongside the relevant measurements and plots.
- **Configuration**: `dir_path` Update this variable to point to the directory where the data files folder is located. `c` - This integer value should be set between 0 and 1, indicating the specific file to be processed for RotatingTrack analysis.

### `RotatingTrack.py`

- **Functionality**: This script processes raw data from videos of a LTCAM on a rotating platform. It creates a "processed" folder with a "Rotating" subfolder, which also includes "Measurements", "Plots", and "Videos" subfolders.
- **Purpose**: It performs circumnotation tracking measurements and generates an infrared video of the LTCAM's movement.
- **Configuration**: `dir_path` Update this variable to point to the directory where the data files folder is located. `c` - This integer value should be set between 0 and 5, indicating the specific file to be processed for RotatingTrack analysis.

### `Plots 3.py`

- **Functionality**: Tailored to work with data from `RotatingTrack.py`, this script needs to be run after `RotatingTrack.py` has processed data with `c` values from 0 to 5.
- **Purpose**: It analyzes the circumutation movement, specifically focusing on the position of the LTCAM's end, and generates corresponding plots that are stored in the "Plots" subfolder of the "Rotating" directory.


## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

