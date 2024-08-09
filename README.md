# Light-driven multidirectional bending in artificial muscles
<a href="https://doi.org/10.5281/zenodo.10695914"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10695914.svg" alt="DOI"></a>

Supporting code for the research article (click on the image to access the full article)

<a href="https://onlinelibrary.wiley.com/doi/10.1002/adma.202405917"><img src="https://raw.githubusercontent.com/p3d2/LTCAM/main/images/toc_with_a_cat.png">

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

```bash
mamba env create -f env.yml -p env/
```

This command creates a new Conda environment named LTCAM and installs all the required packages listed in env.yml.

Activating the Environment
Before running the scripts, ensure that the LTCAM environment is activated:

```bash
conda activate ./env
```

### Obtaining and Preparing Raw Data

Before running the analysis scripts, it is essential to download the raw data files and place them in the appropriate directory within the project repository. The raw data for this project is hosted on Zenodo. Follow the steps below to prepare your data for analysis:

1. Visit the Zenodo link https://zenodo.org/records/10695914
2. Download the dataset to your local machine.
3. Extract the files.
4. Create a folder named `data` in the root directory of your project repository.
5. Copy all the downloaded raw data files into the `data` folder.
3. Ensure the `dir_path` variable in each Python script (`LTCAMAnalysis.py`, `ButterflyTrack.py`, and `RotatingTrack.py`) is set to the path of the `data` folder. This may look like `dir_path = "./data"` depending on your directory structure.

By following these steps, your data will be correctly positioned for analysis, and the scripts should run without issues related to data file locations.

## Description of Python Files

In this project, multiple Python files are utilized to process raw data, track movements, and generate plots and videos. Below is an overview of what each script does:

### `LTCAMAnalysis.py`

- **Functionality**: This script initiates the data processing workflow by creating a "processed" folder, within which a subfolder named "LTCAM" is generated. Inside the "LTCAM" folder, it further creates subfolders named "Measurements", "Plots", and "Videos".
- **Purpose**: It analyzes raw data to make measurements and produce an infrared video, organizing the output into the respective subfolders.
- **Configuration**: `dir_path` Update this variable to point to the directory where the data files folder is located. `c` - This integer value should be set between 0 and 13, indicating the specific file to be processed for LTCAM analysis.

### `Plots 1.py` and `Plots 2.py`

- **Functionality**: These scripts are designed to generate plots based on the measurement data obtained from running `LTCAMAnalysis.py`.
  - `Plots 1.py` should be run after `LTCAMAnalysis.py` has been executed with `c` values ranging from 0 to 5.
  - `Plots 2.py` requires `LTCAMAnalysis.py` to be run with `c` values from 6 to 13.
- **Purpose**: They store the generated plots in the "Plots" subfolder within the "LTCAM" directory, visually representing the analysis results.
- **Configuration**: `dir_path` Update this variable to point to the directory where the data files folder is located.

### `ButterflyTrack.py`

- **Functionality**: Similar to `LTCAMAnalysis.py`, this script creates a "processed" folder with a specific subfolder named "Butterfly", which then contains "Measurements", "Plots", and "Videos" subfolders.
- **Purpose**: It tracks the movement of a butterfly textile, producing a video that showcases this tracking alongside the relevant measurements and plots.
- **Configuration**: `dir_path` Update this variable to point to the directory where the data files folder is located. `c` - This integer value should be set between 0 and 1, indicating the specific file to be processed for RotatingTrack analysis.

### `RotatingTrack.py`

- **Functionality**: This script processes raw data from videos of a LTCAM on a rotating platform. It creates a "processed" folder with a "Rotating" subfolder, which also includes "Measurements", "Plots", and "Videos" subfolders.
- **Purpose**: It performs circumnutation tracking measurements and generates an infrared video of the LTCAM's movement.
- **Configuration**: `dir_path` Update this variable to point to the directory where the data files folder is located. `c` - This integer value should be set between 0 and 5, indicating the specific file to be processed for RotatingTrack analysis.

### `Plots 3.py`

- **Functionality**: Tailored to work with data from `RotatingTrack.py`, this script needs to be run after `RotatingTrack.py` has processed data with `c` values from 0 to 5.
- **Purpose**: It analyzes the circumnutation movement, specifically focusing on the position of the LTCAM's end, and generates corresponding plots stored in the "Plots" subfolder of the "Rotating" directory.
- **Configuration**: `dir_path` Update this variable to point to the directory where the data files folder is located.

## Running the python scripts

Before running the Python scripts, change the `dir_path` and `c` variables as mentioned in the configuration of each file. Then, the scripts can be run using the following command:

```bash
python PYTHONFILE.py
```

## About the data and experiments

[Information about the dataset](https://github.com/p3d2/LTCAM/wiki/Info-Data)

## License

This project is licensed under the MIT License - see the [LICENSE.md](MIT-LICENSE.txt) file for details.

## Acknowledgments

We acknowledge the computational resources provided by the Aalto Science-IT project.
