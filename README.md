# README.md for the PAPC Transformer

## Project Overview

This project focuses on implementing and validating a Pilot Contamination Aware Transformer Neural Network for downlink power control in Cell-Free Massive MIMO (CFmMIMO) systems. The project provides a learning-based approach to handle pilot contamination and optimize spectral efficiency with significant computational efficiency improvements over traditional methods.

Reference Paper: [Pilot Contamination Aware Transformer Neural Network for Downlink Power Control in Cell-Free Massive MIMO Networks](https://arxiv.org/abs/2411.19020).

## Repository Layout

- **Main Script:**
  - `cellFreeMassMimoPowCtrl.py`: Entry point for running simulations, data generation, training, and testing.
- **Domain-Specific Modules:**
  - `parameters/`: Holds system and simulation parameters.
  - `powerControl/`: Contains learning and testing logic for power control models.
  - `generateBetaAndPilots.py`: Generates necessary simulation data.
- **Utilities:**
  - `utils/`: Contains helper scripts and auxiliary utilities for argument handling and general functionality.
- **Supporting Files:**
  - `environment.yml`: Python environment dependencies.
  - `consolidatedResults/` and `updatedResults/`: Directories for simulation outputs.
  - `simIdX.sh`: Author-specific scripts for particular simulation setups.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Set up the Python environment using `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate <env-name>
   ```
3. Verify installation:
   ```bash
   python --version
   ```

## How to Run

To run the main script, use the following command:

```bash
python cellFreeMassMimoPowCtrl.py [options]
```

### Example

```bash
python cellFreeMassMimoPowCtrl.py -id 0 -s 500 -m 1 -sc 0
```

### Modes of Operation

- `TRAINING`: Generates training data and trains the model.
- `TESTING`: Tests the model and generates results.
- `PLOTTING_ONLY`: Visualizes already computed results.
- `ALL`: Combines training and testing.
- `CONSOL`: Generates consolidated plots. Author-specific scripts for particular simulation setups.
- `LOCAL`: Processes and annotates consolidated plots. Author-specific scripts for particular simulation setups.

## Detailed Usage

### Command-Line Arguments

- `-id` or `--simulationId`: Unique identifier for simulations (default: 0).
- `-s` or `--samples`: Number of samples for training.
- `-m` or `--mode`: Operating mode (1-6 for different phases).
- `-sc` or `--scenario`: Scenario number (0-3).
- `-c` or `--clean`: Clears all logs and results (optional).

Run the script with the '-h' or '--help' option to view the complete argument list.

## Theory and Validation

This project implements the methods described in the referenced paper. Key highlights:

- **Transformer-based Model**: Handles pilot contamination using masking techniques and attention mechanisms.
- **Unsupervised Learning**: Trains models without requiring labeled data.
- **Scalability**: Adapts to large CFmMIMO networks.

Theoretical insights and numerical results can be found in the reference paper.

## Results

Simulation outputs are saved in:

- **`simIdX/`**: Here X is the simulation ID. This folder contains all the data and plors related to a simulation.

## Contributions

Authors and contributors:
- Atchutaram K. Kocharlakota, Sergiy A. Vorobyov, Robert W. Heath Jr.

## Future Work and Contact
- Explore scalability for even larger CFmMIMO networks.
- Address additional real-world scenarios.

For questions or contributions, contact:
- atchut.ram434@gmail.com
