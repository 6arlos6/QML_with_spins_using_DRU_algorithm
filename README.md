# Quantum Machine Learning with Spins using DRU algorithm

## Overview

This project explores a Quantum Machine Learning (QML) model for classification tasks, referred to as the "DRU model" (`Modelo_DRU`). The research investigates the performance of this model on various datasets, such as the classic IRIS dataset and the `make_moons` dataset. The experiments focus on benchmarking different cost functions, analyzing the impact of entanglement, and evaluating the model's performance across different numbers of qubits.

The core of the project is built using Python, with QML capabilities provided by the `PennyLane` and `QuTiP` libraries.

## Project Structure

The repository is organized into several directories, each serving a specific purpose:

-   **/ (root):** Contains the main Jupyter notebooks for conducting experiments, along with configuration and result files.
-   **`dll/`:** This directory holds the core Python modules that define the QML models, cost functions, and utility functions.
-   **`results_of_test/`:** Stores the outputs of the experiments, including generated plots, trained model parameters, and performance metrics in Excel files.
-   **`esquemas_propios/`:** Contains custom diagrams and schematics related to the project's concepts.
-   **`out_ranking/`:** Appears to store results related to model ranking or benchmarking.

### Key Files

-   **`test_*.ipynb`:** A series of Jupyter notebooks, each representing a specific experiment. For example, `test_1_base_line_dru.ipynb` establishes a baseline for the DRU model.
-   **`requirements.txt`:** Lists all the Python dependencies required to run the project.
-   **`dll/main_fun.py`:** A utility module with helper functions for data generation, visualization (e.g., Bloch sphere), and ML evaluation.
-   **`dll/baseline_dru_v2.py`:** Likely contains the implementation of the `Modelo_DRU`.
-   **`dll/models.py`:** Defines the various quantum circuit architectures used in the experiments (e.g., `qcircuit_1_qubit_mixed`, `qcircuit_2_qubit_mixed`).
-   **`dll/cost_functions.py`:** Contains implementations of different cost functions like `fidelity_cost`, `Renyi_Divergence`, and `Von_Neumman_Divergence`.

## Getting Started

### Dependencies

To set up the environment, you need to install the Python libraries listed in `requirements.txt`. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Running the Experiments

The experiments are conducted through the Jupyter notebooks in the root directory. To run an experiment:

1.  Start a Jupyter Notebook server in the project's root directory.
2.  Open one of the `test_*.ipynb` notebooks (e.g., `test_1_base_line_dru.ipynb`).
3.  Run the cells in the notebook sequentially to reproduce the experiment.

The notebooks are self-contained and will train the models, generate plots, and save the results to the `results_of_test/` directory.

## Core Concepts

-   **DRU Model:** A custom QML classification model that is the focus of this research.
-   **Quantum Circuits:** The classification is performed using parameterized quantum circuits defined in `dll/models.py`.
-   **Entanglement:** The project explicitly investigates the role of entanglement in the models' performance by including or excluding entangling gates (like CZ) in the circuits.
-   **Cost Functions:** The training process is benchmarked against several cost functions to find the most effective one for the given tasks.
-   **Noise Simulation:** The codebase includes functionality to simulate noise, allowing for the study of the model's robustness under non-ideal conditions.

## Results

The results of the experiments, including performance metrics, generated figures, and model parameters, are automatically saved in the `results_of_test/` directory, organized into subdirectories corresponding to each test.