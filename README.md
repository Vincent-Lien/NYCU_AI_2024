# NYCU_AI_2024
National Yang Ming Chiao Tung University Graduate Course, Artificial Intelligence.  
國立陽明交通大學 2024 Fall Semester 人工智慧 楊元福教授

## Assignment Description

This repository contains three assignment folders:
- HW1
- HW2
- HW3

## Installation Steps

1. Clone this repository to your local machine:
    ```sh
    git clone https://github.com/Vincent-Lien/NYCU_AI_2024.git
    cd NYCU_AI_2024
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirement.txt
    ```

- This repository is based on Python 3.10.

## Running Assignments

### HW1
HW1 is a Jupyter Notebook. Please open and run [HW1_code.ipynb](HW1/HW1_code.ipynb) using Jupyter Notebook or Jupyter Lab.

### HW2
HW2 is also a Jupyter Notebook. Please open and run [HW2_code.ipynb](HW2/HW2_code.ipynb) using Jupyter Notebook or Jupyter Lab.

### HW3
HW3 contains a Python script and some related data. Please follow these steps to run it:

1. Navigate to the HW3 folder:
    ```sh
    cd HW3
    ```

2. Run the [main.py](http://_vscodecontentref_/3) script:
    ```sh
    python main.py [--model {Net,CNN}] [--batch-size N] [--lr LR] [--epochs N]
    ```

    Additional parameters:
    - `--model`: Choose model architecture between Net or CNN (default: Net)
    - `--batch-size`: Set batch size for training (default: 64)
    - `--lr`: Set learning rate (default: 1.0)
    - `--epochs`: Set number of epochs to train (default: 14)

3. Or run all models using the provided shell script:
    ```sh
    ./run_all_models.sh
    ```

    This script will execute all models sequentially and save the results.

    Example configurations:
    ```
    Models: Net, CNN
    Batch sizes: 16, 64, 256
    Learning rates: 0.1, 0.01, 0.001
    Epochs: 10, 100, 500
    ```

    The script will evaluate model performance across these parameter combinations.


    ## Resources and References

    - For a comprehensive Python and NumPy tutorial, visit [CS231n Python NumPy Tutorial](https://cs231n.github.io/python-numpy-tutorial/)