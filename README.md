
# Lattice Boltzmann Simulation in Python

This repository contains a lattice Boltzmann simulation developed in Python, inspired by the Medium article ["Create Your Own Lattice Boltzmann Simulation with Python"](https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c).

## Overview

The lattice Boltzmann method (LBM) is a computational fluid dynamics technique that simulates fluid flows. This project provides a basic implementation of LBM, allowing users to explore and simulate 2D fluid flows in Python.

## Features

- 2D fluid flow simulation
- Adjustable parameters for fluid density and velocity
- Visualization of fluid flow dynamics over time

## Requirements

The project uses Python and the following libraries:

- `numpy`: For numerical computations
- `matplotlib`: For visualizing simulation results

These requirements are listed in the `requirements.txt` file.

## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the simulation:

    ```bash
    python simulation.py
    ```

2. Adjust parameters in `config.py` to experiment with different fluid flow scenarios, such as modifying the fluid density or velocity.

## Project Structure

- `simulation.py`: Main script to run the simulation
- `config.py`: Configuration file to adjust simulation parameters
- `visualization.py`: Contains functions for visualizing the results of the simulation

## Example Output

You can visualize the flow dynamics generated by the simulation. Below is an example of what you might expect:

## License

This project is licensed under the MIT License. See the LICENSE file for details.
