# SIM Card Dispenser Project

## Overview
The SIM Card Dispenser project is designed to create a series of OpenSCAD models for tubes that store and dispense various types of SIM cards, including Nano-SIM, Micro-SIM, and Mini-SIM. The project aims to provide a functional and efficient solution for managing SIM card storage and retrieval.

## Purpose
This project allows users to easily store and dispense SIM cards in a controlled manner. It is particularly useful for applications where quick access to SIM cards is necessary, such as in telecommunications or mobile device repair environments.

## Features
- **Modular Design**: Each SIM card type has its own dedicated model, ensuring that the dimensions and mechanisms are tailored to the specific requirements of each card type.
- **Common Functions**: Shared parameters and functions are defined in a common file to promote code reuse and maintainability.
- **Testing Suite**: Includes tests to verify the clearance and fit of SIM cards within the designed tubes, ensuring reliable operation.

## File Structure
- `src/models/`: Contains the OpenSCAD models for each SIM card type.
  - `sim_tube_nano.scad`: Model for Nano-SIM cards.
  - `sim_tube_micro.scad`: Model for Micro-SIM cards.
  - `sim_tube_mini.scad`: Model for Mini-SIM cards.
  - `common.scad`: Common functions and parameters.
  
- `src/tests/`: Contains test scripts for verifying the designs.
  - `clearance_tests.scad`: Tests for clearance and fit of SIM cards.

- `src/types/`: Contains dimension definitions for SIM card types.
  - `dimensions.scad`: Dimensions for Full-Size, Mini, Micro, and Nano-SIMs.

- `docs/`: Contains design documentation.
  - `design_notes.md`: Design notes and considerations.

- `.gitignore`: Specifies files and directories to be ignored by version control.

## Usage
To use the models, open the respective `.scad` files in OpenSCAD. You can modify parameters as needed to customize the designs for your specific requirements.

## Installation
Ensure you have OpenSCAD installed on your system. Clone the repository and navigate to the `src/models` directory to access the models.

## Contribution
Contributions to improve the design or functionality of the SIM Card Dispenser project are welcome. Please submit a pull request or open an issue for discussion.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.