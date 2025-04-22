# Code Directory Documentation

## Overview
This repository is organized into two primary components for simulation studies:

## 1. Simulation Data Generation

### Core Scripts:
- **`simu_compare.py`**  
  Generates comparative simulation data for:
  - MHT (Method A)
  - NW (Method B) 
  - MWC (Method C)
  with known rank conditions

- **`simu_shfactor.py`**  
  Produces simulation data for the two-step shared factor method using:
  - Normally distributed residuals

- **`simu_student.py`**  
  Extends the shared factor method to handle:
  - Student's t-distributed residuals (df=5 and df=9)

## 2. Simulation Data Analysis

### Analysis Scripts:
- **`simu_data_normal.py`**  
  Comparative performance analysis:
  - Our method vs. baseline methods
  - Normal residual conditions

- **`simu_data_student.py`**  
  MSE evaluation under different residual distributions:
  - Normal
  - Student's t (t5 and t9)

## Data Storage
All simulation outputs are stored in directories like "data_*/"
