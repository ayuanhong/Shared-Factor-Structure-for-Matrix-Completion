# Algorithm Implementation Directory

## Method Implementations

### Our Proposed Method
- **`shfactor.py`**  
  Implements our novel two-step estimator featuring:
  - Step 1: Use matrix MCP regularized estimator to estimate ranks
  - Step 2: Use the Oracle estimator with known ranks

### Baseline Methods

#### Simulation Baselines
- **`compare_simu.py`**  
  Baseline methods for simulation studies, see simu directory.
  - Assumes known rank conditions
  - Includes:
    - MHT 
    - NW 
    - MWC

#### Empirical Data Baselines  
- **`compare_empi.py`**  
  Provides baseline methods for real-world data analysis, see empi directory.
  - Uses nuclear norm penalty for matrix estimation
