
# Shared Factor Structure for Matrix Completion

Official implementation of the paper:  
**"Leveraging Shared Factor Structures for Enhanced Matrix Completion with Nonconvex Penalty Regularization"**  
*[arXiv:2504.04020](https://arxiv.org/abs/2504.04020) | [PDF](https://arxiv.org/pdf/2504.04020.pdf)*

---
## Repository Structure
.
├── methods/ # Implementation of our two-step method and baselines
│
├── simulation/ # Code for synthetic data experiments
│
├── empirical/ # Real-world data analysis

---
## Basic Usage
The two-step estimator is implemented in `methods/shfactor.py` through the `matest` class. 

---

## Citation

If you use this code in your research, please cite our paper:

``````bibtex
@misc{a2025leveraging,
  title={Leveraging Shared Factor Structures for Enhanced Matrix Completion with Nonconvex Penalty Regularization}, 
  author={Yuanhong A and Xinyan Fan and Bingyi Jing and Bo Zhang},
  year={2025},
  eprint={2504.04020},
  archivePrefix={arXiv},
  primaryClass={stat.ME},
  url={https://arxiv.org/abs/2504.04020}
}
``````
