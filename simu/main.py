import os


files = [
    "simu/simu_compare.py",
    "simu/simu_shfactor.py",
    "simu/simu_student.py",
    "simu/simu_data_normal.py",
    "simu/simu_data_student.py",
]


for file in files:
    os.system(f"python {file}")
