import os


files = [
    "empi/empi_all.py",
    "empi/mse.py",
    "empi/rank.py",
]


for file in files:
    os.system(f"python {file}")
