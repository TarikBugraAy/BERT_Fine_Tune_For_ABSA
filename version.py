import os
import sys
import platform
import torch
import pandas as pd
import numpy as np
from sklearn import __version__ as sklearn_version
from transformers import __version__ as transformers_version

# Get Python version
python_version = platform.python_version()

libraries = {
    "python": python_version,
    "torch": torch.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "transformers": transformers_version,
    "sklearn": sklearn_version
}

print("Installed Library Versions:")
for lib, version in libraries.items():
    print(f"{lib}: {version}")

# print("\nDetails for Installed Libraries:")
# for lib in libraries.keys():
#     if lib == "python":
#         print(f"Python Version: {python_version}")
#     else:
#         print(f"Details for {lib}:")
#         os.system(f"pip show {lib}")
#         print("-" * 40)
