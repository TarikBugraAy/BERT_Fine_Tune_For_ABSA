# import os
# import subprocess

# def get_gpu_info():
#     try:
#         # Check if NVIDIA GPU is available
#         result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
#                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

#         if result.returncode != 0:
#             print("No NVIDIA GPU detected or 'nvidia-smi' is not installed.")
#             return None

#         gpu_info = result.stdout.strip().split('\n')
#         return gpu_info
#     except Exception as e:
#         print(f"Error checking GPU: {e}")
#         return None

# def check_cuda_version():
#     try:
#         # Check CUDA version
#         result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

#         if result.returncode != 0:
#             print("CUDA Toolkit is not installed or 'nvcc' is not in PATH.")
#             return None

#         for line in result.stdout.split('\n'):
#             if "release" in line:
#                 return line.strip()

#         return "CUDA version not found in output."
#     except Exception as e:
#         print(f"Error checking CUDA version: {e}")
#         return None

# # Run GPU info and CUDA version checks
# gpu_details = get_gpu_info()
# cuda_version = check_cuda_version()

# # Display the results
# print("=== GPU Information ===")
# if gpu_details:
#     for gpu in gpu_details:
#         name, memory, driver = gpu.split(',')
#         print(f"GPU Name: {name.strip()}")
#         print(f"Total Memory: {memory.strip()}")
#         print(f"Driver Version: {driver.strip()}\n")
# else:
#     print("No GPU information available.")

# print("=== CUDA Toolkit Information ===")
# if cuda_version:
#     print(f"CUDA Version: {cuda_version}")
# else:
#     print("CUDA Toolkit is not installed.")

import torch
print("CUDA Available: ", torch.cuda.is_available())
print("GPU Name: ", torch.cuda.get_device_name(0))
