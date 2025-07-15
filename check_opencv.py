# import sys

# # Report Python version
# py_ver = sys.version_info
# print(f"Python version: {py_ver.major}.{py_ver.minor}.{py_ver.micro} "
#       f"({sys.executable})")

# # Try to import cv2 and print its version
# try:
#     import cv2
#     print(f"OpenCV (cv2) version: {cv2.__version__}")
# except ModuleNotFoundError:
#     print("OpenCV (cv2) is NOT installed in this environment.")


import sys

print("Interpreter:", sys.executable)
print("Version:    ", sys.version)
try:
    import cv2
    print("cv2 found, version:", cv2.__version__)
except ModuleNotFoundError:
    print("cv2 NOT found in this environment.")
