# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("cv2", "skimage", "pandas", "numpy", "scipy")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError(f"Missing required dependencies {missing_dependencies}")

del hard_dependencies, dependency, missing_dependencies
