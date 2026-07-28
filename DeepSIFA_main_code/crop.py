from PIL import Image
import tifffile
from pathlib import Path

# Open the example TIFF image bundled with this repository.
project_root = Path(__file__).resolve().parent
tif_path = project_root / "data" / "MLKL" / "film1_crop.tif"
img = tifffile.imread(tif_path)

# Gets the image of the specified area (note: the index in Python starts from 0)
x_start, x_end = 100, 140  # x coordinate range
y_start, y_end = 190, 230  # range of y coordinates

# image cropping
cropped_img = img[:, y_start:y_end, x_start:x_end]  # because the tif image is (frames, height, width)

# Save the cropped image next to the input.
output_path = project_root / "data" / "MLKL" / "film1_crop_region.tif"
tifffile.imwrite(output_path, cropped_img)

print(f"Cropped image saved to {output_path}")
