import os
import glob
import math
import base64
from PIL import Image

def create_combined_plot_svg(directory_path, output_filename="combined_plots.svg"):
    """
    Finds plot images in a directory, combines them into a single grid (3 columns, N rows),
    and exports the result as an SVG file containing the embedded raster image.

    Args:
        directory_path (str): The path to the directory containing the plot images.
        output_filename (str): The name for the final SVG output file.
    """

    # --- Configuration ---
    # Define the standardized size for each subplot
    SUBPLOT_WIDTH = 600
    SUBPLOT_HEIGHT = 350
    
    # The required number of columns
    COLUMNS = 3
    
    # Image file extensions to search for
    IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg')
    
    print(f"--- Plot Combiner Starting ---")
    print(f"Target Directory: {directory_path}")
    print(f"Subplot size: {SUBPLOT_WIDTH}x{SUBPLOT_HEIGHT} px")

    # 1. Find all plot files using glob
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(directory_path, ext)))

    if not image_paths:
        print(f"Error: No images found in '{directory_path}' with extensions {IMAGE_EXTENSIONS}.")
        return

    print(f"Found {len(image_paths)} plot files.")

    # 2. Load and standardize images
    images = []
    for path in image_paths:
        try:
            # Open image and convert to RGBA (for consistent background handling)
            img = Image.open(path).convert("RGBA") 
            # Resize image to the standardized subplot dimensions
            img_resized = img.resize((SUBPLOT_WIDTH, SUBPLOT_HEIGHT))
            images.append(img_resized)
        except Exception as e:
            print(f"Skipping file '{path}' due to error: {e}")

    if not images:
        print("Error: Could not load any valid images.")
        return

    # 3. Calculate grid dimensions
    total_plots = len(images)
    rows = math.ceil(total_plots / COLUMNS)

    TOTAL_WIDTH = COLUMNS * SUBPLOT_WIDTH
    TOTAL_HEIGHT = rows * SUBPLOT_HEIGHT

    print(f"Grid calculated: {COLUMNS} columns x {rows} rows. Final size: {TOTAL_WIDTH}x{TOTAL_HEIGHT} px.")

    # 4. Create the final canvas (white background)
    combined_image = Image.new('RGBA', (TOTAL_WIDTH, TOTAL_HEIGHT), (255, 255, 255, 255)) 

    # 5. Paste images onto the canvas
    for index, img in enumerate(images):
        row = index // COLUMNS
        col = index % COLUMNS

        x_offset = col * SUBPLOT_WIDTH
        y_offset = row * SUBPLOT_HEIGHT

        # Paste the resized image at the calculated offset
        combined_image.paste(img, (x_offset, y_offset))
        
    # --- Prepare for SVG Export (Embedding Raster into Vector) ---
    
    # 6. Save the combined image temporarily as PNG (high quality, lossless)
    temp_png_path = "temp_combined_plots_raster.png"
    combined_image.save(temp_png_path, "PNG")

    # 7. Embed the temporary PNG into an SVG file structure
    
    # Read the temporary PNG content as binary data
    with open(temp_png_path, "rb") as f:
        png_data = f.read()
    
    # Encode the binary data to Base64 for embedding as a Data URI
    base64_data = base64.b64encode(png_data).decode('utf-8')
    data_uri = f"data:image/png;base64,{base64_data}"

    # Create the SVG content string with the embedded image
    svg_content = f'''<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{TOTAL_WIDTH}" height="{TOTAL_HEIGHT}" viewBox="0 0 {TOTAL_WIDTH} {TOTAL_HEIGHT}"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
  <!-- 
    The combined raster image is embedded here. 
    If the original plots were PNG/JPG, the final output will scale the embedded 
    raster image, not the vector data.
  -->
  <image width="{TOTAL_WIDTH}" height="{TOTAL_HEIGHT}" href="{data_uri}"/>
</svg>'''

    # 8. Save the final SVG file
    with open(output_filename, "w") as f:
        f.write(svg_content)
    
    # 9. Clean up the temporary file
    os.remove(temp_png_path)
    
    print(f"\nSuccessfully created embedded SVG file: {output_filename}")
    print(f"The temporary PNG file ({temp_png_path}) has been cleaned up.")


create_combined_plot_svg("/home/patwuch/projects/gee_dengue_ml/reports/figures/20251010/wavelets/maluku")

print("\n(Joined.)")
