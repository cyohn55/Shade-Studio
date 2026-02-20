import os
from PIL import Image

# This finds the folder where your pixelate.py is actually located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "images_input") 
OUTPUT_DIR = os.path.join(BASE_DIR, "pixelated_pngs")

# Adjust this number to increase/decrease pixel density.
PIXEL_SIZE = 11 

def process_images():
    # Check if the input folder exists; if not, create it and tell the user
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created folder: {INPUT_DIR}")
        print("Please drop your images into that folder and run the script again.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGBA")
            
            # Transparency: Masks out the gray checkerboard/white background
            datas = img.getdata()
            new_data = []
            for item in datas:
                # Target common background colors (white to light gray)
                if item[0] > 200 and item[1] > 200 and item[2] > 200:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
            img.putdata(new_data)

            # Pixelate via Scaling (Nearest Neighbor is key)
            small_size = (max(1, img.width // PIXEL_SIZE), max(1, img.height // PIXEL_SIZE))
            img_small = img.resize(small_size, resample=Image.NEAREST)
            result = img_small.resize(img.size, resample=Image.NEAREST)

            base_name = os.path.splitext(filename)[0]
            result.save(os.path.join(OUTPUT_DIR, f"{base_name}_pixel.png"), "PNG")
            print(f"✅ Success: {filename}")

if __name__ == "__main__":
    process_images()