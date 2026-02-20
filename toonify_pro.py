import cv2
import os
import numpy as np

# =================================================================
# --- PARAMETER CONTROL PANEL ---
# =================================================================
CONTRAST = 1.2      
BRIGHTNESS = 15     
SATURATION = 1.5    
SHADOW_TINT = [45, 15, 15]  
TINT_THRESHOLD = 60         

# --- HIGHLIGHT PROTECTION (FIXES THE STATIC/BLACK ISSUE) ---
HIGHLIGHT_PROTECTION = 160  # (0-255) Pixels brighter than this won't get inked or textured.
# -----------------------------------------------------------

LINE_THICKNESS = 5    
LINE_SENSITIVITY = 2  # Increased to ignore bright edges
LINE_BOLDNESS = 1      

K_COLORS = 12          # High K-value to ensure white stays white
COLOR_SMOOTHING = 15   
SHARPEN_STRENGTH = 0.3 # Reduced to prevent "static" look on edges

PAPER_TEXTURE_INTENSITY = 0.01 
TRANSPARENCY_THRESHOLD = 220 
# =================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "images_input")
OUTPUT_DIR = os.path.join(BASE_DIR, "toon_pngs")

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def apply_kmeans(img, k):
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return np.uint8(center)[label.flatten()].reshape((img.shape))

def toonify_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # 1. PRE-PROCESS
    img = cv2.convertScaleAbs(img, alpha=CONTRAST, beta=BRIGHTNESS)
    gray_val = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create the Highlight Shield Mask
    highlight_shield = gray_val > HIGHLIGHT_PROTECTION

    # 2. TINTING (Only on non-highlights)
    mask = (gray_val < TINT_THRESHOLD)
    tint_layer = np.full(img.shape, SHADOW_TINT, dtype=np.uint8)
    img[mask] = cv2.addWeighted(img, 0.85, tint_layer, 0.15, 0)[mask]

    # 3. COLOR SMOOTHING & K-MEANS
    color = cv2.bilateralFilter(img, COLOR_SMOOTHING, 250, 250)
    color = apply_kmeans(color, K_COLORS)

    # 4. LINE DETECTION
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray_val, 5), 255, 
                                 cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 
                                 LINE_THICKNESS, LINE_SENSITIVITY)
    
    # FORCE HIGHLIGHTS TO BE WHITE IN THE EDGE MASK (Removes black spots)
    edges[highlight_shield] = 255 

    if LINE_BOLDNESS > 0:
        edges = cv2.erode(edges, np.ones((LINE_BOLDNESS + 1, LINE_BOLDNESS + 1), np.uint8), iterations=1)

    # 5. NOISE (Only on non-highlights to prevent static)
    noise = np.random.normal(0, 255 * PAPER_TEXTURE_INTENSITY, color.shape).astype('int16')
    noised_color = np.clip(color.astype('int16') + noise, 0, 255).astype('uint8')
    color[~highlight_shield] = noised_color[~highlight_shield]
    
    # 6. FINAL MERGE
    toon = cv2.bitwise_and(color, color, mask=edges)
    tmp = cv2.cvtColor(toon, cv2.COLOR_BGR2BGRA)
    
    # Final Transparency
    bg_mask = cv2.inRange(tmp, np.array([TRANSPARENCY_THRESHOLD, TRANSPARENCY_THRESHOLD, TRANSPARENCY_THRESHOLD, 0]), 
                               np.array([255, 255, 255, 255]))
    tmp[bg_mask > 0] = [0, 0, 0, 0] 

    return tmp

def process_folder():
    if not os.path.exists(INPUT_DIR): return
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            result = toonify_image(os.path.join(INPUT_DIR, filename))
            if result is not None:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_fixed.png"), result)
                print(f"✅ Shielded Render: {filename}")

if __name__ == "__main__":
    process_folder()