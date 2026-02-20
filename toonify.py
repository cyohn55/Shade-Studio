import cv2
import os
import numpy as np

# --- IMAGE PRE-PROCESSING ---
CONTRAST = 1.3      
SATURATION = 1.4    
SHADOW_TINT = [45, 15, 15]  # [B, G, R] Blue-ish shadow tint

# --- RIM LIGHT & SPECIAL EFFECTS ---
RIM_LIGHT_STRENGTH = 1.5    # 0 to 2.0. Adds a bright glow to the silhouette edges.
PAPER_TEXTURE_INTENSITY = 0.05 # 0 to 0.1. Adds a subtle "grain" to colors.

# --- LINE CONTROL ---
LINE_THICKNESS = 9    
LINE_SENSITIVITY = 7   
LINE_BOLDNESS = 1      
DETAIL_CLEANING = 3    

# --- COLOR CONTROL ---
K_COLORS = 6           
COLOR_SMOOTHING = 20   
SHARPEN_STRENGTH = 0.6 
# -----------------------------

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
    
    # 1. PRE-PROCESS & TINTING
    img = cv2.convertScaleAbs(img, alpha=CONTRAST, beta=0)
    low_light_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 110
    img[low_light_mask] = cv2.addWeighted(img[low_light_mask], 0.85, np.array(SHADOW_TINT, dtype=np.uint8), 0.15, 0)

    # 2. SATURATION & K-MEANS COLORS
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] *= SATURATION
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype("uint8"), cv2.COLOR_HSV2BGR)
    color = cv2.bilateralFilter(img, COLOR_SMOOTHING, 250, 250)
    color = apply_kmeans(color, K_COLORS)

    # 3. RIM LIGHT SIMULATION
    # Detect the silhouette by finding the non-white background
    gray_for_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_for_mask, 230, 255, cv2.THRESH_BINARY_INV)
    # Find the inner edge
    kernel_rim = np.ones((5, 5), np.uint8)
    inner_edge = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel_rim)
    # Apply the bright rim
    color[inner_edge > 0] = cv2.addWeighted(color[inner_edge > 0], 0.5, np.array([255, 255, 255], dtype=np.uint8), 0.5 * RIM_LIGHT_STRENGTH, 0)

    # 4. LINE DETECTION & CLEANING
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray_for_mask, 5), 255, 
                                 cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 
                                 LINE_THICKNESS, LINE_SENSITIVITY)
    if DETAIL_CLEANING > 0:
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((DETAIL_CLEANING, DETAIL_CLEANING), np.uint8))
    if LINE_BOLDNESS > 0:
        edges = cv2.erode(edges, np.ones((LINE_BOLDNESS + 1, LINE_BOLDNESS + 1), np.uint8), iterations=1)

    # 5. PAPER TEXTURE / NOISE
    noise = np.random.normal(0, 255 * PAPER_TEXTURE_INTENSITY, color.shape).astype('uint8')
    color = cv2.add(color, noise)

    # 6. FINAL MERGE
    toon = cv2.bitwise_and(color, color, mask=edges)
    tmp = cv2.cvtColor(toon, cv2.COLOR_BGR2BGRA)
    
    # Final Transparency
    bg_mask = cv2.inRange(tmp, np.array([215, 215, 215, 0]), np.array([255, 255, 255, 255]))
    tmp[bg_mask > 0] = [0, 0, 0, 0] 

    return tmp

def process_folder():
    if not os.path.exists(INPUT_DIR): return
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            result = toonify_image(os.path.join(INPUT_DIR, filename))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_pro_toon.png"), result)
            print(f"🎬 Render Finished: {filename}")

if __name__ == "__main__":
    process_folder()