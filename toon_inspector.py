import cv2
import os
import numpy as np

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "images_input")

def nothing(x):
    pass

def apply_kmeans(img, k):
    if k < 1: k = 1
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    return np.uint8(center)[label.flatten()].reshape((img.shape))

def run_inspector():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("No images found!")
        return
    
    # Load the image
    original_img = cv2.imread(os.path.join(INPUT_DIR, files[0]))
    
    cv2.namedWindow('Toon Inspector', cv2.WINDOW_NORMAL)
    
    # NEW: Sliders to control lighting stability
    cv2.createTrackbar('Contrast', 'Toon Inspector', 13, 30, nothing)
    cv2.createTrackbar('Brightness', 'Toon Inspector', 15, 100, nothing)
    cv2.createTrackbar('Stabilize', 'Toon Inspector', 0, 1, nothing) # Toggle Normalization
    cv2.createTrackbar('K-Colors', 'Toon Inspector', 6, 16, nothing)
    cv2.createTrackbar('Line Thick', 'Toon Inspector', 9, 21, nothing)
    cv2.createTrackbar('Line Sens', 'Toon Inspector', 8, 20, nothing)
    cv2.createTrackbar('Highlight Prot', 'Toon Inspector', 230, 255, nothing)

    while True:
        con = cv2.getTrackbarPos('Contrast', 'Toon Inspector') / 10.0
        brt = cv2.getTrackbarPos('Brightness', 'Toon Inspector')
        stab = cv2.getTrackbarPos('Stabilize', 'Toon Inspector')
        k_val = cv2.getTrackbarPos('K-Colors', 'Toon Inspector')
        thick = cv2.getTrackbarPos('Line Thick', 'Toon Inspector')
        sens = cv2.getTrackbarPos('Line Sens', 'Toon Inspector')
        prot = cv2.getTrackbarPos('Highlight Prot', 'Toon Inspector')

        # --- LIGHTING STABILIZATION STEP ---
        working_img = original_img.copy()
        if stab == 1:
            # Normalizes the histogram to prevent "lighting jumps"
            # It maps the darkest pixel to 0 and brightest to 255
            lab = cv2.cvtColor(working_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            working_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 1. Pre-process
        img = cv2.convertScaleAbs(working_img, alpha=con, beta=brt)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shield = gray > prot

        # 2. Color pass
        color = cv2.bilateralFilter(img, 15, 250, 250)
        if k_val > 0:
            color = apply_kmeans(color, k_val)

        # 3. Line pass
        if thick % 2 == 0: thick += 1
        if thick < 3: thick = 3
        edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255, 
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 
                                     thick, sens)
        edges[shield] = 255 

        result = cv2.bitwise_and(color, color, mask=edges)
        cv2.imshow('Toon Inspector', result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s'):
            print(f"STABILIZE = {stab}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inspector()