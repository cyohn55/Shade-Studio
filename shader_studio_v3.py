"""
Shader Studio v2 - Professional shader tool with advanced features.
Features: 2D/3D rendering, custom presets, AI upscaling, video processing,
          node-based shader editor, LUT support, and more.
"""

import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui, QtOpenGL
from PyQt6.QtWidgets import QSplitter
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader, QOpenGLTexture
from OpenGL import GL
from PIL import Image, ImageFilter
import os
import json
import math
from pathlib import Path
import struct
import random
import tempfile
import subprocess
import copy

# Try to import video processing
VIDEO_AVAILABLE = False
try:
    import cv2
    VIDEO_AVAILABLE = True
    print("OpenCV available for video processing")
except ImportError:
    print("OpenCV not available - video processing disabled")

# Try to import GIF support
GIF_AVAILABLE = False
try:
    import imageio
    GIF_AVAILABLE = True
    print("imageio available for GIF export")
except ImportError:
    print("imageio not available - GIF export will use PIL fallback")

# Try to import optional AI upscaling dependencies
UPSCALING_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    UPSCALING_AVAILABLE = True
    print("PyTorch available for AI upscaling")
except ImportError:
    print("PyTorch not available - AI upscaling will use Lanczos fallback")

# Try to import optional 3D format loaders
GLTF_AVAILABLE = False
FBX_AVAILABLE = False
try:
    import pygltflib
    GLTF_AVAILABLE = True
    print("pygltflib available for GLTF loading")
except ImportError:
    print("pygltflib not available - GLTF loading disabled")

try:
    import fbx
    FBX_AVAILABLE = True
    print("FBX SDK available")
except ImportError:
    pass  # FBX SDK is rarely available

# --- PRESET STORAGE ---
PRESETS_FILE = Path(__file__).parent / "shader_presets.json"
USER_PRESETS = {}  # Will be loaded from file

def load_user_presets():
    """Load user presets from JSON file."""
    global USER_PRESETS
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, 'r') as f:
                USER_PRESETS = json.load(f)
            print(f"Loaded {len(USER_PRESETS)} user presets")
        except Exception as e:
            print(f"Error loading presets: {e}")
            USER_PRESETS = {}
    return USER_PRESETS

def save_user_presets():
    """Save user presets to JSON file."""
    try:
        with open(PRESETS_FILE, 'w') as f:
            json.dump(USER_PRESETS, f, indent=2)
        print(f"Saved {len(USER_PRESETS)} user presets")
    except Exception as e:
        print(f"Error saving presets: {e}")

# Load presets on startup
load_user_presets()

# --- RECENT FILES ---
SETTINGS_FILE = Path(__file__).parent / "shader_settings.json"
RECENT_FILES = []
MAX_RECENT = 10
CLIPBOARD_PARAMS = None  # For copy/paste parameters

def load_settings():
    """Load application settings including recent files."""
    global RECENT_FILES
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                RECENT_FILES = settings.get("recent_files", [])
                # Filter out non-existent files
                RECENT_FILES = [f for f in RECENT_FILES if os.path.exists(f)]
        except Exception as e:
            print(f"Error loading settings: {e}")

def save_settings():
    """Save application settings."""
    try:
        settings = {"recent_files": RECENT_FILES[:MAX_RECENT]}
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Error saving settings: {e}")

def add_recent_file(path):
    """Add a file to recent files list."""
    global RECENT_FILES
    if path in RECENT_FILES:
        RECENT_FILES.remove(path)
    RECENT_FILES.insert(0, path)
    RECENT_FILES = RECENT_FILES[:MAX_RECENT]
    save_settings()

load_settings()

# --- HOTKEYS ---
HOTKEYS = {
    "undo": "Ctrl+Z",
    "redo": "Ctrl+Y",
    "save": "Ctrl+S",
    "export": "Ctrl+E",
    "open": "Ctrl+O",
    "fullscreen": "F11",
    "randomize": "Ctrl+R",
    "copy_params": "Ctrl+C",
    "paste_params": "Ctrl+V",
    "reset": "Ctrl+D",
}


# --- AI UPSCALING ---
class SimpleUpscaler:
    """Simple upscaling class with Lanczos fallback and optional AI models."""

    def __init__(self):
        self.model = None
        self.device = None
        if UPSCALING_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Upscaler using device: {self.device}")

    def upscale_lanczos(self, image, scale=2):
        """Upscale using Lanczos interpolation (fallback)."""
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        new_size = (img.width * scale, img.height * scale)
        upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
        return np.array(upscaled)

    def upscale_bilinear(self, image, scale=2):
        """Upscale using bilinear interpolation."""
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        new_size = (img.width * scale, img.height * scale)
        upscaled = img.resize(new_size, Image.Resampling.BILINEAR)
        return np.array(upscaled)

    def upscale_bicubic(self, image, scale=2):
        """Upscale using bicubic interpolation."""
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        new_size = (img.width * scale, img.height * scale)
        upscaled = img.resize(new_size, Image.Resampling.BICUBIC)
        return np.array(upscaled)

    def upscale(self, image, scale=2, method="lanczos"):
        """Upscale image using specified method."""
        if method == "lanczos":
            return self.upscale_lanczos(image, scale)
        elif method == "bilinear":
            return self.upscale_bilinear(image, scale)
        elif method == "bicubic":
            return self.upscale_bicubic(image, scale)
        elif method == "ai" and UPSCALING_AVAILABLE:
            # Simple AI-enhanced upscale using sharpening after Lanczos
            upscaled = self.upscale_lanczos(image, scale)
            return self._enhance_sharpness(upscaled)
        else:
            return self.upscale_lanczos(image, scale)

    def _enhance_sharpness(self, image):
        """Apply AI-like sharpening enhancement."""
        if not UPSCALING_AVAILABLE:
            return image

        # Convert to tensor
        img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # Apply unsharp mask enhancement
        img_tensor = img_tensor.to(self.device)

        # Simple sharpening kernel
        kernel = torch.tensor([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        # Apply to each channel
        channels = []
        for c in range(img_tensor.shape[1]):
            channel = img_tensor[:, c:c+1, :, :]
            padded = torch.nn.functional.pad(channel, (1, 1, 1, 1), mode='reflect')
            sharpened = torch.nn.functional.conv2d(padded, kernel)
            channels.append(sharpened)

        result = torch.cat(channels, dim=1)
        result = torch.clamp(result, 0, 1)

        # Convert back to numpy
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (result * 255).astype(np.uint8)


# Global upscaler instance
UPSCALER = SimpleUpscaler()


# --- LUT (Color Lookup Table) SUPPORT ---
def load_cube_lut(filepath):
    """Load a .cube LUT file and return as 3D numpy array."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        size = 0
        data = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[1])
            elif line.startswith('TITLE') or line.startswith('DOMAIN'):
                continue
            else:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                        data.append([r, g, b])
                    except ValueError:
                        continue

        if size == 0 or len(data) != size ** 3:
            print(f"Invalid LUT: size={size}, data points={len(data)}")
            return None

        # Reshape to 3D LUT
        lut = np.array(data, dtype=np.float32).reshape(size, size, size, 3)
        return lut

    except Exception as e:
        print(f"Error loading LUT: {e}")
        return None


def apply_lut(image, lut):
    """Apply a 3D LUT to an image."""
    if lut is None:
        return image

    size = lut.shape[0]
    img = image.astype(np.float32) / 255.0

    # Trilinear interpolation
    coords = img * (size - 1)
    coords_floor = np.floor(coords).astype(int)
    coords_ceil = np.minimum(coords_floor + 1, size - 1)
    coords_frac = coords - coords_floor

    # Get corner values
    r0, g0, b0 = coords_floor[:, :, 0], coords_floor[:, :, 1], coords_floor[:, :, 2]
    r1, g1, b1 = coords_ceil[:, :, 0], coords_ceil[:, :, 1], coords_ceil[:, :, 2]
    fr, fg, fb = coords_frac[:, :, 0:1], coords_frac[:, :, 1:2], coords_frac[:, :, 2:3]

    # Trilinear interpolation
    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    c00 = c000 * (1 - fr) + c100 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c11 = c011 * (1 - fr) + c111 * fr

    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg

    result = c0 * (1 - fb) + c1 * fb
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    # Preserve alpha if present
    if image.shape[2] == 4:
        result = np.dstack([result, image[:, :, 3]])

    return result


# --- TEXTURE OVERLAY ---
def apply_texture_overlay(image, overlay_path, blend_mode="multiply", opacity=0.5):
    """Apply a texture overlay to an image."""
    try:
        overlay = Image.open(overlay_path).convert("RGBA")
        overlay = overlay.resize((image.shape[1], image.shape[0]), Image.Resampling.LANCZOS)
        overlay_arr = np.array(overlay).astype(np.float32) / 255.0

        img = image.astype(np.float32) / 255.0
        if img.shape[2] == 3:
            img = np.dstack([img, np.ones((img.shape[0], img.shape[1]))])

        # Blend modes
        if blend_mode == "multiply":
            blended = img[:, :, :3] * overlay_arr[:, :, :3]
        elif blend_mode == "screen":
            blended = 1 - (1 - img[:, :, :3]) * (1 - overlay_arr[:, :, :3])
        elif blend_mode == "overlay":
            mask = img[:, :, :3] < 0.5
            blended = np.where(mask, 2 * img[:, :, :3] * overlay_arr[:, :, :3],
                              1 - 2 * (1 - img[:, :, :3]) * (1 - overlay_arr[:, :, :3]))
        elif blend_mode == "soft_light":
            blended = (1 - 2 * overlay_arr[:, :, :3]) * img[:, :, :3] ** 2 + 2 * overlay_arr[:, :, :3] * img[:, :, :3]
        elif blend_mode == "add":
            blended = img[:, :, :3] + overlay_arr[:, :, :3]
        else:
            blended = overlay_arr[:, :, :3]

        # Apply opacity
        result = img[:, :, :3] * (1 - opacity) + blended * opacity
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        if image.shape[2] == 4:
            result = np.dstack([result, image[:, :, 3]])

        return result
    except Exception as e:
        print(f"Error applying overlay: {e}")
        return image


# --- GLTF/GLB LOADER ---
def load_gltf(filepath):
    """Load a GLTF/GLB file and return vertices, uvs, normals."""
    if not GLTF_AVAILABLE:
        print("GLTF loading requires pygltflib: pip install pygltflib")
        return None, None, None

    try:
        gltf = pygltflib.GLTF2().load(filepath)

        all_vertices = []
        all_uvs = []
        all_normals = []

        for mesh in gltf.meshes:
            for primitive in mesh.primitives:
                # Get accessor indices
                pos_accessor_idx = primitive.attributes.POSITION
                if pos_accessor_idx is None:
                    continue

                # Read position data
                pos_accessor = gltf.accessors[pos_accessor_idx]
                pos_buffer_view = gltf.bufferViews[pos_accessor.bufferView]
                pos_buffer = gltf.buffers[pos_buffer_view.buffer]

                # Get binary data
                if hasattr(gltf, '_glb_data') and gltf._glb_data:
                    data = gltf._glb_data
                else:
                    buffer_path = Path(filepath).parent / pos_buffer.uri
                    with open(buffer_path, 'rb') as f:
                        data = f.read()

                # Extract positions
                offset = pos_buffer_view.byteOffset + (pos_accessor.byteOffset or 0)
                count = pos_accessor.count
                for i in range(count):
                    idx = offset + i * 12  # 3 floats * 4 bytes
                    x, y, z = struct.unpack('fff', data[idx:idx+12])
                    all_vertices.extend([x, y, z])

                # Read UVs if available
                if hasattr(primitive.attributes, 'TEXCOORD_0') and primitive.attributes.TEXCOORD_0 is not None:
                    uv_accessor = gltf.accessors[primitive.attributes.TEXCOORD_0]
                    uv_buffer_view = gltf.bufferViews[uv_accessor.bufferView]
                    uv_offset = uv_buffer_view.byteOffset + (uv_accessor.byteOffset or 0)
                    for i in range(uv_accessor.count):
                        idx = uv_offset + i * 8  # 2 floats * 4 bytes
                        u, v = struct.unpack('ff', data[idx:idx+8])
                        all_uvs.extend([u, v])
                else:
                    all_uvs.extend([0.0, 0.0] * count)

                # Read normals if available
                if hasattr(primitive.attributes, 'NORMAL') and primitive.attributes.NORMAL is not None:
                    norm_accessor = gltf.accessors[primitive.attributes.NORMAL]
                    norm_buffer_view = gltf.bufferViews[norm_accessor.bufferView]
                    norm_offset = norm_buffer_view.byteOffset + (norm_accessor.byteOffset or 0)
                    for i in range(norm_accessor.count):
                        idx = norm_offset + i * 12
                        nx, ny, nz = struct.unpack('fff', data[idx:idx+12])
                        all_normals.extend([nx, ny, nz])
                else:
                    all_normals.extend([0.0, 1.0, 0.0] * count)

        if not all_vertices:
            return None, None, None

        return (np.array(all_vertices, dtype=np.float32),
                np.array(all_uvs, dtype=np.float32),
                np.array(all_normals, dtype=np.float32))

    except Exception as e:
        print(f"Error loading GLTF: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# --- SHADER LIBRARY ---
SHADERS = {
    "Original": {
        "category": "Basic",
        "description": "Original image with basic adjustments",
        "uniforms": {
            "brightness": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
            "contrast": {"min": 0.0, "max": 3.0, "default": 1.0, "step": 0.01},
            "saturation": {"min": 0.0, "max": 3.0, "default": 1.0, "step": 0.01},
            "gamma": {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.01},
            "exposure": {"min": -3.0, "max": 3.0, "default": 0.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float brightness;
            uniform float contrast;
            uniform float saturation;
            uniform float gamma;
            uniform float exposure;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec4 color = texture(u_texture, v_uv);

                // Exposure
                color.rgb *= pow(2.0, exposure);

                // Brightness
                color.rgb += brightness;

                // Contrast
                color.rgb = (color.rgb - 0.5) * contrast + 0.5;

                // Saturation
                float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                color.rgb = mix(vec3(gray), color.rgb, saturation);

                // Gamma
                color.rgb = pow(max(color.rgb, vec3(0.0)), vec3(1.0 / gamma));

                f_color = vec4(clamp(color.rgb, 0.0, 1.0), color.a);
            }
        """
    },

    "Pixelation": {
        "category": "Stylized",
        "description": "Retro pixel art effect with adjustable resolution",
        "uniforms": {
            "pixel_size": {"min": 1.0, "max": 64.0, "default": 8.0, "step": 1.0},
            "color_depth": {"min": 2.0, "max": 256.0, "default": 32.0, "step": 1.0},
            "dither_amount": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "saturation_boost": {"min": 0.5, "max": 2.0, "default": 1.0, "step": 0.05},
            "outline_strength": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "outline_threshold": {"min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float pixel_size;
            uniform float color_depth;
            uniform float dither_amount;
            uniform float saturation_boost;
            uniform float outline_strength;
            uniform float outline_threshold;
            in vec2 v_uv;
            out vec4 f_color;

            float bayer8(vec2 p) {
                int x = int(mod(p.x, 8.0));
                int y = int(mod(p.y, 8.0));
                int index = x + y * 8;
                float bayer[64] = float[64](
                    0.0, 32.0, 8.0, 40.0, 2.0, 34.0, 10.0, 42.0,
                    48.0, 16.0, 56.0, 24.0, 50.0, 18.0, 58.0, 26.0,
                    12.0, 44.0, 4.0, 36.0, 14.0, 46.0, 6.0, 38.0,
                    60.0, 28.0, 52.0, 20.0, 62.0, 30.0, 54.0, 22.0,
                    3.0, 35.0, 11.0, 43.0, 1.0, 33.0, 9.0, 41.0,
                    51.0, 19.0, 59.0, 27.0, 49.0, 17.0, 57.0, 25.0,
                    15.0, 47.0, 7.0, 39.0, 13.0, 45.0, 5.0, 37.0,
                    63.0, 31.0, 55.0, 23.0, 61.0, 29.0, 53.0, 21.0
                );
                return (bayer[index] / 64.0 - 0.5) * dither_amount;
            }

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel_coord = floor(v_uv * tex_size / pixel_size);
                vec2 pixel_uv = pixel_coord * pixel_size / tex_size;

                vec4 color = texture(u_texture, pixel_uv);

                // Saturation boost
                float gray = luminance(color.rgb);
                color.rgb = mix(vec3(gray), color.rgb, saturation_boost);

                // Dithering
                float dither = bayer8(pixel_coord);
                color.rgb += dither / color_depth;

                // Color quantization
                float levels = color_depth;
                color.rgb = floor(color.rgb * levels + 0.5) / levels;

                // Outline detection
                if(outline_strength > 0.0) {
                    vec2 pixel = pixel_size / tex_size;
                    float edge = 0.0;
                    float center = luminance(color.rgb);
                    for(int y = -1; y <= 1; y++) {
                        for(int x = -1; x <= 1; x++) {
                            if(x == 0 && y == 0) continue;
                            float neighbor = luminance(texture(u_texture, pixel_uv + vec2(x, y) * pixel).rgb);
                            edge += abs(center - neighbor);
                        }
                    }
                    edge = smoothstep(outline_threshold, outline_threshold + 0.1, edge / 8.0);
                    color.rgb = mix(color.rgb, vec3(0.0), edge * outline_strength);
                }

                f_color = color;
            }
        """
    },

    "Grease Pencil": {
        "category": "Sketch",
        "description": "Hand-drawn pencil sketch effect like Blender's Grease Pencil",
        "uniforms": {
            "line_thickness": {"min": 0.5, "max": 8.0, "default": 1.5, "step": 0.1},
            "line_darkness": {"min": 0.0, "max": 3.0, "default": 1.0, "step": 0.05},
            "fill_opacity": {"min": 0.0, "max": 1.0, "default": 0.8, "step": 0.05},
            "edge_threshold": {"min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
            "edge_softness": {"min": 0.01, "max": 0.3, "default": 0.1, "step": 0.01},
            "noise_amount": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.05},
            "paper_texture": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
            "stroke_color_r": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.05},
            "stroke_color_g": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.05},
            "stroke_color_b": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.05},
            "color_variation": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float line_thickness;
            uniform float line_darkness;
            uniform float fill_opacity;
            uniform float edge_threshold;
            uniform float edge_softness;
            uniform float noise_amount;
            uniform float paper_texture;
            uniform float stroke_color_r;
            uniform float stroke_color_g;
            uniform float stroke_color_b;
            uniform float color_variation;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = hash(i);
                float b = hash(i + vec2(1.0, 0.0));
                float c = hash(i + vec2(0.0, 1.0));
                float d = hash(i + vec2(1.0, 1.0));
                return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
            }

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = line_thickness / tex_size;

                vec4 base_color = texture(u_texture, v_uv);

                // Add noise jitter to sampling for hand-drawn feel
                vec2 jitter = vec2(
                    noise(v_uv * tex_size * 0.5) - 0.5,
                    noise(v_uv * tex_size * 0.5 + 100.0) - 0.5
                ) * noise_amount * pixel;

                float samples[9];
                int idx = 0;
                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        vec3 c = texture(u_texture, v_uv + vec2(x, y) * pixel + jitter).rgb;
                        samples[idx++] = luminance(c);
                    }
                }

                float gx = samples[2] + 2.0*samples[5] + samples[8] - samples[0] - 2.0*samples[3] - samples[6];
                float gy = samples[0] + 2.0*samples[1] + samples[2] - samples[6] - 2.0*samples[7] - samples[8];
                float edge = sqrt(gx*gx + gy*gy);

                // Add noise to edge for organic feel
                edge += (noise(v_uv * tex_size) - 0.5) * noise_amount * 0.2;

                float stroke = smoothstep(edge_threshold, edge_threshold + edge_softness, edge);
                stroke *= line_darkness;

                // Fill color with optional color variation
                vec3 fill_color = base_color.rgb;
                if(color_variation > 0.0) {
                    float hue_shift = (noise(v_uv * tex_size * 0.2) - 0.5) * color_variation * 0.1;
                    fill_color = mix(fill_color, fill_color.gbr, hue_shift);
                }

                vec3 fill = fill_color * fill_opacity + vec3(1.0 - fill_opacity);

                // Stroke color
                vec3 stroke_color = vec3(stroke_color_r, stroke_color_g, stroke_color_b);

                vec3 final_color = mix(fill, stroke_color, stroke);

                // Paper texture overlay
                float paper = 0.95 + noise(v_uv * tex_size * 0.3) * paper_texture * 0.1;
                final_color *= paper;

                f_color = vec4(final_color, base_color.a);
            }
        """
    },

    "Toon Shader": {
        "category": "Toon",
        "description": "Classic cel-shaded cartoon look with customizable shading",
        "uniforms": {
            "color_bands": {"min": 2.0, "max": 12.0, "default": 4.0, "step": 1.0},
            "edge_thickness": {"min": 0.5, "max": 8.0, "default": 1.5, "step": 0.1},
            "edge_threshold": {"min": 0.01, "max": 0.5, "default": 0.15, "step": 0.01},
            "edge_softness": {"min": 0.0, "max": 0.2, "default": 0.02, "step": 0.01},
            "saturation_boost": {"min": 0.0, "max": 3.0, "default": 1.2, "step": 0.05},
            "brightness_boost": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
            "shadow_intensity": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
            "highlight_intensity": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.05},
            "outline_color_r": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "outline_color_g": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "outline_color_b": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "quantize_hue": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float color_bands;
            uniform float edge_thickness;
            uniform float edge_threshold;
            uniform float edge_softness;
            uniform float saturation_boost;
            uniform float brightness_boost;
            uniform float shadow_intensity;
            uniform float highlight_intensity;
            uniform float outline_color_r;
            uniform float outline_color_g;
            uniform float outline_color_b;
            uniform float quantize_hue;
            in vec2 v_uv;
            out vec4 f_color;

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = edge_thickness / tex_size;

                vec4 base = texture(u_texture, v_uv);

                vec3 hsv = rgb2hsv(base.rgb);

                // Optionally quantize hue
                if(quantize_hue > 0.5) {
                    hsv.x = floor(hsv.x * 12.0 + 0.5) / 12.0;
                }

                // Saturation boost
                hsv.y = clamp(hsv.y * saturation_boost, 0.0, 1.0);

                // Quantize value (brightness bands)
                float original_v = hsv.z;
                hsv.z = floor(hsv.z * color_bands + 0.5) / color_bands;

                // Apply brightness boost
                hsv.z = clamp(hsv.z + brightness_boost, 0.0, 1.0);

                vec3 quantized = hsv2rgb(hsv);

                // Add shadow tint to dark areas
                float shadow_mask = 1.0 - smoothstep(0.0, 0.4, original_v);
                quantized = mix(quantized, quantized * vec3(0.8, 0.85, 1.0), shadow_mask * shadow_intensity);

                // Add highlight to bright areas
                float highlight_mask = smoothstep(0.6, 1.0, original_v);
                quantized = mix(quantized, quantized + vec3(0.1), highlight_mask * highlight_intensity);

                // Edge detection
                float samples[9];
                int idx = 0;
                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        vec3 c = texture(u_texture, v_uv + vec2(x, y) * pixel).rgb;
                        samples[idx++] = luminance(c);
                    }
                }

                float gx = samples[2] + 2.0*samples[5] + samples[8] - samples[0] - 2.0*samples[3] - samples[6];
                float gy = samples[0] + 2.0*samples[1] + samples[2] - samples[6] - 2.0*samples[7] - samples[8];
                float edge = sqrt(gx*gx + gy*gy);

                float edge_mask = smoothstep(edge_threshold, edge_threshold + edge_softness, edge);
                vec3 outline_color = vec3(outline_color_r, outline_color_g, outline_color_b);
                vec3 final_color = mix(quantized, outline_color, edge_mask);

                f_color = vec4(final_color, base.a);
            }
        """
    },

    "Comic Book": {
        "category": "Comic",
        "description": "Classic comic book style with halftone dots and bold outlines",
        "uniforms": {
            "dot_size": {"min": 2.0, "max": 30.0, "default": 8.0, "step": 1.0},
            "dot_angle": {"min": 0.0, "max": 1.57, "default": 0.785, "step": 0.05},
            "dot_softness": {"min": 0.01, "max": 0.2, "default": 0.05, "step": 0.01},
            "outline_thickness": {"min": 0.5, "max": 8.0, "default": 2.0, "step": 0.25},
            "outline_threshold": {"min": 0.05, "max": 0.5, "default": 0.15, "step": 0.01},
            "outline_softness": {"min": 0.0, "max": 0.3, "default": 0.1, "step": 0.01},
            "color_levels": {"min": 2.0, "max": 16.0, "default": 4.0, "step": 1.0},
            "color_strength": {"min": 0.0, "max": 1.0, "default": 0.8, "step": 0.05},
            "saturation": {"min": 0.0, "max": 2.0, "default": 1.2, "step": 0.05},
            "ink_density": {"min": 0.0, "max": 1.0, "default": 0.9, "step": 0.05},
            "paper_white": {"min": 0.8, "max": 1.0, "default": 0.95, "step": 0.01},
            "cmyk_mode": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float dot_size;
            uniform float dot_angle;
            uniform float dot_softness;
            uniform float outline_thickness;
            uniform float outline_threshold;
            uniform float outline_softness;
            uniform float color_levels;
            uniform float color_strength;
            uniform float saturation;
            uniform float ink_density;
            uniform float paper_white;
            uniform float cmyk_mode;
            in vec2 v_uv;
            out vec4 f_color;

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            float halftone(vec2 uv, float angle, float size, float value) {
                float c = cos(angle);
                float s = sin(angle);
                vec2 rotated = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
                vec2 cell = fract(rotated / size) - 0.5;
                float dist = length(cell);
                float radius = (1.0 - value) * 0.5;
                return smoothstep(radius - dot_softness, radius + dot_softness, dist);
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = outline_thickness / tex_size;
                vec2 uv_pixels = v_uv * tex_size;

                vec4 base = texture(u_texture, v_uv);

                // Boost saturation
                float gray = luminance(base.rgb);
                vec3 saturated = mix(vec3(gray), base.rgb, saturation);

                // Quantize colors
                vec3 comic_color = floor(saturated * color_levels + 0.5) / color_levels;

                vec3 color;
                if(cmyk_mode > 0.5) {
                    // CMYK-style halftone
                    float c_dot = halftone(uv_pixels, dot_angle, dot_size, 1.0 - comic_color.r);
                    float m_dot = halftone(uv_pixels, dot_angle + 0.26, dot_size, 1.0 - comic_color.g);
                    float y_dot = halftone(uv_pixels, dot_angle + 0.52, dot_size, 1.0 - comic_color.b);
                    color = vec3(c_dot, m_dot, y_dot);
                } else {
                    // Single-channel halftone
                    float lum = luminance(comic_color);
                    float halftone_val = halftone(uv_pixels, dot_angle, dot_size, lum);
                    color = mix(vec3(0.0), comic_color, halftone_val);
                }

                // Mix with paper color
                color = mix(vec3(paper_white), color, color_strength);

                // Edge detection for ink outlines
                float samples[9];
                int idx = 0;
                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        samples[idx++] = luminance(texture(u_texture, v_uv + vec2(x, y) * pixel).rgb);
                    }
                }
                float gx = samples[2] + 2.0*samples[5] + samples[8] - samples[0] - 2.0*samples[3] - samples[6];
                float gy = samples[0] + 2.0*samples[1] + samples[2] - samples[6] - 2.0*samples[7] - samples[8];
                float edge = sqrt(gx*gx + gy*gy);

                float edge_mask = smoothstep(outline_threshold, outline_threshold + outline_softness, edge);
                color = mix(color, vec3(0.0), edge_mask * ink_density);

                f_color = vec4(color, base.a);
            }
        """
    },

    "Sobel Edge": {
        "category": "Effects",
        "description": "Edge detection with customizable appearance",
        "uniforms": {
            "edge_intensity": {"min": 0.5, "max": 10.0, "default": 2.0, "step": 0.1},
            "threshold": {"min": 0.0, "max": 0.5, "default": 0.1, "step": 0.01},
            "softness": {"min": 0.0, "max": 0.3, "default": 0.05, "step": 0.01},
            "invert": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
            "color_edges": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
            "edge_color_r": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.05},
            "edge_color_g": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.05},
            "edge_color_b": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.05},
            "bg_color_r": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "bg_color_g": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "bg_color_b": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "show_direction": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float edge_intensity;
            uniform float threshold;
            uniform float softness;
            uniform float invert;
            uniform float color_edges;
            uniform float edge_color_r;
            uniform float edge_color_g;
            uniform float edge_color_b;
            uniform float bg_color_r;
            uniform float bg_color_g;
            uniform float bg_color_b;
            uniform float show_direction;
            in vec2 v_uv;
            out vec4 f_color;

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = 1.0 / tex_size;

                vec4 base = texture(u_texture, v_uv);
                vec3 base_color = base.rgb;

                float samples[9];
                int idx = 0;
                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        samples[idx++] = luminance(texture(u_texture, v_uv + vec2(x, y) * pixel).rgb);
                    }
                }

                float gx = samples[2] + 2.0*samples[5] + samples[8] - samples[0] - 2.0*samples[3] - samples[6];
                float gy = samples[0] + 2.0*samples[1] + samples[2] - samples[6] - 2.0*samples[7] - samples[8];
                float edge = sqrt(gx*gx + gy*gy) * edge_intensity;
                edge = smoothstep(threshold, threshold + softness, edge);

                if(invert > 0.5) edge = 1.0 - edge;

                vec3 edge_col = vec3(edge_color_r, edge_color_g, edge_color_b);
                vec3 bg_col = vec3(bg_color_r, bg_color_g, bg_color_b);

                vec3 color;
                if(show_direction > 0.5) {
                    // Show edge direction as color
                    float angle = atan(gy, gx);
                    color = vec3(
                        0.5 + 0.5 * cos(angle),
                        0.5 + 0.5 * cos(angle + 2.094),
                        0.5 + 0.5 * cos(angle + 4.189)
                    ) * edge;
                } else if(color_edges > 0.5) {
                    // Use original colors for edges
                    color = mix(bg_col, base_color * edge_col, edge);
                } else {
                    color = mix(bg_col, edge_col, edge);
                }

                f_color = vec4(color, base.a);
            }
        """
    },

    "Color Grade": {
        "category": "Color",
        "description": "Professional color grading with full control",
        "uniforms": {
            "exposure": {"min": -3.0, "max": 3.0, "default": 0.0, "step": 0.05},
            "brightness": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.01},
            "contrast": {"min": 0.0, "max": 3.0, "default": 1.0, "step": 0.02},
            "saturation": {"min": 0.0, "max": 3.0, "default": 1.0, "step": 0.02},
            "vibrance": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "temperature": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "tint": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "gamma": {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.02},
            "lift_r": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
            "lift_g": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
            "lift_b": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
            "gain_r": {"min": 0.5, "max": 1.5, "default": 1.0, "step": 0.02},
            "gain_g": {"min": 0.5, "max": 1.5, "default": 1.0, "step": 0.02},
            "gain_b": {"min": 0.5, "max": 1.5, "default": 1.0, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float exposure;
            uniform float brightness;
            uniform float contrast;
            uniform float saturation;
            uniform float vibrance;
            uniform float temperature;
            uniform float tint;
            uniform float gamma;
            uniform float lift_r, lift_g, lift_b;
            uniform float gain_r, gain_g, gain_b;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec4 color = texture(u_texture, v_uv);

                // Exposure (EV stops)
                color.rgb *= pow(2.0, exposure);

                // Lift (shadows)
                color.rgb += vec3(lift_r, lift_g, lift_b);

                // Gain (highlights)
                color.rgb *= vec3(gain_r, gain_g, gain_b);

                // Brightness
                color.rgb += brightness;

                // Contrast
                color.rgb = (color.rgb - 0.5) * contrast + 0.5;

                // Temperature (warm/cool)
                color.r += temperature * 0.1;
                color.b -= temperature * 0.1;

                // Tint (green/magenta)
                color.g += tint * 0.1;

                // Saturation
                float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                color.rgb = mix(vec3(gray), color.rgb, saturation);

                // Vibrance (saturate less-saturated colors more)
                float max_c = max(color.r, max(color.g, color.b));
                float min_c = min(color.r, min(color.g, color.b));
                float sat = (max_c - min_c) / (max_c + 0.001);
                float vibrance_factor = 1.0 + vibrance * (1.0 - sat);
                color.rgb = mix(vec3(gray), color.rgb, vibrance_factor);

                // Gamma
                color.rgb = pow(max(color.rgb, vec3(0.0)), vec3(1.0 / gamma));

                f_color = vec4(clamp(color.rgb, 0.0, 1.0), color.a);
            }
        """
    },

    "Sepia": {
        "category": "Color",
        "description": "Vintage sepia/duotone effect with customizable colors",
        "uniforms": {
            "intensity": {"min": 0.0, "max": 1.0, "default": 0.8, "step": 0.02},
            "shadow_r": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.02},
            "shadow_g": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.02},
            "shadow_b": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "highlight_r": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.02},
            "highlight_g": {"min": 0.0, "max": 1.0, "default": 0.9, "step": 0.02},
            "highlight_b": {"min": 0.0, "max": 1.0, "default": 0.7, "step": 0.02},
            "contrast": {"min": 0.5, "max": 2.0, "default": 1.1, "step": 0.02},
            "grain": {"min": 0.0, "max": 0.3, "default": 0.05, "step": 0.01},
            "vignette": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float intensity;
            uniform float shadow_r, shadow_g, shadow_b;
            uniform float highlight_r, highlight_g, highlight_b;
            uniform float contrast;
            uniform float grain;
            uniform float vignette;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec4 color = texture(u_texture, v_uv);
                vec2 tex_size = vec2(textureSize(u_texture, 0));

                // Convert to luminance
                float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));

                // Apply contrast
                lum = (lum - 0.5) * contrast + 0.5;
                lum = clamp(lum, 0.0, 1.0);

                // Duotone mapping
                vec3 shadow = vec3(shadow_r, shadow_g, shadow_b);
                vec3 highlight = vec3(highlight_r, highlight_g, highlight_b);
                vec3 toned = mix(shadow, highlight, lum);

                // Mix with original
                vec3 result = mix(color.rgb, toned, intensity);

                // Film grain
                if(grain > 0.0) {
                    float noise = (hash(v_uv * tex_size + fract(float(1))) - 0.5) * grain;
                    result += noise;
                }

                // Vignette
                if(vignette > 0.0) {
                    float dist = length(v_uv - 0.5) * 1.414;
                    result *= 1.0 - dist * dist * vignette;
                }

                f_color = vec4(clamp(result, 0.0, 1.0), color.a);
            }
        """
    },

    "Vignette": {
        "category": "Effects",
        "description": "Customizable vignette with shape and color controls",
        "uniforms": {
            "intensity": {"min": 0.0, "max": 2.0, "default": 0.8, "step": 0.02},
            "radius": {"min": 0.0, "max": 2.0, "default": 0.7, "step": 0.02},
            "softness": {"min": 0.01, "max": 1.5, "default": 0.4, "step": 0.02},
            "roundness": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.02},
            "center_x": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
            "center_y": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
            "color_r": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "color_g": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "color_b": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "highlight_boost": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float intensity;
            uniform float radius;
            uniform float softness;
            uniform float roundness;
            uniform float center_x;
            uniform float center_y;
            uniform float color_r, color_g, color_b;
            uniform float highlight_boost;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec4 color = texture(u_texture, v_uv);
                vec2 center = vec2(center_x, center_y);
                vec2 diff = v_uv - center;

                // Aspect ratio correction
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                float aspect = tex_size.x / tex_size.y;
                diff.x *= mix(1.0, aspect, roundness);

                float dist = length(diff);
                float vignette = smoothstep(radius, radius - softness, dist);

                // Vignette color
                vec3 vig_color = vec3(color_r, color_g, color_b);

                // Apply vignette
                vec3 result = mix(vig_color, color.rgb, mix(1.0 - intensity, 1.0, vignette));

                // Optional highlight boost in center
                if(highlight_boost > 0.0) {
                    float highlight = smoothstep(radius * 0.5, 0.0, dist);
                    result += highlight * highlight_boost * 0.2;
                }

                f_color = vec4(clamp(result, 0.0, 1.0), color.a);
            }
        """
    },

    "Chromatic Aberration": {
        "category": "Effects",
        "description": "Lens distortion with RGB channel separation",
        "uniforms": {
            "strength": {"min": 0.0, "max": 0.05, "default": 0.005, "step": 0.001},
            "radial_power": {"min": 0.5, "max": 3.0, "default": 1.0, "step": 0.1},
            "center_x": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
            "center_y": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
            "red_offset": {"min": -2.0, "max": 2.0, "default": 1.0, "step": 0.1},
            "green_offset": {"min": -2.0, "max": 2.0, "default": 0.0, "step": 0.1},
            "blue_offset": {"min": -2.0, "max": 2.0, "default": -1.0, "step": 0.1},
            "barrel_distortion": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
            "blur_edges": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float strength;
            uniform float radial_power;
            uniform float center_x, center_y;
            uniform float red_offset, green_offset, blue_offset;
            uniform float barrel_distortion;
            uniform float blur_edges;
            in vec2 v_uv;
            out vec4 f_color;

            vec2 distort(vec2 uv, float k) {
                vec2 center = vec2(center_x, center_y);
                vec2 diff = uv - center;
                float r = length(diff);
                float f = 1.0 + k * r * r;
                return center + diff * f;
            }

            void main() {
                vec2 center = vec2(center_x, center_y);
                vec2 diff = v_uv - center;
                float dist = pow(length(diff), radial_power);

                // Per-channel offsets
                vec2 r_offset = diff * strength * red_offset * dist;
                vec2 g_offset = diff * strength * green_offset * dist;
                vec2 b_offset = diff * strength * blue_offset * dist;

                // Apply barrel distortion
                vec2 r_uv = distort(v_uv + r_offset, barrel_distortion);
                vec2 g_uv = distort(v_uv + g_offset, barrel_distortion);
                vec2 b_uv = distort(v_uv + b_offset, barrel_distortion);

                float r = texture(u_texture, r_uv).r;
                float g = texture(u_texture, g_uv).g;
                float b = texture(u_texture, b_uv).b;
                float a = texture(u_texture, v_uv).a;

                vec3 color = vec3(r, g, b);

                // Optional edge blur
                if(blur_edges > 0.0) {
                    float edge_dist = length(v_uv - center) * 2.0;
                    float blur_amount = smoothstep(0.5, 1.0, edge_dist) * blur_edges;
                    if(blur_amount > 0.0) {
                        vec2 pixel = 1.0 / vec2(textureSize(u_texture, 0));
                        vec3 blurred = vec3(0.0);
                        for(int y = -1; y <= 1; y++) {
                            for(int x = -1; x <= 1; x++) {
                                blurred += texture(u_texture, v_uv + vec2(x, y) * pixel * blur_amount * 3.0).rgb;
                            }
                        }
                        color = mix(color, blurred / 9.0, blur_amount);
                    }
                }

                f_color = vec4(color, a);
            }
        """
    },

    "Glitch": {
        "category": "Effects",
        "description": "Digital glitch distortion effect",
        "uniforms": {
            "block_size": {"min": 5.0, "max": 100.0, "default": 30.0, "step": 1.0},
            "intensity": {"min": 0.0, "max": 0.2, "default": 0.03, "step": 0.005},
            "color_shift": {"min": 0.0, "max": 0.1, "default": 0.02, "step": 0.002},
            "scan_lines": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
            "noise": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.02},
            "vertical_shift": {"min": 0.0, "max": 0.1, "default": 0.0, "step": 0.005},
            "seed": {"min": 0.0, "max": 100.0, "default": 42.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float block_size;
            uniform float intensity;
            uniform float color_shift;
            uniform float scan_lines;
            uniform float noise;
            uniform float vertical_shift;
            uniform float seed;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p + seed * 0.01, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 uv = v_uv;

                // Block-based horizontal displacement
                vec2 block = floor(uv * tex_size / block_size);
                float glitch_trigger = step(0.92, hash(block));

                if(glitch_trigger > 0.0) {
                    float offset = (hash(block + 0.1) - 0.5) * intensity;
                    uv.x += offset;
                }

                // Vertical shift
                if(vertical_shift > 0.0) {
                    float v_trigger = step(0.95, hash(vec2(floor(uv.y * 20.0), seed)));
                    uv.y += v_trigger * vertical_shift * (hash(vec2(uv.y, seed)) - 0.5);
                }

                // Color channel separation
                float r = texture(u_texture, uv + vec2(color_shift * glitch_trigger, 0.0)).r;
                float g = texture(u_texture, uv).g;
                float b = texture(u_texture, uv - vec2(color_shift * glitch_trigger, 0.0)).b;
                vec3 color = vec3(r, g, b);

                // Scan lines
                if(scan_lines > 0.0) {
                    float scanline = sin(v_uv.y * tex_size.y * 2.0) * 0.5 + 0.5;
                    color *= 1.0 - scanline * scan_lines * 0.3;
                }

                // Noise
                if(noise > 0.0) {
                    float n = hash(v_uv * tex_size + seed);
                    color += (n - 0.5) * noise * 0.5;
                }

                float a = texture(u_texture, v_uv).a;
                f_color = vec4(clamp(color, 0.0, 1.0), a);
            }
        """
    },

    "Sharpen": {
        "category": "Effects",
        "description": "Image sharpening with unsharp mask",
        "uniforms": {
            "amount": {"min": 0.0, "max": 5.0, "default": 1.0, "step": 0.1},
            "radius": {"min": 0.5, "max": 5.0, "default": 1.0, "step": 0.1},
            "threshold": {"min": 0.0, "max": 0.5, "default": 0.0, "step": 0.01},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float amount;
            uniform float radius;
            uniform float threshold;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = radius / tex_size;

                vec4 center = texture(u_texture, v_uv);

                // Box blur for unsharp mask
                vec3 blur = vec3(0.0);
                float count = 0.0;
                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        blur += texture(u_texture, v_uv + vec2(x, y) * pixel).rgb;
                        count += 1.0;
                    }
                }
                blur /= count;

                // Unsharp mask
                vec3 diff = center.rgb - blur;
                float diff_lum = dot(abs(diff), vec3(0.333));

                // Apply threshold
                if(diff_lum > threshold) {
                    center.rgb += diff * amount;
                }

                f_color = vec4(clamp(center.rgb, 0.0, 1.0), center.a);
            }
        """
    },

    "Blur": {
        "category": "Effects",
        "description": "Gaussian blur with adjustable radius",
        "uniforms": {
            "radius": {"min": 0.0, "max": 20.0, "default": 5.0, "step": 0.5},
            "quality": {"min": 1.0, "max": 3.0, "default": 2.0, "step": 1.0},
            "directional": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
            "direction_angle": {"min": 0.0, "max": 6.28, "default": 0.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float radius;
            uniform float quality;
            uniform float directional;
            uniform float direction_angle;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = 1.0 / tex_size;

                vec4 color = vec4(0.0);
                float total = 0.0;

                int samples = int(quality) * 2 + 1;
                float sigma = radius / 3.0;

                if(directional > 0.5) {
                    // Directional blur
                    vec2 dir = vec2(cos(direction_angle), sin(direction_angle)) * pixel * radius;
                    for(int i = -samples; i <= samples; i++) {
                        float weight = exp(-float(i*i) / (2.0 * sigma * sigma));
                        color += texture(u_texture, v_uv + dir * float(i) / float(samples)) * weight;
                        total += weight;
                    }
                } else {
                    // Box/Gaussian blur
                    for(int y = -samples; y <= samples; y++) {
                        for(int x = -samples; x <= samples; x++) {
                            float weight = exp(-(x*x + y*y) / (2.0 * sigma * sigma));
                            vec2 offset = vec2(x, y) * pixel * radius / float(samples);
                            color += texture(u_texture, v_uv + offset) * weight;
                            total += weight;
                        }
                    }
                }

                f_color = color / total;
            }
        """
    },

    "Oil Painting": {
        "category": "Artistic",
        "description": "Oil painting effect using Kuwahara filter",
        "uniforms": {
            "radius": {"min": 1.0, "max": 8.0, "default": 4.0, "step": 0.5},
            "sharpness": {"min": 1.0, "max": 20.0, "default": 8.0, "step": 0.5},
            "color_levels": {"min": 2.0, "max": 32.0, "default": 8.0, "step": 1.0},
            "saturation": {"min": 0.5, "max": 2.0, "default": 1.2, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float radius;
            uniform float sharpness;
            uniform float color_levels;
            uniform float saturation;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = radius / tex_size;

                int r = int(radius);
                int half_r = r / 2;

                vec3 mean[4];
                float variance[4];

                // Sample 4 quadrants
                for(int q = 0; q < 4; q++) {
                    vec3 sum = vec3(0.0);
                    vec3 sum2 = vec3(0.0);
                    float count = 0.0;

                    int ox = (q % 2) * half_r;
                    int oy = (q / 2) * half_r;

                    for(int j = 0; j <= half_r; j++) {
                        for(int i = 0; i <= half_r; i++) {
                            vec2 offset = vec2(float(i + ox - half_r), float(j + oy - half_r)) * pixel / float(r);
                            vec3 c = texture(u_texture, v_uv + offset).rgb;
                            sum += c;
                            sum2 += c * c;
                            count += 1.0;
                        }
                    }

                    mean[q] = sum / count;
                    vec3 var = (sum2 / count) - (mean[q] * mean[q]);
                    variance[q] = var.r + var.g + var.b;
                }

                // Weighted blend based on variance
                vec3 color = vec3(0.0);
                float total_weight = 0.0;
                for(int q = 0; q < 4; q++) {
                    float weight = 1.0 / (1.0 + variance[q] * sharpness);
                    color += mean[q] * weight;
                    total_weight += weight;
                }
                color /= total_weight;

                // Quantize colors for painterly look
                color = floor(color * color_levels + 0.5) / color_levels;

                // Boost saturation
                float gray = dot(color, vec3(0.299, 0.587, 0.114));
                color = mix(vec3(gray), color, saturation);

                vec4 base = texture(u_texture, v_uv);
                f_color = vec4(color, base.a);
            }
        """
    },

    "Watercolor": {
        "category": "Artistic",
        "description": "Watercolor painting effect with bleeding edges",
        "uniforms": {
            "blur_amount": {"min": 1.0, "max": 15.0, "default": 5.0, "step": 0.5},
            "edge_darkening": {"min": 0.0, "max": 1.0, "default": 0.4, "step": 0.05},
            "color_bleed": {"min": 0.0, "max": 2.0, "default": 0.8, "step": 0.05},
            "paper_texture": {"min": 0.0, "max": 1.0, "default": 0.4, "step": 0.02},
            "granulation": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
            "color_variance": {"min": 0.0, "max": 0.5, "default": 0.1, "step": 0.02},
            "wetness": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float blur_amount;
            uniform float edge_darkening;
            uniform float color_bleed;
            uniform float paper_texture;
            uniform float granulation;
            uniform float color_variance;
            uniform float wetness;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = hash(i);
                float b = hash(i + vec2(1.0, 0.0));
                float c = hash(i + vec2(0.0, 1.0));
                float d = hash(i + vec2(1.0, 1.0));
                return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
            }

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = blur_amount / tex_size;

                // Wobbly sampling for watercolor bleeding
                vec2 wobble = vec2(
                    noise(v_uv * tex_size * 0.05) - 0.5,
                    noise(v_uv * tex_size * 0.05 + 100.0) - 0.5
                ) * color_bleed * pixel * 3.0;

                // Blur with wobble
                vec3 color = vec3(0.0);
                float total = 0.0;
                for(int y = -2; y <= 2; y++) {
                    for(int x = -2; x <= 2; x++) {
                        float weight = exp(-(x*x + y*y) / 4.0);
                        vec2 offset = vec2(x, y) * pixel + wobble * wetness;
                        vec3 sample_color = texture(u_texture, v_uv + offset).rgb;

                        // Add color variance
                        if(color_variance > 0.0) {
                            float hue_shift = (noise(v_uv * tex_size * 0.1 + vec2(x, y)) - 0.5) * color_variance;
                            sample_color = mix(sample_color, sample_color.gbr, hue_shift);
                        }

                        color += sample_color * weight;
                        total += weight;
                    }
                }
                color /= total;

                // Edge darkening (pigment pooling)
                vec4 base = texture(u_texture, v_uv);
                float edge = 0.0;
                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        if(x == 0 && y == 0) continue;
                        vec3 c = texture(u_texture, v_uv + vec2(x, y) * pixel * 0.5).rgb;
                        edge += length(base.rgb - c);
                    }
                }
                edge = clamp(edge * 0.5, 0.0, 1.0);
                color = mix(color, color * 0.6, edge * edge_darkening);

                // Granulation
                if(granulation > 0.0) {
                    float gran = noise(v_uv * tex_size * 0.3);
                    float lum = luminance(color);
                    color *= 1.0 - (1.0 - lum) * gran * granulation * 0.3;
                }

                // Paper texture
                if(paper_texture > 0.0) {
                    float paper = 0.9 + noise(v_uv * tex_size * 0.2) * paper_texture * 0.2;
                    color *= paper;
                }

                // Lighten for watercolor feel
                color = mix(color, vec3(1.0), 0.05 * wetness);

                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "Noir": {
        "category": "Cinematic",
        "description": "Film noir style with high contrast black and white",
        "uniforms": {
            "contrast": {"min": 1.0, "max": 4.0, "default": 2.0, "step": 0.05},
            "brightness": {"min": -0.5, "max": 0.5, "default": -0.1, "step": 0.02},
            "grain_intensity": {"min": 0.0, "max": 0.5, "default": 0.15, "step": 0.02},
            "vignette_strength": {"min": 0.0, "max": 2.0, "default": 1.0, "step": 0.05},
            "vignette_radius": {"min": 0.3, "max": 1.5, "default": 0.8, "step": 0.05},
            "shadow_crush": {"min": 0.0, "max": 0.3, "default": 0.1, "step": 0.01},
            "highlight_bloom": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float contrast;
            uniform float brightness;
            uniform float grain_intensity;
            uniform float vignette_strength;
            uniform float vignette_radius;
            uniform float shadow_crush;
            uniform float highlight_bloom;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec4 color = texture(u_texture, v_uv);
                vec2 tex_size = vec2(textureSize(u_texture, 0));

                // Convert to grayscale with film-like response
                float lum = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));

                // Apply contrast curve
                lum = (lum - 0.5) * contrast + 0.5 + brightness;

                // Crush shadows
                lum = max(lum - shadow_crush, 0.0) / (1.0 - shadow_crush);

                // Highlight bloom
                if(highlight_bloom > 0.0 && lum > 0.8) {
                    lum += (lum - 0.8) * highlight_bloom;
                }

                // Film grain
                float grain = (hash(v_uv * tex_size) - 0.5) * grain_intensity;
                lum += grain;

                // Vignette
                float dist = length(v_uv - 0.5) * 1.414;
                float vig = smoothstep(vignette_radius, vignette_radius - 0.5, dist);
                lum *= mix(1.0 - vignette_strength, 1.0, vig);

                f_color = vec4(vec3(clamp(lum, 0.0, 1.0)), color.a);
            }
        """
    },

    "Cyberpunk": {
        "category": "Cinematic",
        "description": "Neon-lit cyberpunk aesthetic with color grading",
        "uniforms": {
            "neon_intensity": {"min": 0.0, "max": 2.0, "default": 1.0, "step": 0.05},
            "cyan_amount": {"min": 0.0, "max": 1.0, "default": 0.6, "step": 0.02},
            "magenta_amount": {"min": 0.0, "max": 1.0, "default": 0.8, "step": 0.02},
            "contrast": {"min": 0.5, "max": 2.0, "default": 1.3, "step": 0.02},
            "glow_threshold": {"min": 0.3, "max": 0.9, "default": 0.6, "step": 0.02},
            "glow_intensity": {"min": 0.0, "max": 1.0, "default": 0.4, "step": 0.02},
            "scanlines": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.02},
            "chromatic_aberration": {"min": 0.0, "max": 0.02, "default": 0.003, "step": 0.001},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float neon_intensity;
            uniform float cyan_amount;
            uniform float magenta_amount;
            uniform float contrast;
            uniform float glow_threshold;
            uniform float glow_intensity;
            uniform float scanlines;
            uniform float chromatic_aberration;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 center = vec2(0.5);
                vec2 dir = v_uv - center;

                // Chromatic aberration
                float r = texture(u_texture, v_uv + dir * chromatic_aberration).r;
                float g = texture(u_texture, v_uv).g;
                float b = texture(u_texture, v_uv - dir * chromatic_aberration).b;
                vec3 color = vec3(r, g, b);

                // Contrast
                color = (color - 0.5) * contrast + 0.5;

                // Cyberpunk color grading - push shadows to cyan, highlights to magenta
                float lum = dot(color, vec3(0.299, 0.587, 0.114));
                vec3 cyan = vec3(0.0, 0.8, 1.0);
                vec3 magenta = vec3(1.0, 0.2, 0.8);

                vec3 shadow_tint = mix(color, color * cyan, (1.0 - lum) * cyan_amount * neon_intensity);
                vec3 highlight_tint = mix(shadow_tint, shadow_tint + magenta * 0.3, lum * magenta_amount * neon_intensity);
                color = highlight_tint;

                // Glow on bright areas
                if(glow_intensity > 0.0) {
                    float glow = smoothstep(glow_threshold, 1.0, lum) * glow_intensity;
                    color += vec3(glow * 0.5, glow * 0.2, glow * 0.8);
                }

                // Scanlines
                if(scanlines > 0.0) {
                    float scan = sin(v_uv.y * tex_size.y * 2.0) * 0.5 + 0.5;
                    color *= 1.0 - scan * scanlines * 0.3;
                }

                f_color = vec4(clamp(color, 0.0, 1.0), texture(u_texture, v_uv).a);
            }
        """
    },

    "Vintage Film": {
        "category": "Cinematic",
        "description": "Aged film look with fading, scratches, and color shift",
        "uniforms": {
            "age_amount": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
            "fade_amount": {"min": 0.0, "max": 0.5, "default": 0.15, "step": 0.02},
            "warmth": {"min": 0.0, "max": 1.0, "default": 0.4, "step": 0.02},
            "grain": {"min": 0.0, "max": 0.3, "default": 0.1, "step": 0.01},
            "vignette": {"min": 0.0, "max": 1.5, "default": 0.6, "step": 0.02},
            "scratches": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
            "flicker": {"min": 0.0, "max": 0.2, "default": 0.05, "step": 0.01},
            "saturation": {"min": 0.0, "max": 1.5, "default": 0.7, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float age_amount;
            uniform float fade_amount;
            uniform float warmth;
            uniform float grain;
            uniform float vignette;
            uniform float scratches;
            uniform float flicker;
            uniform float saturation;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
                           mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
            }

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec3 color = base.rgb;

                // Desaturate
                float gray = dot(color, vec3(0.299, 0.587, 0.114));
                color = mix(vec3(gray), color, saturation);

                // Warm color shift
                color.r += warmth * 0.1 * age_amount;
                color.g += warmth * 0.05 * age_amount;
                color.b -= warmth * 0.1 * age_amount;

                // Fade blacks and whites
                color = mix(color, vec3(0.1 + fade_amount * 0.2), fade_amount * (1.0 - gray));
                color = mix(color, vec3(0.9 - fade_amount * 0.1), fade_amount * gray * 0.5);

                // Film grain
                float g = (hash(v_uv * tex_size + fract(float(1))) - 0.5) * grain;
                color += g;

                // Vertical scratches
                if(scratches > 0.0) {
                    float scratch = step(0.998, hash(vec2(floor(v_uv.x * tex_size.x * 0.1), 1.0)));
                    scratch *= noise(vec2(v_uv.y * 50.0, floor(v_uv.x * tex_size.x * 0.1)));
                    color += scratch * scratches * 0.3;
                }

                // Flicker
                float flick = 1.0 + (hash(vec2(floor(v_uv.y * 3.0), 1.0)) - 0.5) * flicker;
                color *= flick;

                // Vignette
                float dist = length(v_uv - 0.5) * 1.414;
                color *= 1.0 - dist * dist * vignette;

                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "Cross Process": {
        "category": "Color",
        "description": "Cross-processed film look with shifted colors",
        "uniforms": {
            "intensity": {"min": 0.0, "max": 1.0, "default": 0.7, "step": 0.02},
            "contrast": {"min": 0.5, "max": 2.0, "default": 1.4, "step": 0.02},
            "saturation": {"min": 0.5, "max": 2.0, "default": 1.3, "step": 0.02},
            "cyan_red": {"min": -1.0, "max": 1.0, "default": 0.2, "step": 0.02},
            "magenta_green": {"min": -1.0, "max": 1.0, "default": -0.1, "step": 0.02},
            "yellow_blue": {"min": -1.0, "max": 1.0, "default": 0.3, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float intensity;
            uniform float contrast;
            uniform float saturation;
            uniform float cyan_red;
            uniform float magenta_green;
            uniform float yellow_blue;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec3 color = base.rgb;

                // Apply color channel curves (simplified cross-process)
                color.r = pow(color.r, 1.0 - cyan_red * 0.3);
                color.g = pow(color.g, 1.0 - magenta_green * 0.3);
                color.b = pow(color.b, 1.0 + yellow_blue * 0.3);

                // Shadows get different treatment than highlights
                float lum = dot(color, vec3(0.299, 0.587, 0.114));
                color.r += (1.0 - lum) * cyan_red * 0.1;
                color.b += lum * yellow_blue * 0.15;

                // Contrast
                color = (color - 0.5) * contrast + 0.5;

                // Saturation
                float gray = dot(color, vec3(0.299, 0.587, 0.114));
                color = mix(vec3(gray), color, saturation);

                // Mix with original
                color = mix(base.rgb, color, intensity);

                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "Posterize": {
        "category": "Stylized",
        "description": "Reduce colors to create poster-like effect",
        "uniforms": {
            "levels": {"min": 2.0, "max": 16.0, "default": 4.0, "step": 1.0},
            "saturation": {"min": 0.0, "max": 2.0, "default": 1.2, "step": 0.05},
            "outline": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
            "outline_thickness": {"min": 0.5, "max": 4.0, "default": 1.5, "step": 0.1},
            "smooth_edges": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float levels;
            uniform float saturation;
            uniform float outline;
            uniform float outline_thickness;
            uniform float smooth_edges;
            in vec2 v_uv;
            out vec4 f_color;

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = outline_thickness / tex_size;

                vec4 base = texture(u_texture, v_uv);
                vec3 color = base.rgb;

                // Boost saturation
                float gray = luminance(color);
                color = mix(vec3(gray), color, saturation);

                // Posterize with optional smoothing
                if(smooth_edges > 0.0) {
                    vec3 stepped = floor(color * levels + 0.5) / levels;
                    color = mix(stepped, color, smooth_edges * 0.5);
                    color = floor(color * levels + 0.5) / levels;
                } else {
                    color = floor(color * levels + 0.5) / levels;
                }

                // Edge detection for outline
                if(outline > 0.0) {
                    float samples[9];
                    int idx = 0;
                    for(int y = -1; y <= 1; y++) {
                        for(int x = -1; x <= 1; x++) {
                            samples[idx++] = luminance(texture(u_texture, v_uv + vec2(x, y) * pixel).rgb);
                        }
                    }
                    float gx = samples[2] + 2.0*samples[5] + samples[8] - samples[0] - 2.0*samples[3] - samples[6];
                    float gy = samples[0] + 2.0*samples[1] + samples[2] - samples[6] - 2.0*samples[7] - samples[8];
                    float edge = sqrt(gx*gx + gy*gy);
                    color = mix(color, vec3(0.0), smoothstep(0.1, 0.3, edge) * outline);
                }

                f_color = vec4(color, base.a);
            }
        """
    },

    "Thermal": {
        "category": "Scientific",
        "description": "Thermal/infrared camera visualization",
        "uniforms": {
            "sensitivity": {"min": 0.5, "max": 2.0, "default": 1.0, "step": 0.02},
            "cold_threshold": {"min": 0.0, "max": 0.5, "default": 0.25, "step": 0.02},
            "hot_threshold": {"min": 0.5, "max": 1.0, "default": 0.75, "step": 0.02},
            "color_mode": {"min": 0.0, "max": 2.0, "default": 0.0, "step": 1.0},
            "noise": {"min": 0.0, "max": 0.2, "default": 0.05, "step": 0.01},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float sensitivity;
            uniform float cold_threshold;
            uniform float hot_threshold;
            uniform float color_mode;
            uniform float noise;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            vec3 thermal_palette(float t) {
                // Iron palette: black -> blue -> magenta -> orange -> yellow -> white
                if(t < 0.2) return mix(vec3(0.0), vec3(0.0, 0.0, 0.5), t * 5.0);
                if(t < 0.4) return mix(vec3(0.0, 0.0, 0.5), vec3(0.8, 0.0, 0.8), (t - 0.2) * 5.0);
                if(t < 0.6) return mix(vec3(0.8, 0.0, 0.8), vec3(1.0, 0.5, 0.0), (t - 0.4) * 5.0);
                if(t < 0.8) return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.6) * 5.0);
                return mix(vec3(1.0, 1.0, 0.0), vec3(1.0), (t - 0.8) * 5.0);
            }

            vec3 rainbow_palette(float t) {
                return vec3(
                    sin(t * 6.28) * 0.5 + 0.5,
                    sin(t * 6.28 + 2.09) * 0.5 + 0.5,
                    sin(t * 6.28 + 4.19) * 0.5 + 0.5
                );
            }

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec2 tex_size = vec2(textureSize(u_texture, 0));

                float temp = dot(base.rgb, vec3(0.299, 0.587, 0.114));
                temp = clamp((temp - cold_threshold) / (hot_threshold - cold_threshold), 0.0, 1.0);
                temp = pow(temp, 1.0 / sensitivity);

                // Add noise
                temp += (hash(v_uv * tex_size) - 0.5) * noise;
                temp = clamp(temp, 0.0, 1.0);

                vec3 color;
                if(color_mode < 0.5) {
                    color = thermal_palette(temp);
                } else if(color_mode < 1.5) {
                    color = rainbow_palette(temp);
                } else {
                    // Grayscale
                    color = vec3(temp);
                }

                f_color = vec4(color, base.a);
            }
        """
    },

    "Emboss": {
        "category": "Effects",
        "description": "3D embossed relief effect",
        "uniforms": {
            "strength": {"min": 0.0, "max": 5.0, "default": 2.0, "step": 0.1},
            "angle": {"min": 0.0, "max": 6.28, "default": 0.785, "step": 0.1},
            "blend": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
            "gray_base": {"min": 0.3, "max": 0.7, "default": 0.5, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float strength;
            uniform float angle;
            uniform float blend;
            uniform float gray_base;
            in vec2 v_uv;
            out vec4 f_color;

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = 1.0 / tex_size;

                vec2 dir = vec2(cos(angle), sin(angle)) * pixel;

                float s1 = luminance(texture(u_texture, v_uv - dir).rgb);
                float s2 = luminance(texture(u_texture, v_uv + dir).rgb);

                float emboss = (s2 - s1) * strength + gray_base;

                vec4 base = texture(u_texture, v_uv);
                vec3 color = mix(vec3(emboss), base.rgb * emboss * 2.0, blend);

                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "Mosaic": {
        "category": "Stylized",
        "description": "Stained glass mosaic effect",
        "uniforms": {
            "cell_size": {"min": 5.0, "max": 50.0, "default": 20.0, "step": 1.0},
            "edge_width": {"min": 0.0, "max": 0.3, "default": 0.1, "step": 0.01},
            "edge_color_r": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.02},
            "edge_color_g": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.02},
            "edge_color_b": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.02},
            "randomness": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
            "color_variation": {"min": 0.0, "max": 0.5, "default": 0.1, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float cell_size;
            uniform float edge_width;
            uniform float edge_color_r;
            uniform float edge_color_g;
            uniform float edge_color_b;
            uniform float randomness;
            uniform float color_variation;
            in vec2 v_uv;
            out vec4 f_color;

            vec2 hash2(vec2 p) {
                p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
                return fract(sin(p) * 43758.5453);
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 uv = v_uv * tex_size / cell_size;
                vec2 cell = floor(uv);
                vec2 local = fract(uv);

                // Voronoi-like cells with randomness
                float min_dist = 10.0;
                vec2 nearest_cell = cell;

                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        vec2 neighbor = cell + vec2(x, y);
                        vec2 point = neighbor + hash2(neighbor) * randomness;
                        float dist = length(uv - point);
                        if(dist < min_dist) {
                            min_dist = dist;
                            nearest_cell = neighbor;
                        }
                    }
                }

                // Sample color from cell center
                vec2 sample_uv = (nearest_cell + 0.5) * cell_size / tex_size;
                vec4 base = texture(u_texture, sample_uv);
                vec3 color = base.rgb;

                // Add color variation
                vec2 cell_hash = hash2(nearest_cell);
                color += (cell_hash.x - 0.5) * color_variation;
                color += (cell_hash.y - 0.5) * color_variation * vec3(1.0, -0.5, -0.5);

                // Edge detection
                float edge = smoothstep(0.5 - edge_width, 0.5, min_dist);
                vec3 edge_col = vec3(edge_color_r, edge_color_g, edge_color_b);
                color = mix(edge_col, color, edge);

                f_color = vec4(clamp(color, 0.0, 1.0), texture(u_texture, v_uv).a);
            }
        """
    },

    "Sketch Lines": {
        "category": "Sketch",
        "description": "Cross-hatch sketch drawing style",
        "uniforms": {
            "line_density": {"min": 20.0, "max": 100.0, "default": 50.0, "step": 5.0},
            "line_thickness": {"min": 0.1, "max": 1.0, "default": 0.4, "step": 0.02},
            "angle_1": {"min": 0.0, "max": 3.14, "default": 0.785, "step": 0.1},
            "angle_2": {"min": 0.0, "max": 3.14, "default": 2.356, "step": 0.1},
            "levels": {"min": 2.0, "max": 6.0, "default": 4.0, "step": 1.0},
            "paper_color_r": {"min": 0.8, "max": 1.0, "default": 0.95, "step": 0.02},
            "paper_color_g": {"min": 0.8, "max": 1.0, "default": 0.93, "step": 0.02},
            "paper_color_b": {"min": 0.7, "max": 1.0, "default": 0.88, "step": 0.02},
            "ink_color_r": {"min": 0.0, "max": 0.3, "default": 0.1, "step": 0.02},
            "ink_color_g": {"min": 0.0, "max": 0.3, "default": 0.08, "step": 0.02},
            "ink_color_b": {"min": 0.0, "max": 0.3, "default": 0.05, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float line_density;
            uniform float line_thickness;
            uniform float angle_1;
            uniform float angle_2;
            uniform float levels;
            uniform float paper_color_r, paper_color_g, paper_color_b;
            uniform float ink_color_r, ink_color_g, ink_color_b;
            in vec2 v_uv;
            out vec4 f_color;

            float hatch(vec2 uv, float angle, float density, float thickness) {
                float c = cos(angle);
                float s = sin(angle);
                vec2 rotated = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
                return smoothstep(thickness, thickness * 0.5, abs(sin(rotated.x * density)));
            }

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec2 tex_size = vec2(textureSize(u_texture, 0));

                float lum = dot(base.rgb, vec3(0.299, 0.587, 0.114));
                float dark = 1.0 - lum;

                // Quantize darkness levels
                dark = floor(dark * levels) / levels;

                vec2 uv = v_uv * tex_size;
                float lines = 0.0;

                // Apply hatching based on darkness
                if(dark > 0.2) {
                    lines = max(lines, hatch(uv, angle_1, line_density, line_thickness) * smoothstep(0.2, 0.4, dark));
                }
                if(dark > 0.4) {
                    lines = max(lines, hatch(uv, angle_2, line_density * 0.9, line_thickness) * smoothstep(0.4, 0.6, dark));
                }
                if(dark > 0.6) {
                    lines = max(lines, hatch(uv, angle_1 + 0.4, line_density * 1.1, line_thickness) * smoothstep(0.6, 0.8, dark));
                }
                if(dark > 0.8) {
                    lines = max(lines, hatch(uv, angle_2 + 0.4, line_density, line_thickness) * smoothstep(0.8, 1.0, dark));
                }

                vec3 paper = vec3(paper_color_r, paper_color_g, paper_color_b);
                vec3 ink = vec3(ink_color_r, ink_color_g, ink_color_b);
                vec3 color = mix(paper, ink, lines);

                f_color = vec4(color, base.a);
            }
        """
    },

    "Halftone": {
        "category": "Print",
        "description": "Classic print halftone dot pattern",
        "uniforms": {
            "dot_size": {"min": 2.0, "max": 20.0, "default": 6.0, "step": 0.5},
            "dot_angle": {"min": 0.0, "max": 1.57, "default": 0.262, "step": 0.05},
            "dot_shape": {"min": 0.0, "max": 2.0, "default": 0.0, "step": 1.0},
            "color_mode": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
            "contrast": {"min": 0.5, "max": 2.0, "default": 1.2, "step": 0.05},
            "paper_color_r": {"min": 0.8, "max": 1.0, "default": 1.0, "step": 0.02},
            "paper_color_g": {"min": 0.8, "max": 1.0, "default": 0.98, "step": 0.02},
            "paper_color_b": {"min": 0.8, "max": 1.0, "default": 0.95, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float dot_size;
            uniform float dot_angle;
            uniform float dot_shape;
            uniform float color_mode;
            uniform float contrast;
            uniform float paper_color_r, paper_color_g, paper_color_b;
            in vec2 v_uv;
            out vec4 f_color;

            float halftone_dot(vec2 uv, float angle, float size, float value) {
                float c = cos(angle);
                float s = sin(angle);
                vec2 rotated = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
                vec2 cell = fract(rotated / size) - 0.5;

                float dist;
                if(dot_shape < 0.5) {
                    // Circle
                    dist = length(cell);
                } else if(dot_shape < 1.5) {
                    // Diamond
                    dist = abs(cell.x) + abs(cell.y);
                } else {
                    // Square
                    dist = max(abs(cell.x), abs(cell.y));
                }

                float radius = (1.0 - value) * 0.5;
                return smoothstep(radius - 0.02, radius + 0.02, dist);
            }

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 uv = v_uv * tex_size;

                vec3 color = base.rgb;
                color = (color - 0.5) * contrast + 0.5;
                color = clamp(color, 0.0, 1.0);

                vec3 paper = vec3(paper_color_r, paper_color_g, paper_color_b);
                vec3 result;

                if(color_mode > 0.5) {
                    // CMYK-style
                    float c = halftone_dot(uv, dot_angle, dot_size, 1.0 - color.r);
                    float m = halftone_dot(uv, dot_angle + 0.26, dot_size, 1.0 - color.g);
                    float y = halftone_dot(uv, dot_angle + 0.52, dot_size, 1.0 - color.b);
                    result = vec3(c, m, y);
                } else {
                    // Grayscale
                    float lum = dot(color, vec3(0.299, 0.587, 0.114));
                    float dot = halftone_dot(uv, dot_angle, dot_size, lum);
                    result = mix(vec3(0.0), paper, dot);
                }

                f_color = vec4(result, base.a);
            }
        """
    },

    "Night Vision": {
        "category": "Scientific",
        "description": "Night vision goggle effect",
        "uniforms": {
            "brightness": {"min": 1.0, "max": 5.0, "default": 2.5, "step": 0.1},
            "noise": {"min": 0.0, "max": 0.5, "default": 0.15, "step": 0.02},
            "scanlines": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
            "vignette": {"min": 0.0, "max": 2.0, "default": 1.2, "step": 0.05},
            "green_tint": {"min": 0.5, "max": 1.0, "default": 0.8, "step": 0.02},
            "flicker": {"min": 0.0, "max": 0.2, "default": 0.05, "step": 0.01},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float brightness;
            uniform float noise;
            uniform float scanlines;
            uniform float vignette;
            uniform float green_tint;
            uniform float flicker;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec2 tex_size = vec2(textureSize(u_texture, 0));

                // Convert to luminance and boost
                float lum = dot(base.rgb, vec3(0.299, 0.587, 0.114));
                lum *= brightness;

                // Night vision green color
                vec3 nvColor = vec3(0.1, green_tint, 0.1);
                vec3 color = nvColor * lum;

                // Noise
                float n = (hash(v_uv * tex_size + fract(float(1))) - 0.5) * noise;
                color += n;

                // Scanlines
                float scan = sin(v_uv.y * tex_size.y * 2.0) * 0.5 + 0.5;
                color *= 1.0 - scan * scanlines * 0.5;

                // Vignette (circular, like goggles)
                float dist = length(v_uv - 0.5) * 2.0;
                float vig = 1.0 - smoothstep(0.7, 1.0, dist) * vignette;
                color *= vig;

                // Flicker
                color *= 1.0 + (hash(vec2(floor(v_uv.y * 5.0), 1.0)) - 0.5) * flicker;

                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "Dreamy Glow": {
        "category": "Artistic",
        "description": "Soft dreamy glow with bloom effect",
        "uniforms": {
            "glow_radius": {"min": 1.0, "max": 15.0, "default": 5.0, "step": 0.5},
            "glow_intensity": {"min": 0.0, "max": 2.0, "default": 0.8, "step": 0.05},
            "threshold": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
            "saturation": {"min": 0.5, "max": 1.5, "default": 1.1, "step": 0.02},
            "softness": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
            "tint_r": {"min": 0.8, "max": 1.2, "default": 1.0, "step": 0.02},
            "tint_g": {"min": 0.8, "max": 1.2, "default": 1.0, "step": 0.02},
            "tint_b": {"min": 0.8, "max": 1.2, "default": 1.05, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float glow_radius;
            uniform float glow_intensity;
            uniform float threshold;
            uniform float saturation;
            uniform float softness;
            uniform float tint_r, tint_g, tint_b;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = glow_radius / tex_size;

                vec4 base = texture(u_texture, v_uv);
                vec3 original = base.rgb;

                // Sample for blur/glow
                vec3 blur = vec3(0.0);
                float total = 0.0;
                int samples = 5;

                for(int y = -samples; y <= samples; y++) {
                    for(int x = -samples; x <= samples; x++) {
                        float weight = exp(-(x*x + y*y) / (2.0 * 4.0));
                        vec2 offset = vec2(x, y) * pixel / float(samples);
                        vec3 sample_color = texture(u_texture, v_uv + offset).rgb;

                        // Only blur bright areas
                        float lum = dot(sample_color, vec3(0.299, 0.587, 0.114));
                        if(lum > threshold) {
                            blur += sample_color * weight;
                            total += weight;
                        }
                    }
                }

                if(total > 0.0) blur /= total;
                else blur = vec3(0.0);

                // Combine original with glow
                vec3 color = original + blur * glow_intensity;

                // Apply softness (blend with blurred version)
                vec3 soft_blur = vec3(0.0);
                total = 0.0;
                for(int y = -2; y <= 2; y++) {
                    for(int x = -2; x <= 2; x++) {
                        float weight = 1.0;
                        soft_blur += texture(u_texture, v_uv + vec2(x, y) * pixel * 0.5).rgb * weight;
                        total += weight;
                    }
                }
                soft_blur /= total;
                color = mix(color, soft_blur, softness * 0.5);

                // Saturation
                float gray = dot(color, vec3(0.299, 0.587, 0.114));
                color = mix(vec3(gray), color, saturation);

                // Color tint
                color *= vec3(tint_r, tint_g, tint_b);

                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "Negative": {
        "category": "Color",
        "description": "Photo negative with adjustable color inversion",
        "uniforms": {
            "invert_amount": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.02},
            "preserve_luminance": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
            "color_shift_r": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
            "color_shift_g": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
            "color_shift_b": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
            "orange_mask": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float invert_amount;
            uniform float preserve_luminance;
            uniform float color_shift_r, color_shift_g, color_shift_b;
            uniform float orange_mask;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec3 color = base.rgb;

                // Invert
                vec3 inverted = 1.0 - color;

                // Optional: preserve luminance while inverting hue
                if(preserve_luminance > 0.5) {
                    float orig_lum = dot(color, vec3(0.299, 0.587, 0.114));
                    float inv_lum = dot(inverted, vec3(0.299, 0.587, 0.114));
                    inverted *= orig_lum / max(inv_lum, 0.001);
                }

                color = mix(color, inverted, invert_amount);

                // Color shift
                color.r += color_shift_r;
                color.g += color_shift_g;
                color.b += color_shift_b;

                // Film negative orange mask simulation
                if(orange_mask > 0.0) {
                    color.r += orange_mask * 0.2;
                    color.g += orange_mask * 0.1;
                }

                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "Split Tone": {
        "category": "Color",
        "description": "Split toning for shadows and highlights",
        "uniforms": {
            "shadow_hue": {"min": 0.0, "max": 1.0, "default": 0.6, "step": 0.02},
            "shadow_saturation": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
            "highlight_hue": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.02},
            "highlight_saturation": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.02},
            "balance": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "blend": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float shadow_hue;
            uniform float shadow_saturation;
            uniform float highlight_hue;
            uniform float highlight_saturation;
            uniform float balance;
            uniform float blend;
            in vec2 v_uv;
            out vec4 f_color;

            vec3 hsl2rgb(vec3 hsl) {
                vec3 rgb = clamp(abs(mod(hsl.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
                return hsl.z + hsl.y * (rgb - 0.5) * (1.0 - abs(2.0 * hsl.z - 1.0));
            }

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec3 color = base.rgb;

                float lum = dot(color, vec3(0.299, 0.587, 0.114));

                // Create tint colors
                vec3 shadow_tint = hsl2rgb(vec3(shadow_hue, shadow_saturation, 0.5));
                vec3 highlight_tint = hsl2rgb(vec3(highlight_hue, highlight_saturation, 0.5));

                // Adjust balance point
                float shadow_weight = smoothstep(0.5 + balance * 0.5, 0.0, lum);
                float highlight_weight = smoothstep(0.5 - balance * 0.5, 1.0, lum);

                // Apply tints
                vec3 tinted = color;
                tinted = mix(tinted, tinted * shadow_tint * 2.0, shadow_weight * shadow_saturation);
                tinted = mix(tinted, tinted * highlight_tint * 2.0, highlight_weight * highlight_saturation);

                color = mix(color, tinted, blend);

                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "ASCII Art": {
        "category": "Stylized",
        "description": "Convert image to ASCII character representation",
        "uniforms": {
            "cell_size": {"min": 4.0, "max": 20.0, "default": 8.0, "step": 1.0},
            "char_brightness": {"min": 0.5, "max": 2.0, "default": 1.0, "step": 0.05},
            "bg_color_r": {"min": 0.0, "max": 0.3, "default": 0.0, "step": 0.02},
            "bg_color_g": {"min": 0.0, "max": 0.3, "default": 0.0, "step": 0.02},
            "bg_color_b": {"min": 0.0, "max": 0.3, "default": 0.0, "step": 0.02},
            "text_color_r": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "text_color_g": {"min": 0.5, "max": 1.0, "default": 1.0, "step": 0.02},
            "text_color_b": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.02},
            "colored": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float cell_size;
            uniform float char_brightness;
            uniform float bg_color_r, bg_color_g, bg_color_b;
            uniform float text_color_r, text_color_g, text_color_b;
            uniform float colored;
            in vec2 v_uv;
            out vec4 f_color;

            // Simplified ASCII density patterns
            float ascii_pattern(vec2 uv, float lum) {
                vec2 cell = fract(uv);
                float density = 0.0;

                // Different patterns for different luminance levels
                if(lum > 0.9) {
                    density = 0.9; // Nearly solid
                } else if(lum > 0.7) {
                    density = step(0.3, cell.x) * step(0.3, cell.y) * 0.8; // Block
                } else if(lum > 0.5) {
                    density = (step(0.4, cell.x) + step(0.4, cell.y)) * 0.3; // Plus
                } else if(lum > 0.3) {
                    float d = length(cell - 0.5);
                    density = step(d, 0.25) * 0.5; // Dot
                } else if(lum > 0.15) {
                    density = step(0.45, cell.x) * step(0.3, cell.y) * step(cell.y, 0.7) * 0.3; // Dash
                } else if(lum > 0.05) {
                    density = step(0.45, cell.x) * step(0.45, cell.y) * 0.2; // Small dot
                }

                return density;
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 cell_uv = floor(v_uv * tex_size / cell_size) * cell_size / tex_size;
                vec2 local_uv = fract(v_uv * tex_size / cell_size);

                vec4 base = texture(u_texture, cell_uv + cell_size * 0.5 / tex_size);
                float lum = dot(base.rgb, vec3(0.299, 0.587, 0.114)) * char_brightness;
                lum = clamp(lum, 0.0, 1.0);

                float pattern = ascii_pattern(local_uv, lum);

                vec3 bg = vec3(bg_color_r, bg_color_g, bg_color_b);
                vec3 fg;
                if(colored > 0.5) {
                    fg = base.rgb;
                } else {
                    fg = vec3(text_color_r, text_color_g, text_color_b);
                }

                vec3 color = mix(bg, fg, pattern);

                f_color = vec4(color, 1.0);
            }
        """
    },

    "Bloom": {
        "category": "Effects",
        "description": "Glowing bloom/glow effect for bright areas",
        "uniforms": {
            "bloom_intensity": {"min": 0.0, "max": 3.0, "default": 1.0, "step": 0.05},
            "bloom_threshold": {"min": 0.0, "max": 1.0, "default": 0.7, "step": 0.02},
            "bloom_radius": {"min": 1.0, "max": 20.0, "default": 8.0, "step": 0.5},
            "bloom_softness": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05},
            "saturation": {"min": 0.0, "max": 2.0, "default": 1.0, "step": 0.05},
            "brightness": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float bloom_intensity;
            uniform float bloom_threshold;
            uniform float bloom_radius;
            uniform float bloom_softness;
            uniform float saturation;
            uniform float brightness;
            in vec2 v_uv;
            out vec4 f_color;

            vec3 sampleBloom(vec2 uv, float radius) {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = 1.0 / tex_size;
                vec3 bloom = vec3(0.0);
                float total = 0.0;
                int samples = int(radius);

                for(int y = -samples; y <= samples; y++) {
                    for(int x = -samples; x <= samples; x++) {
                        float dist = length(vec2(x, y)) / radius;
                        if(dist > 1.0) continue;
                        float weight = 1.0 - dist;
                        weight = pow(weight, 2.0);

                        vec3 col = texture(u_texture, uv + vec2(x, y) * pixel * radius * 0.5).rgb;
                        float lum = dot(col, vec3(0.299, 0.587, 0.114));
                        float mask = smoothstep(bloom_threshold - bloom_softness, bloom_threshold + bloom_softness, lum);
                        bloom += col * mask * weight;
                        total += weight;
                    }
                }
                return bloom / max(total, 1.0);
            }

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec3 bloom = sampleBloom(v_uv, bloom_radius);
                vec3 color = base.rgb + bloom * bloom_intensity;

                // Saturation
                float gray = dot(color, vec3(0.299, 0.587, 0.114));
                color = mix(vec3(gray), color, saturation);

                color += brightness;
                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "Depth of Field": {
        "category": "Effects",
        "description": "Simulated camera depth of field blur based on luminance",
        "uniforms": {
            "focus_point": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.02},
            "focus_range": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.02},
            "blur_amount": {"min": 0.0, "max": 20.0, "default": 5.0, "step": 0.5},
            "blur_quality": {"min": 1.0, "max": 3.0, "default": 2.0, "step": 1.0},
            "bokeh_shape": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.1},
            "brightness": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float focus_point;
            uniform float focus_range;
            uniform float blur_amount;
            uniform float blur_quality;
            uniform float bokeh_shape;
            uniform float brightness;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = 1.0 / tex_size;

                vec4 center = texture(u_texture, v_uv);
                float lum = dot(center.rgb, vec3(0.299, 0.587, 0.114));

                // Distance from focus point determines blur
                float dist = abs(lum - focus_point);
                float blur_factor = smoothstep(0.0, focus_range, dist);
                float radius = blur_amount * blur_factor;

                if(radius < 0.5) {
                    f_color = vec4(center.rgb + brightness, center.a);
                    return;
                }

                vec3 color = vec3(0.0);
                float total = 0.0;
                int samples = int(radius * blur_quality);
                samples = clamp(samples, 1, 16);

                for(int y = -samples; y <= samples; y++) {
                    for(int x = -samples; x <= samples; x++) {
                        float d = length(vec2(x, y)) / float(samples);
                        if(d > 1.0) continue;

                        // Bokeh shape (0 = circular, 1 = hexagonal-ish)
                        float weight = 1.0 - d * (1.0 - bokeh_shape);
                        weight = max(weight, 0.0);

                        vec2 offset = vec2(x, y) * pixel * radius / blur_quality;
                        color += texture(u_texture, v_uv + offset).rgb * weight;
                        total += weight;
                    }
                }

                color = color / max(total, 1.0);
                f_color = vec4(color + brightness, center.a);
            }
        """
    },

    "Film Grain": {
        "category": "Cinematic",
        "description": "Vintage film grain and noise effect",
        "uniforms": {
            "grain_amount": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
            "grain_size": {"min": 0.5, "max": 3.0, "default": 1.0, "step": 0.1},
            "grain_color": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
            "scratches": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.02},
            "dust": {"min": 0.0, "max": 1.0, "default": 0.05, "step": 0.01},
            "vignette": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
            "flicker": {"min": 0.0, "max": 0.2, "default": 0.02, "step": 0.01},
            "saturation": {"min": 0.0, "max": 1.5, "default": 0.9, "step": 0.05},
            "contrast": {"min": 0.5, "max": 1.5, "default": 1.1, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float grain_amount;
            uniform float grain_size;
            uniform float grain_color;
            uniform float scratches;
            uniform float dust;
            uniform float vignette;
            uniform float flicker;
            uniform float saturation;
            uniform float contrast;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
                           mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
            }

            void main() {
                vec4 base = texture(u_texture, v_uv);
                vec3 color = base.rgb;

                // Flicker
                float flick = 1.0 + (hash(vec2(floor(v_uv.y * 100.0), 0.0)) - 0.5) * flicker;
                color *= flick;

                // Film grain
                vec2 grain_uv = v_uv * vec2(textureSize(u_texture, 0)) / grain_size;
                float grain = (noise(grain_uv) - 0.5) * grain_amount;
                if(grain_color > 0.5) {
                    // Colored grain
                    color.r += grain * 1.2;
                    color.g += grain;
                    color.b += grain * 0.8;
                } else {
                    color += grain;
                }

                // Scratches
                if(scratches > 0.0) {
                    float scratch = step(0.998, hash(vec2(floor(v_uv.x * 500.0), 0.0)));
                    scratch *= step(hash(v_uv), scratches);
                    color = mix(color, vec3(0.9), scratch * 0.5);
                }

                // Dust particles
                if(dust > 0.0) {
                    float d = step(1.0 - dust * 0.01, hash(floor(v_uv * 200.0)));
                    color = mix(color, vec3(0.2), d * 0.3);
                }

                // Vignette
                vec2 vig_uv = v_uv * (1.0 - v_uv.yx);
                float vig = vig_uv.x * vig_uv.y * 15.0;
                vig = pow(vig, vignette * 0.5 + 0.1);
                color *= vig;

                // Saturation
                float gray = dot(color, vec3(0.299, 0.587, 0.114));
                color = mix(vec3(gray), color, saturation);

                // Contrast
                color = (color - 0.5) * contrast + 0.5;

                f_color = vec4(clamp(color, 0.0, 1.0), base.a);
            }
        """
    },

    "CRT Monitor": {
        "category": "Retro",
        "description": "Classic CRT monitor with scanlines and curvature",
        "uniforms": {
            "scanline_intensity": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.02},
            "scanline_count": {"min": 100.0, "max": 800.0, "default": 300.0, "step": 10.0},
            "curvature": {"min": 0.0, "max": 0.5, "default": 0.1, "step": 0.02},
            "vignette": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
            "brightness": {"min": 0.5, "max": 1.5, "default": 1.1, "step": 0.02},
            "saturation": {"min": 0.5, "max": 1.5, "default": 1.2, "step": 0.05},
            "rgb_offset": {"min": 0.0, "max": 5.0, "default": 1.0, "step": 0.2},
            "flicker": {"min": 0.0, "max": 0.1, "default": 0.02, "step": 0.005},
            "noise": {"min": 0.0, "max": 0.2, "default": 0.05, "step": 0.01},
            "glow": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float scanline_intensity;
            uniform float scanline_count;
            uniform float curvature;
            uniform float vignette;
            uniform float brightness;
            uniform float saturation;
            uniform float rgb_offset;
            uniform float flicker;
            uniform float noise;
            uniform float glow;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            vec2 curve(vec2 uv) {
                uv = uv * 2.0 - 1.0;
                vec2 offset = abs(uv.yx) / vec2(6.0, 4.0) * curvature;
                uv = uv + uv * offset * offset;
                uv = uv * 0.5 + 0.5;
                return uv;
            }

            void main() {
                vec2 curved_uv = curve(v_uv);

                // Check bounds
                if(curved_uv.x < 0.0 || curved_uv.x > 1.0 || curved_uv.y < 0.0 || curved_uv.y > 1.0) {
                    f_color = vec4(0.0, 0.0, 0.0, 1.0);
                    return;
                }

                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = 1.0 / tex_size;

                // RGB offset (chromatic aberration)
                float r = texture(u_texture, curved_uv + vec2(rgb_offset, 0.0) * pixel).r;
                float g = texture(u_texture, curved_uv).g;
                float b = texture(u_texture, curved_uv - vec2(rgb_offset, 0.0) * pixel).b;
                vec3 color = vec3(r, g, b);

                // Scanlines
                float scanline = sin(curved_uv.y * scanline_count * 3.14159) * 0.5 + 0.5;
                scanline = pow(scanline, 1.5) * scanline_intensity;
                color *= 1.0 - scanline;

                // Phosphor glow simulation
                if(glow > 0.0) {
                    vec3 bloom = vec3(0.0);
                    for(int i = -2; i <= 2; i++) {
                        for(int j = -2; j <= 2; j++) {
                            bloom += texture(u_texture, curved_uv + vec2(i, j) * pixel * 2.0).rgb;
                        }
                    }
                    bloom /= 25.0;
                    color += bloom * glow * 0.5;
                }

                // Brightness and saturation
                color *= brightness;
                float gray = dot(color, vec3(0.299, 0.587, 0.114));
                color = mix(vec3(gray), color, saturation);

                // Flicker
                color *= 1.0 + (hash(vec2(curved_uv.y * 100.0, 0.0)) - 0.5) * flicker;

                // Static noise
                color += (hash(curved_uv * tex_size) - 0.5) * noise;

                // Vignette
                vec2 vig_uv = curved_uv * (1.0 - curved_uv.yx);
                float vig = vig_uv.x * vig_uv.y * 15.0;
                vig = pow(vig, vignette * 0.5 + 0.1);
                color *= vig;

                f_color = vec4(clamp(color, 0.0, 1.0), 1.0);
            }
        """
    },

    "Motion Blur": {
        "category": "Effects",
        "description": "Directional motion blur effect",
        "uniforms": {
            "blur_amount": {"min": 0.0, "max": 50.0, "default": 10.0, "step": 1.0},
            "blur_angle": {"min": 0.0, "max": 6.28, "default": 0.0, "step": 0.1},
            "blur_samples": {"min": 3.0, "max": 20.0, "default": 10.0, "step": 1.0},
            "center_focus": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "radial_blur": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float blur_amount;
            uniform float blur_angle;
            uniform float blur_samples;
            uniform float center_focus;
            uniform float radial_blur;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = 1.0 / tex_size;
                int samples = int(blur_samples);

                // Blur direction
                vec2 dir;
                if(radial_blur > 0.5) {
                    // Radial blur from center
                    dir = normalize(v_uv - 0.5);
                } else {
                    // Directional blur
                    dir = vec2(cos(blur_angle), sin(blur_angle));
                }

                // Center focus falloff
                float dist_from_center = length(v_uv - 0.5) * 2.0;
                float blur_mult = mix(1.0, dist_from_center, center_focus);

                vec3 color = vec3(0.0);
                float total = 0.0;

                for(int i = -samples; i <= samples; i++) {
                    float t = float(i) / float(samples);
                    float weight = 1.0 - abs(t);
                    vec2 offset = dir * t * blur_amount * pixel * blur_mult;
                    color += texture(u_texture, v_uv + offset).rgb * weight;
                    total += weight;
                }

                f_color = vec4(color / total, 1.0);
            }
        """
    },
    "PBR (Physically Based)": {
        "uniforms": {
            "metallic": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "roughness": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "ao_strength": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "light_intensity": {"type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
            "environment_strength": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "fresnel_power": {"type": "float", "default": 5.0, "min": 1.0, "max": 10.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float metallic;
            uniform float roughness;
            uniform float ao_strength;
            uniform float light_intensity;
            uniform float environment_strength;
            uniform float fresnel_power;
            in vec2 v_uv;
            out vec4 f_color;

            const float PI = 3.14159265359;

            // GGX/Trowbridge-Reitz normal distribution
            float DistributionGGX(vec3 N, vec3 H, float rough) {
                float a = rough * rough;
                float a2 = a * a;
                float NdotH = max(dot(N, H), 0.0);
                float NdotH2 = NdotH * NdotH;
                float nom = a2;
                float denom = (NdotH2 * (a2 - 1.0) + 1.0);
                denom = PI * denom * denom;
                return nom / denom;
            }

            // Schlick-GGX geometry function
            float GeometrySchlickGGX(float NdotV, float rough) {
                float r = (rough + 1.0);
                float k = (r * r) / 8.0;
                float nom = NdotV;
                float denom = NdotV * (1.0 - k) + k;
                return nom / denom;
            }

            float GeometrySmith(vec3 N, vec3 V, vec3 L, float rough) {
                float NdotV = max(dot(N, V), 0.0);
                float NdotL = max(dot(N, L), 0.0);
                float ggx2 = GeometrySchlickGGX(NdotV, rough);
                float ggx1 = GeometrySchlickGGX(NdotL, rough);
                return ggx1 * ggx2;
            }

            // Fresnel-Schlick approximation
            vec3 fresnelSchlick(float cosTheta, vec3 F0) {
                return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), fresnel_power);
            }

            void main() {
                vec3 albedo = texture(u_texture, v_uv).rgb;

                // Simulate normal from texture gradient
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));
                float hL = dot(texture(u_texture, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hR = dot(texture(u_texture, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hD = dot(texture(u_texture, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float hU = dot(texture(u_texture, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                vec3 N = normalize(vec3(hL - hR, hD - hU, 1.0));

                vec3 V = vec3(0.0, 0.0, 1.0);
                vec3 lightPos = vec3(1.0, 1.0, 2.0);
                vec3 L = normalize(lightPos);
                vec3 H = normalize(V + L);

                vec3 F0 = vec3(0.04);
                F0 = mix(F0, albedo, metallic);

                // Cook-Torrance BRDF
                float NDF = DistributionGGX(N, H, roughness);
                float G = GeometrySmith(N, V, L, roughness);
                vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

                vec3 kS = F;
                vec3 kD = vec3(1.0) - kS;
                kD *= 1.0 - metallic;

                vec3 numerator = NDF * G * F;
                float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                vec3 specular = numerator / denominator;

                float NdotL = max(dot(N, L), 0.0);
                vec3 Lo = (kD * albedo / PI + specular) * vec3(1.0) * NdotL * light_intensity;

                // Ambient/environment
                vec3 ambient = environment_strength * albedo;

                // Simple AO from luminance variance
                float ao = mix(1.0, dot(albedo, vec3(0.299, 0.587, 0.114)), ao_strength * 0.5);

                vec3 color = ambient + Lo * ao;

                // HDR tonemapping
                color = color / (color + vec3(1.0));
                // Gamma correction
                color = pow(color, vec3(1.0/2.2));

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Lambert (Matte Diffuse)": {
        "uniforms": {
            "light_x": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0},
            "light_y": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0},
            "light_z": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0},
            "light_color_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "light_color_g": {"type": "float", "default": 0.95, "min": 0.0, "max": 1.0},
            "light_color_b": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0},
            "ambient": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
            "wrap": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float light_x;
            uniform float light_y;
            uniform float light_z;
            uniform float light_color_r;
            uniform float light_color_g;
            uniform float light_color_b;
            uniform float ambient;
            uniform float wrap;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 albedo = texture(u_texture, v_uv).rgb;

                // Generate normal from texture
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));
                float hL = dot(texture(u_texture, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hR = dot(texture(u_texture, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hD = dot(texture(u_texture, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float hU = dot(texture(u_texture, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                vec3 N = normalize(vec3(hL - hR, hD - hU, 1.0));

                vec3 L = normalize(vec3(light_x, light_y, light_z));
                vec3 lightColor = vec3(light_color_r, light_color_g, light_color_b);

                // Lambert diffuse with optional wrap lighting
                float NdotL = dot(N, L);
                float diffuse = (NdotL + wrap) / (1.0 + wrap);
                diffuse = max(diffuse, 0.0);

                vec3 color = albedo * (ambient + diffuse * lightColor);

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Blinn-Phong (Specular)": {
        "uniforms": {
            "light_x": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0},
            "light_y": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0},
            "light_z": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0},
            "ambient": {"type": "float", "default": 0.15, "min": 0.0, "max": 1.0},
            "diffuse_strength": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0},
            "specular_strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 2.0},
            "shininess": {"type": "float", "default": 32.0, "min": 1.0, "max": 256.0},
            "specular_color_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "specular_color_g": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "specular_color_b": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float light_x;
            uniform float light_y;
            uniform float light_z;
            uniform float ambient;
            uniform float diffuse_strength;
            uniform float specular_strength;
            uniform float shininess;
            uniform float specular_color_r;
            uniform float specular_color_g;
            uniform float specular_color_b;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 albedo = texture(u_texture, v_uv).rgb;

                // Generate normal from texture gradient
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));
                float hL = dot(texture(u_texture, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hR = dot(texture(u_texture, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hD = dot(texture(u_texture, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float hU = dot(texture(u_texture, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                vec3 N = normalize(vec3(hL - hR, hD - hU, 1.0));

                vec3 L = normalize(vec3(light_x, light_y, light_z));
                vec3 V = vec3(0.0, 0.0, 1.0);
                vec3 H = normalize(L + V);  // Halfway vector (Blinn)

                // Diffuse
                float diff = max(dot(N, L), 0.0);

                // Specular (Blinn-Phong)
                float spec = pow(max(dot(N, H), 0.0), shininess);

                vec3 specColor = vec3(specular_color_r, specular_color_g, specular_color_b);
                vec3 color = albedo * ambient +
                             albedo * diffuse_strength * diff +
                             specColor * specular_strength * spec;

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Anisotropic (Brushed Metal)": {
        "uniforms": {
            "anisotropy": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0},
            "roughness_x": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0},
            "roughness_y": {"type": "float", "default": 0.5, "min": 0.01, "max": 1.0},
            "rotation": {"type": "float", "default": 0.0, "min": 0.0, "max": 6.28},
            "metallic": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0},
            "light_intensity": {"type": "float", "default": 1.5, "min": 0.0, "max": 3.0},
            "ambient": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float anisotropy;
            uniform float roughness_x;
            uniform float roughness_y;
            uniform float rotation;
            uniform float metallic;
            uniform float light_intensity;
            uniform float ambient;
            in vec2 v_uv;
            out vec4 f_color;

            const float PI = 3.14159265359;

            void main() {
                vec3 albedo = texture(u_texture, v_uv).rgb;

                // Generate normal from texture
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));
                float hL = dot(texture(u_texture, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hR = dot(texture(u_texture, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hD = dot(texture(u_texture, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float hU = dot(texture(u_texture, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                vec3 N = normalize(vec3(hL - hR, hD - hU, 1.0));

                // Tangent and bitangent with rotation
                float c = cos(rotation);
                float s = sin(rotation);
                vec3 T = normalize(vec3(c, s, 0.0));
                vec3 B = normalize(vec3(-s, c, 0.0));

                vec3 L = normalize(vec3(1.0, 1.0, 1.5));
                vec3 V = vec3(0.0, 0.0, 1.0);
                vec3 H = normalize(L + V);

                // Anisotropic GGX
                float HdotT = dot(H, T);
                float HdotB = dot(H, B);
                float HdotN = dot(H, N);

                float ax = roughness_x;
                float ay = roughness_y;

                float denom = (HdotT * HdotT) / (ax * ax) +
                              (HdotB * HdotB) / (ay * ay) +
                              HdotN * HdotN;
                float D = 1.0 / (PI * ax * ay * denom * denom);

                // Simplified anisotropic specular
                float NdotL = max(dot(N, L), 0.0);
                float NdotV = max(dot(N, V), 0.0);
                float spec = D * NdotL / (4.0 * NdotV + 0.001);
                spec = mix(spec, spec * anisotropy, anisotropy);

                // Metal reflects its own color
                vec3 specColor = mix(vec3(1.0), albedo, metallic);

                vec3 color = albedo * ambient +
                             albedo * (1.0 - metallic) * NdotL * 0.5 +
                             specColor * spec * light_intensity;

                // Tone mapping
                color = color / (color + vec3(1.0));

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Subsurface Scattering": {
        "uniforms": {
            "scatter_radius": {"type": "float", "default": 10.0, "min": 1.0, "max": 50.0},
            "scatter_strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "scatter_color_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "scatter_color_g": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0},
            "scatter_color_b": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "translucency": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "light_wrap": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "ambient": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.5},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float scatter_radius;
            uniform float scatter_strength;
            uniform float scatter_color_r;
            uniform float scatter_color_g;
            uniform float scatter_color_b;
            uniform float translucency;
            uniform float light_wrap;
            uniform float ambient;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 albedo = texture(u_texture, v_uv).rgb;
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                // Gaussian blur for scattering simulation
                vec3 scattered = vec3(0.0);
                float total = 0.0;
                int samples = 8;

                for(int x = -samples; x <= samples; x++) {
                    for(int y = -samples; y <= samples; y++) {
                        float dist = length(vec2(x, y));
                        if(dist <= float(samples)) {
                            float weight = exp(-dist * dist / (scatter_radius * 0.5));
                            vec2 offset = vec2(x, y) * texel * scatter_radius * 0.2;
                            scattered += texture(u_texture, v_uv + offset).rgb * weight;
                            total += weight;
                        }
                    }
                }
                scattered /= total;

                // Scatter color tinting
                vec3 scatterColor = vec3(scatter_color_r, scatter_color_g, scatter_color_b);
                scattered *= scatterColor;

                // Generate normal for lighting
                float hL = dot(texture(u_texture, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hR = dot(texture(u_texture, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hD = dot(texture(u_texture, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float hU = dot(texture(u_texture, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                vec3 N = normalize(vec3(hL - hR, hD - hU, 1.0));

                vec3 L = normalize(vec3(1.0, 1.0, 1.0));

                // Wrapped diffuse lighting
                float NdotL = dot(N, L);
                float diffuse = (NdotL + light_wrap) / (1.0 + light_wrap);
                diffuse = max(diffuse, 0.0);

                // Translucency (light from behind)
                float backlight = max(-NdotL, 0.0) * translucency;
                vec3 backlightColor = albedo * scatterColor * backlight;

                // Combine
                vec3 color = albedo * ambient +
                             albedo * diffuse * (1.0 - scatter_strength) +
                             scattered * scatter_strength +
                             backlightColor;

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Dithering (Retro)": {
        "uniforms": {
            "dither_size": {"type": "float", "default": 4.0, "min": 1.0, "max": 16.0},
            "color_levels": {"type": "float", "default": 4.0, "min": 2.0, "max": 16.0},
            "pattern_type": {"type": "float", "default": 0.0, "min": 0.0, "max": 3.0},
            "contrast": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0},
            "monochrome": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float dither_size;
            uniform float color_levels;
            uniform float pattern_type;
            uniform float contrast;
            uniform float monochrome;
            in vec2 v_uv;
            out vec4 f_color;

            // 8x8 Bayer matrix
            float bayer8x8(vec2 pos) {
                int x = int(mod(pos.x, 8.0));
                int y = int(mod(pos.y, 8.0));

                int bayer[64] = int[64](
                     0, 32,  8, 40,  2, 34, 10, 42,
                    48, 16, 56, 24, 50, 18, 58, 26,
                    12, 44,  4, 36, 14, 46,  6, 38,
                    60, 28, 52, 20, 62, 30, 54, 22,
                     3, 35, 11, 43,  1, 33,  9, 41,
                    51, 19, 59, 27, 49, 17, 57, 25,
                    15, 47,  7, 39, 13, 45,  5, 37,
                    63, 31, 55, 23, 61, 29, 53, 21
                );

                return float(bayer[y * 8 + x]) / 64.0;
            }

            // Ordered dithering 4x4
            float ordered4x4(vec2 pos) {
                int x = int(mod(pos.x, 4.0));
                int y = int(mod(pos.y, 4.0));
                int ordered[16] = int[16](
                    0, 8, 2, 10,
                    12, 4, 14, 6,
                    3, 11, 1, 9,
                    15, 7, 13, 5
                );
                return float(ordered[y * 4 + x]) / 16.0;
            }

            // Blue noise approximation
            float blueNoise(vec2 pos) {
                return fract(sin(dot(pos, vec2(12.9898, 78.233))) * 43758.5453);
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel_pos = v_uv * tex_size / dither_size;

                // Apply contrast
                color = (color - 0.5) * contrast + 0.5;

                // Convert to monochrome if enabled
                if(monochrome > 0.5) {
                    float lum = dot(color, vec3(0.299, 0.587, 0.114));
                    color = vec3(lum);
                }

                // Get dither threshold based on pattern
                float threshold;
                int ptype = int(pattern_type);
                if(ptype == 0) {
                    threshold = bayer8x8(pixel_pos);
                } else if(ptype == 1) {
                    threshold = ordered4x4(pixel_pos);
                } else if(ptype == 2) {
                    threshold = blueNoise(pixel_pos);
                } else {
                    // Halftone pattern
                    vec2 center = floor(pixel_pos) + 0.5;
                    float dist = length(fract(pixel_pos) - 0.5);
                    threshold = dist;
                }

                // Quantize with dithering
                float levels = color_levels - 1.0;
                vec3 quantized;
                for(int i = 0; i < 3; i++) {
                    float c = color[i];
                    float lower = floor(c * levels) / levels;
                    float upper = ceil(c * levels) / levels;
                    float t = fract(c * levels);
                    quantized[i] = t > threshold ? upper : lower;
                }

                f_color = vec4(quantized, 1.0);
            }
        """
    },
    "VHS Tape": {
        "uniforms": {
            "scan_intensity": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "noise_amount": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.5},
            "color_bleed": {"type": "float", "default": 3.0, "min": 0.0, "max": 10.0},
            "tracking_error": {"type": "float", "default": 0.02, "min": 0.0, "max": 0.1},
            "tape_crease": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "head_switching": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.3},
            "saturation_loss": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "time": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float scan_intensity;
            uniform float noise_amount;
            uniform float color_bleed;
            uniform float tracking_error;
            uniform float tape_crease;
            uniform float head_switching;
            uniform float saturation_loss;
            uniform float time;
            in vec2 v_uv;
            out vec4 f_color;

            float rand(vec2 co) {
                return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
            }

            void main() {
                vec2 uv = v_uv;
                vec2 tex_size = vec2(textureSize(u_texture, 0));

                // Tracking error wobble
                float wobble = sin(uv.y * 50.0 + time * 5.0) * tracking_error;
                wobble += rand(vec2(floor(uv.y * 100.0), time)) * tracking_error * 0.5;
                uv.x += wobble;

                // Head switching noise at bottom
                if(uv.y < head_switching) {
                    float switch_noise = rand(vec2(uv.x * 10.0, time)) * 0.1;
                    uv.x += switch_noise;
                    uv.y += rand(vec2(uv.y, time)) * 0.02;
                }

                // Tape crease effect
                float crease_pos = fract(time * 0.1) * 1.2 - 0.1;
                if(abs(uv.y - crease_pos) < 0.02 * tape_crease) {
                    float crease = 1.0 - abs(uv.y - crease_pos) / (0.02 * tape_crease);
                    uv.x += crease * 0.05 * tape_crease;
                }

                // Color channel separation (YIQ simulation)
                vec2 pixel = 1.0 / tex_size;
                float r = texture(u_texture, uv + vec2(color_bleed * pixel.x, 0.0)).r;
                float g = texture(u_texture, uv).g;
                float b = texture(u_texture, uv - vec2(color_bleed * pixel.x, 0.0)).b;
                vec3 color = vec3(r, g, b);

                // Reduce saturation (VHS color loss)
                float lum = dot(color, vec3(0.299, 0.587, 0.114));
                color = mix(color, vec3(lum), saturation_loss);

                // Scanlines
                float scanline = sin(uv.y * tex_size.y * 3.14159) * 0.5 + 0.5;
                scanline = pow(scanline, 1.5);
                color *= 1.0 - scan_intensity * (1.0 - scanline);

                // Tape noise
                float noise = rand(vec2(uv.x * 100.0, uv.y * 100.0 + time * 100.0));
                color += (noise - 0.5) * noise_amount;

                // Occasional horizontal noise lines
                if(rand(vec2(floor(uv.y * 300.0), time)) > 0.995) {
                    color += vec3(0.3) * rand(vec2(uv.x, time));
                }

                // Slight color shift to yellowed look
                color.r *= 1.05;
                color.b *= 0.95;

                // Clamp
                color = clamp(color, 0.0, 1.0);

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Perlin Noise": {
        "uniforms": {
            "noise_scale": {"type": "float", "default": 5.0, "min": 0.5, "max": 20.0},
            "octaves": {"type": "float", "default": 4.0, "min": 1.0, "max": 8.0},
            "persistence": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "lacunarity": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
            "blend_mode": {"type": "float", "default": 0.0, "min": 0.0, "max": 3.0},
            "blend_amount": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "color1_r": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "color1_g": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "color1_b": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "color2_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "color2_g": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "color2_b": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "time": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0},
            "animate": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float noise_scale;
            uniform float octaves;
            uniform float persistence;
            uniform float lacunarity;
            uniform float blend_mode;
            uniform float blend_amount;
            uniform float color1_r;
            uniform float color1_g;
            uniform float color1_b;
            uniform float color2_r;
            uniform float color2_g;
            uniform float color2_b;
            uniform float time;
            uniform float animate;
            in vec2 v_uv;
            out vec4 f_color;

            // Permutation table
            vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
            vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

            // 3D Perlin noise
            float snoise(vec3 v) {
                const vec2 C = vec2(1.0/6.0, 1.0/3.0);
                const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

                vec3 i  = floor(v + dot(v, C.yyy));
                vec3 x0 = v - i + dot(i, C.xxx);

                vec3 g = step(x0.yzx, x0.xyz);
                vec3 l = 1.0 - g;
                vec3 i1 = min(g.xyz, l.zxy);
                vec3 i2 = max(g.xyz, l.zxy);

                vec3 x1 = x0 - i1 + C.xxx;
                vec3 x2 = x0 - i2 + C.yyy;
                vec3 x3 = x0 - D.yyy;

                i = mod289(i);
                vec4 p = permute(permute(permute(
                    i.z + vec4(0.0, i1.z, i2.z, 1.0))
                    + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                    + i.x + vec4(0.0, i1.x, i2.x, 1.0));

                float n_ = 0.142857142857;
                vec3 ns = n_ * D.wyz - D.xzx;

                vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

                vec4 x_ = floor(j * ns.z);
                vec4 y_ = floor(j - 7.0 * x_);

                vec4 x = x_ *ns.x + ns.yyyy;
                vec4 y = y_ *ns.x + ns.yyyy;
                vec4 h = 1.0 - abs(x) - abs(y);

                vec4 b0 = vec4(x.xy, y.xy);
                vec4 b1 = vec4(x.zw, y.zw);

                vec4 s0 = floor(b0)*2.0 + 1.0;
                vec4 s1 = floor(b1)*2.0 + 1.0;
                vec4 sh = -step(h, vec4(0.0));

                vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
                vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;

                vec3 p0 = vec3(a0.xy, h.x);
                vec3 p1 = vec3(a0.zw, h.y);
                vec3 p2 = vec3(a1.xy, h.z);
                vec3 p3 = vec3(a1.zw, h.w);

                vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
                p0 *= norm.x;
                p1 *= norm.y;
                p2 *= norm.z;
                p3 *= norm.w;

                vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
                m = m * m;
                return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
            }

            // FBM (Fractal Brownian Motion)
            float fbm(vec3 p) {
                float value = 0.0;
                float amplitude = 1.0;
                float frequency = 1.0;
                float total_amp = 0.0;

                int oct = int(octaves);
                for(int i = 0; i < oct; i++) {
                    value += amplitude * snoise(p * frequency);
                    total_amp += amplitude;
                    amplitude *= persistence;
                    frequency *= lacunarity;
                }

                return value / total_amp;
            }

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;

                // Generate noise
                float t = animate > 0.5 ? time * 0.5 : 0.0;
                vec3 p = vec3(v_uv * noise_scale, t);
                float n = fbm(p);
                n = n * 0.5 + 0.5; // Normalize to 0-1

                // Create gradient colors
                vec3 color1 = vec3(color1_r, color1_g, color1_b);
                vec3 color2 = vec3(color2_r, color2_g, color2_b);
                vec3 noise_color = mix(color1, color2, n);

                // Blend with texture
                vec3 result;
                int mode = int(blend_mode);
                if(mode == 0) {
                    // Mix
                    result = mix(tex_color, noise_color, blend_amount);
                } else if(mode == 1) {
                    // Multiply
                    result = mix(tex_color, tex_color * noise_color, blend_amount);
                } else if(mode == 2) {
                    // Add
                    result = tex_color + noise_color * blend_amount;
                } else {
                    // Overlay
                    vec3 overlay = vec3(
                        tex_color.r < 0.5 ? 2.0 * tex_color.r * noise_color.r : 1.0 - 2.0 * (1.0 - tex_color.r) * (1.0 - noise_color.r),
                        tex_color.g < 0.5 ? 2.0 * tex_color.g * noise_color.g : 1.0 - 2.0 * (1.0 - tex_color.g) * (1.0 - noise_color.g),
                        tex_color.b < 0.5 ? 2.0 * tex_color.b * noise_color.b : 1.0 - 2.0 * (1.0 - tex_color.b) * (1.0 - noise_color.b)
                    );
                    result = mix(tex_color, overlay, blend_amount);
                }

                f_color = vec4(clamp(result, 0.0, 1.0), 1.0);
            }
        """
    },
    "Water/Waves": {
        "uniforms": {
            "wave_scale": {"type": "float", "default": 20.0, "min": 1.0, "max": 100.0},
            "wave_speed": {"type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
            "wave_height": {"type": "float", "default": 0.02, "min": 0.0, "max": 0.1},
            "ripple_freq": {"type": "float", "default": 10.0, "min": 1.0, "max": 50.0},
            "refraction": {"type": "float", "default": 0.03, "min": 0.0, "max": 0.1},
            "fresnel_power": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0},
            "water_color_r": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "water_color_g": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "water_color_b": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "water_tint": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
            "caustics": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "time": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float wave_scale;
            uniform float wave_speed;
            uniform float wave_height;
            uniform float ripple_freq;
            uniform float refraction;
            uniform float fresnel_power;
            uniform float water_color_r;
            uniform float water_color_g;
            uniform float water_color_b;
            uniform float water_tint;
            uniform float caustics;
            uniform float time;
            in vec2 v_uv;
            out vec4 f_color;

            float wave(vec2 p, float t) {
                float w1 = sin(p.x * wave_scale + t * wave_speed) * 0.5;
                float w2 = sin(p.y * wave_scale * 0.8 + t * wave_speed * 1.3) * 0.5;
                float w3 = sin((p.x + p.y) * wave_scale * 0.5 + t * wave_speed * 0.7) * 0.3;
                float w4 = sin(length(p - 0.5) * ripple_freq - t * wave_speed * 2.0) * 0.2;
                return (w1 + w2 + w3 + w4) * wave_height;
            }

            void main() {
                vec2 uv = v_uv;

                // Calculate wave normal
                float eps = 0.01;
                float h = wave(uv, time);
                float hx = wave(uv + vec2(eps, 0.0), time);
                float hy = wave(uv + vec2(0.0, eps), time);

                vec3 normal = normalize(vec3(
                    (h - hx) / eps,
                    (h - hy) / eps,
                    1.0
                ));

                // Refraction offset
                vec2 refract_offset = normal.xy * refraction;
                vec2 refracted_uv = uv + refract_offset;
                refracted_uv = clamp(refracted_uv, 0.0, 1.0);

                vec3 tex_color = texture(u_texture, refracted_uv).rgb;

                // Water color tint
                vec3 waterColor = vec3(water_color_r, water_color_g, water_color_b);
                tex_color = mix(tex_color, tex_color * waterColor + waterColor * 0.1, water_tint);

                // Fresnel effect
                vec3 view_dir = vec3(0.0, 0.0, 1.0);
                float fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), fresnel_power);

                // Specular highlight
                vec3 light_dir = normalize(vec3(1.0, 1.0, 2.0));
                vec3 half_vec = normalize(light_dir + view_dir);
                float spec = pow(max(dot(normal, half_vec), 0.0), 64.0);

                // Caustics pattern
                float caustic = 0.0;
                if(caustics > 0.0) {
                    vec2 cp = uv * 20.0 + time * 0.5;
                    caustic = sin(cp.x) * sin(cp.y);
                    caustic += sin(cp.x * 1.5 + cp.y * 0.5) * sin(cp.y * 1.3);
                    caustic = pow(abs(caustic) * 0.5, 2.0) * caustics;
                }

                vec3 color = tex_color;
                color += vec3(1.0) * spec * 0.5;
                color += vec3(0.8, 0.9, 1.0) * fresnel * 0.3;
                color += vec3(0.5, 0.7, 0.9) * caustic;

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Volumetric Fog": {
        "uniforms": {
            "fog_density": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "fog_start": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "fog_end": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "fog_color_r": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0},
            "fog_color_g": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0},
            "fog_color_b": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0},
            "noise_scale": {"type": "float", "default": 3.0, "min": 0.5, "max": 10.0},
            "noise_amount": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "light_scatter": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "light_x": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "light_y": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "time": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float fog_density;
            uniform float fog_start;
            uniform float fog_end;
            uniform float fog_color_r;
            uniform float fog_color_g;
            uniform float fog_color_b;
            uniform float noise_scale;
            uniform float noise_amount;
            uniform float light_scatter;
            uniform float light_x;
            uniform float light_y;
            uniform float time;
            in vec2 v_uv;
            out vec4 f_color;

            // Simple noise
            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);

                float a = hash(i);
                float b = hash(i + vec2(1.0, 0.0));
                float c = hash(i + vec2(0.0, 1.0));
                float d = hash(i + vec2(1.0, 1.0));

                return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
            }

            float fbm(vec2 p) {
                float value = 0.0;
                float amplitude = 0.5;
                for(int i = 0; i < 4; i++) {
                    value += amplitude * noise(p);
                    p *= 2.0;
                    amplitude *= 0.5;
                }
                return value;
            }

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;

                // Simulate depth from luminance (brighter = closer)
                float lum = dot(tex_color, vec3(0.299, 0.587, 0.114));
                float depth = 1.0 - lum;
                depth = smoothstep(fog_start, fog_end, depth);

                // Add volumetric noise
                vec2 noise_uv = v_uv * noise_scale + vec2(time * 0.1, time * 0.05);
                float fog_noise = fbm(noise_uv);
                fog_noise = fog_noise * noise_amount;

                // Calculate fog factor
                float fog_factor = depth * fog_density + fog_noise;
                fog_factor = clamp(fog_factor, 0.0, 1.0);

                vec3 fogColor = vec3(fog_color_r, fog_color_g, fog_color_b);

                // Light scattering (god rays approximation)
                vec2 light_pos = vec2(light_x, light_y);
                vec2 delta = v_uv - light_pos;
                float dist_to_light = length(delta);

                float scatter = 0.0;
                if(light_scatter > 0.0) {
                    int samples = 16;
                    vec2 step_size = delta / float(samples);
                    for(int i = 0; i < samples; i++) {
                        vec2 sample_pos = light_pos + step_size * float(i);
                        if(sample_pos.x >= 0.0 && sample_pos.x <= 1.0 &&
                           sample_pos.y >= 0.0 && sample_pos.y <= 1.0) {
                            float sample_lum = dot(texture(u_texture, sample_pos).rgb, vec3(0.299, 0.587, 0.114));
                            scatter += sample_lum;
                        }
                    }
                    scatter /= float(samples);
                    scatter *= light_scatter * (1.0 - dist_to_light);
                    scatter = max(scatter, 0.0);
                }

                // Apply fog
                vec3 color = mix(tex_color, fogColor, fog_factor);
                color += fogColor * scatter;

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Parallax/Normal Map": {
        "uniforms": {
            "height_scale": {"type": "float", "default": 0.05, "min": 0.0, "max": 0.2},
            "parallax_layers": {"type": "float", "default": 16.0, "min": 4.0, "max": 64.0},
            "normal_strength": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0},
            "light_x": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0},
            "light_y": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0},
            "light_z": {"type": "float", "default": 1.0, "min": 0.1, "max": 2.0},
            "view_angle_x": {"type": "float", "default": 0.0, "min": -0.5, "max": 0.5},
            "view_angle_y": {"type": "float", "default": 0.0, "min": -0.5, "max": 0.5},
            "ambient": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.5},
            "specular": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "shininess": {"type": "float", "default": 32.0, "min": 1.0, "max": 128.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float height_scale;
            uniform float parallax_layers;
            uniform float normal_strength;
            uniform float light_x;
            uniform float light_y;
            uniform float light_z;
            uniform float view_angle_x;
            uniform float view_angle_y;
            uniform float ambient;
            uniform float specular;
            uniform float shininess;
            in vec2 v_uv;
            out vec4 f_color;

            float getHeight(vec2 uv) {
                return dot(texture(u_texture, uv).rgb, vec3(0.299, 0.587, 0.114));
            }

            vec2 parallaxMapping(vec2 uv, vec3 viewDir) {
                float numLayers = parallax_layers;
                float layerDepth = 1.0 / numLayers;
                float currentLayerDepth = 0.0;

                vec2 P = viewDir.xy * height_scale;
                vec2 deltaTexCoords = P / numLayers;

                vec2 currentTexCoords = uv;
                float currentDepthMapValue = 1.0 - getHeight(currentTexCoords);

                for(int i = 0; i < int(numLayers); i++) {
                    if(currentLayerDepth >= currentDepthMapValue) break;
                    currentTexCoords -= deltaTexCoords;
                    currentDepthMapValue = 1.0 - getHeight(currentTexCoords);
                    currentLayerDepth += layerDepth;
                }

                // Parallax occlusion mapping interpolation
                vec2 prevTexCoords = currentTexCoords + deltaTexCoords;
                float afterDepth = currentDepthMapValue - currentLayerDepth;
                float beforeDepth = (1.0 - getHeight(prevTexCoords)) - currentLayerDepth + layerDepth;
                float weight = afterDepth / (afterDepth - beforeDepth);

                return mix(currentTexCoords, prevTexCoords, weight);
            }

            vec3 getNormal(vec2 uv) {
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                float hL = getHeight(uv - vec2(texel.x, 0.0));
                float hR = getHeight(uv + vec2(texel.x, 0.0));
                float hD = getHeight(uv - vec2(0.0, texel.y));
                float hU = getHeight(uv + vec2(0.0, texel.y));

                vec3 normal = vec3(
                    (hL - hR) * normal_strength,
                    (hD - hU) * normal_strength,
                    1.0
                );

                return normalize(normal);
            }

            void main() {
                vec3 viewDir = normalize(vec3(view_angle_x, view_angle_y, 1.0));

                // Apply parallax mapping
                vec2 parallax_uv = parallaxMapping(v_uv, viewDir);

                // Clamp to valid range
                if(parallax_uv.x < 0.0 || parallax_uv.x > 1.0 ||
                   parallax_uv.y < 0.0 || parallax_uv.y > 1.0) {
                    parallax_uv = v_uv;
                }

                vec3 albedo = texture(u_texture, parallax_uv).rgb;
                vec3 normal = getNormal(parallax_uv);

                vec3 lightDir = normalize(vec3(light_x, light_y, light_z));
                vec3 halfDir = normalize(lightDir + viewDir);

                // Diffuse
                float diff = max(dot(normal, lightDir), 0.0);

                // Specular
                float spec = pow(max(dot(normal, halfDir), 0.0), shininess);

                vec3 color = albedo * ambient +
                             albedo * diff * 0.8 +
                             vec3(1.0) * spec * specular;

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Inverted Hull Outline": {
        "description": "Classic inverted hull technique for cartoon/anime-style outlines. Simulates the effect of rendering a backface-expanded mesh for clean edge detection.",
        "category": "Stylized",
        "uniforms": {
            "outline_width": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1},
            "outline_color_r": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
            "outline_color_g": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
            "outline_color_b": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
            "depth_threshold": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01},
            "normal_threshold": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
            "depth_influence": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1},
            "normal_influence": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1},
            "inner_outline": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1},
            "outline_falloff": {"type": "float", "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1},
            "color_blend": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1},
            "silhouette_only": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float outline_width;
            uniform float outline_color_r;
            uniform float outline_color_g;
            uniform float outline_color_b;
            uniform float depth_threshold;
            uniform float normal_threshold;
            uniform float depth_influence;
            uniform float normal_influence;
            uniform float inner_outline;
            uniform float outline_falloff;
            uniform float color_blend;
            uniform float silhouette_only;
            in vec2 v_uv;
            out vec4 f_color;

            // Estimate depth from luminance (brighter = closer in typical scenes)
            float getDepth(vec2 uv) {
                vec3 col = texture(u_texture, uv).rgb;
                return dot(col, vec3(0.299, 0.587, 0.114));
            }

            // Estimate normal from depth gradient
            vec3 getNormal(vec2 uv, vec2 texel) {
                float d = getDepth(uv);
                float dL = getDepth(uv - vec2(texel.x, 0.0));
                float dR = getDepth(uv + vec2(texel.x, 0.0));
                float dD = getDepth(uv - vec2(0.0, texel.y));
                float dU = getDepth(uv + vec2(0.0, texel.y));

                vec3 normal = vec3(
                    (dL - dR) * 2.0,
                    (dD - dU) * 2.0,
                    1.0
                );
                return normalize(normal);
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 texel = 1.0 / tex_size;
                vec2 offset = texel * outline_width;

                vec4 center_color = texture(u_texture, v_uv);
                float center_depth = getDepth(v_uv);
                vec3 center_normal = getNormal(v_uv, texel);

                // Sample in 8 directions for hull detection
                vec2 offsets[8] = vec2[8](
                    vec2(-1.0, -1.0), vec2(0.0, -1.0), vec2(1.0, -1.0),
                    vec2(-1.0,  0.0),                  vec2(1.0,  0.0),
                    vec2(-1.0,  1.0), vec2(0.0,  1.0), vec2(1.0,  1.0)
                );

                float depth_edge = 0.0;
                float normal_edge = 0.0;
                float max_depth_diff = 0.0;
                float max_normal_diff = 0.0;

                // Inverted hull detection - find where normals face away (backface simulation)
                for(int i = 0; i < 8; i++) {
                    vec2 sample_uv = v_uv + offsets[i] * offset;
                    sample_uv = clamp(sample_uv, vec2(0.0), vec2(1.0));

                    float sample_depth = getDepth(sample_uv);
                    vec3 sample_normal = getNormal(sample_uv, texel);

                    // Depth discontinuity detection
                    float depth_diff = abs(center_depth - sample_depth);
                    max_depth_diff = max(max_depth_diff, depth_diff);

                    // Normal discontinuity (simulates hull inversion)
                    float normal_diff = 1.0 - max(dot(center_normal, sample_normal), 0.0);
                    max_normal_diff = max(max_normal_diff, normal_diff);

                    // Backface detection simulation
                    vec3 view_dir = vec3(0.0, 0.0, 1.0);
                    float facing = dot(sample_normal, view_dir);
                    if(facing < 0.0) {
                        normal_edge += abs(facing) * 0.5;
                    }
                }

                // Apply thresholds
                depth_edge = smoothstep(depth_threshold * 0.5, depth_threshold, max_depth_diff);
                normal_edge = smoothstep(normal_threshold * 0.5, normal_threshold, max_normal_diff);

                // Combine edges with influence weights
                float edge = depth_edge * depth_influence + normal_edge * normal_influence;
                edge = clamp(edge, 0.0, 1.0);

                // Apply falloff for softer/harder edges
                edge = pow(edge, outline_falloff);

                // Inner outline detection (creases and folds)
                float inner_edge = 0.0;
                if(inner_outline > 0.0) {
                    // Sobel-like operator for inner details
                    float sobelX = 0.0;
                    float sobelY = 0.0;

                    // 3x3 Sobel kernels
                    float kernelX[9] = float[9](
                        -1.0, 0.0, 1.0,
                        -2.0, 0.0, 2.0,
                        -1.0, 0.0, 1.0
                    );
                    float kernelY[9] = float[9](
                        -1.0, -2.0, -1.0,
                         0.0,  0.0,  0.0,
                         1.0,  2.0,  1.0
                    );

                    int idx = 0;
                    for(int y = -1; y <= 1; y++) {
                        for(int x = -1; x <= 1; x++) {
                            vec2 sample_uv = v_uv + vec2(float(x), float(y)) * texel * outline_width * 0.5;
                            float lum = getDepth(sample_uv);
                            sobelX += lum * kernelX[idx];
                            sobelY += lum * kernelY[idx];
                            idx++;
                        }
                    }

                    inner_edge = sqrt(sobelX * sobelX + sobelY * sobelY);
                    inner_edge = smoothstep(0.05, 0.2, inner_edge) * inner_outline;
                }

                // Combine outer and inner edges
                float final_edge = max(edge, inner_edge);

                // Silhouette-only mode
                if(silhouette_only > 0.5) {
                    // Only show edges at depth discontinuities (true silhouette)
                    final_edge = depth_edge * depth_influence;
                    final_edge = pow(final_edge, outline_falloff);
                }

                // Outline color
                vec3 outline_color = vec3(outline_color_r, outline_color_g, outline_color_b);

                // Optional: blend outline color with underlying color
                if(color_blend > 0.0) {
                    vec3 dark_color = center_color.rgb * 0.3;
                    outline_color = mix(outline_color, dark_color, color_blend);
                }

                // Final composition
                vec3 final_color = mix(center_color.rgb, outline_color, final_edge);

                f_color = vec4(final_color, 1.0);
            }
        """
    },
    # ==================== TEXTURE/PATTERN GENERATION ====================
    "Voronoi Texture": {
        "description": "Cellular/organic noise pattern for stone, scales, and biological textures.",
        "category": "Procedural",
        "uniforms": {
            "scale": {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5},
            "randomness": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "feature": {"type": "float", "default": 0.0, "min": 0.0, "max": 3.0, "step": 1.0},
            "blend_mode": {"type": "float", "default": 0.0, "min": 0.0, "max": 3.0, "step": 1.0},
            "blend_amount": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "color1_r": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
            "color1_g": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
            "color1_b": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_g": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_b": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "time": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float scale;
            uniform float randomness;
            uniform float feature;
            uniform float blend_mode;
            uniform float blend_amount;
            uniform float color1_r, color1_g, color1_b;
            uniform float color2_r, color2_g, color2_b;
            uniform float time;
            in vec2 v_uv;
            out vec4 f_color;

            vec2 hash2(vec2 p) {
                p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
                return fract(sin(p) * 43758.5453);
            }

            vec3 voronoi(vec2 x, float r) {
                vec2 n = floor(x);
                vec2 f = fract(x);

                float dist1 = 8.0;
                float dist2 = 8.0;
                vec2 cell1 = vec2(0.0);

                for(int j = -1; j <= 1; j++) {
                    for(int i = -1; i <= 1; i++) {
                        vec2 g = vec2(float(i), float(j));
                        vec2 o = hash2(n + g) * r;
                        vec2 diff = g + o - f;
                        float d = dot(diff, diff);

                        if(d < dist1) {
                            dist2 = dist1;
                            dist1 = d;
                            cell1 = n + g;
                        } else if(d < dist2) {
                            dist2 = d;
                        }
                    }
                }

                return vec3(sqrt(dist1), sqrt(dist2), hash2(cell1).x);
            }

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;
                vec2 uv = v_uv * scale + time * 0.1;

                vec3 v = voronoi(uv, randomness);

                float value;
                int feat = int(feature);
                if(feat == 0) value = v.x;           // F1 - distance to closest
                else if(feat == 1) value = v.y;      // F2 - distance to second closest
                else if(feat == 2) value = v.y - v.x; // F2-F1 - cell edges
                else value = v.z;                     // Cell ID

                vec3 color1 = vec3(color1_r, color1_g, color1_b);
                vec3 color2 = vec3(color2_r, color2_g, color2_b);
                vec3 pattern = mix(color1, color2, value);

                vec3 result;
                int mode = int(blend_mode);
                if(mode == 0) result = mix(tex_color, pattern, blend_amount);
                else if(mode == 1) result = mix(tex_color, tex_color * pattern, blend_amount);
                else if(mode == 2) result = tex_color + pattern * blend_amount;
                else {
                    vec3 overlay = mix(2.0 * tex_color * pattern, 1.0 - 2.0 * (1.0 - tex_color) * (1.0 - pattern), step(0.5, tex_color));
                    result = mix(tex_color, overlay, blend_amount);
                }

                f_color = vec4(result, 1.0);
            }
        """
    },
    "Musgrave Texture": {
        "description": "Complex fractal noise for terrains, clouds, and organic surfaces.",
        "category": "Procedural",
        "uniforms": {
            "scale": {"type": "float", "default": 3.0, "min": 0.5, "max": 15.0, "step": 0.5},
            "detail": {"type": "float", "default": 4.0, "min": 1.0, "max": 10.0, "step": 1.0},
            "dimension": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
            "lacunarity": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1},
            "offset": {"type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1},
            "gain": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1},
            "type": {"type": "float", "default": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
            "blend_amount": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float scale;
            uniform float detail;
            uniform float dimension;
            uniform float lacunarity;
            uniform float offset;
            uniform float gain;
            uniform float type;
            uniform float blend_amount;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = hash(i);
                float b = hash(i + vec2(1.0, 0.0));
                float c = hash(i + vec2(0.0, 1.0));
                float d = hash(i + vec2(1.0, 1.0));
                return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
            }

            float musgrave_fBm(vec2 p, float octaves, float dim, float lac) {
                float value = 0.0;
                float amplitude = 1.0;
                float frequency = 1.0;
                for(int i = 0; i < int(octaves); i++) {
                    value += amplitude * (noise(p * frequency) - 0.5);
                    amplitude *= pow(lac, -dim);
                    frequency *= lac;
                }
                return value;
            }

            float musgrave_ridged(vec2 p, float octaves, float dim, float lac, float off, float g) {
                float value = 0.0;
                float weight = 1.0;
                float amplitude = 1.0;
                float frequency = 1.0;
                for(int i = 0; i < int(octaves); i++) {
                    float signal = off - abs(noise(p * frequency) - 0.5);
                    signal *= signal * weight;
                    weight = clamp(signal * g, 0.0, 1.0);
                    value += amplitude * signal;
                    amplitude *= pow(lac, -dim);
                    frequency *= lac;
                }
                return value;
            }

            float musgrave_hybrid(vec2 p, float octaves, float dim, float lac, float off) {
                float value = (noise(p) - 0.5 + off);
                float weight = value;
                float amplitude = 1.0;
                float frequency = lac;
                for(int i = 1; i < int(octaves); i++) {
                    amplitude *= pow(lac, -dim);
                    weight = clamp(weight, 0.0, 1.0);
                    float signal = (noise(p * frequency) - 0.5 + off) * amplitude;
                    value += weight * signal;
                    weight *= signal;
                    frequency *= lac;
                }
                return value;
            }

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;
                vec2 p = v_uv * scale;

                float value;
                int t = int(type);
                if(t == 0) value = musgrave_fBm(p, detail, dimension, lacunarity);
                else if(t == 1) value = musgrave_ridged(p, detail, dimension, lacunarity, offset, gain);
                else if(t == 2) value = musgrave_hybrid(p, detail, dimension, lacunarity, offset);
                else if(t == 3) value = noise(p * detail); // Simple
                else value = abs(musgrave_fBm(p, detail, dimension, lacunarity)); // Turbulence

                value = value * 0.5 + 0.5;
                vec3 pattern = vec3(value);
                vec3 result = mix(tex_color, tex_color * pattern + pattern * 0.2, blend_amount);

                f_color = vec4(result, 1.0);
            }
        """
    },
    "Wave Texture": {
        "description": "Bands, rings, and sine-wave patterns.",
        "category": "Procedural",
        "uniforms": {
            "wave_type": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 1.0},
            "profile": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 1.0},
            "scale": {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5},
            "distortion": {"type": "float", "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.5},
            "detail": {"type": "float", "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5},
            "detail_scale": {"type": "float", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
            "phase": {"type": "float", "default": 0.0, "min": 0.0, "max": 6.28, "step": 0.1},
            "blend_amount": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float wave_type;
            uniform float profile;
            uniform float scale;
            uniform float distortion;
            uniform float detail;
            uniform float detail_scale;
            uniform float phase;
            uniform float blend_amount;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
                           mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
            }

            float fbm(vec2 p, float octaves) {
                float value = 0.0;
                float amplitude = 0.5;
                for(int i = 0; i < int(octaves); i++) {
                    value += amplitude * noise(p);
                    p *= 2.0;
                    amplitude *= 0.5;
                }
                return value;
            }

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;
                vec2 p = v_uv * scale;

                float coord;
                int wt = int(wave_type);
                if(wt == 0) coord = p.x;  // Bands
                else if(wt == 1) coord = length(p - vec2(scale * 0.5));  // Rings
                else coord = (p.x + p.y) * 0.5;  // Diagonal

                if(distortion > 0.0) {
                    coord += fbm(p * detail_scale, detail) * distortion;
                }

                coord += phase;
                float value;
                int pr = int(profile);
                if(pr == 0) value = sin(coord) * 0.5 + 0.5;  // Sine
                else if(pr == 1) value = fract(coord);  // Saw
                else value = step(0.5, fract(coord));  // Square

                vec3 pattern = vec3(value);
                vec3 result = mix(tex_color, tex_color * pattern + pattern * 0.3, blend_amount);

                f_color = vec4(result, 1.0);
            }
        """
    },
    "Brick Texture": {
        "description": "Procedural brick and tile patterns.",
        "category": "Procedural",
        "uniforms": {
            "brick_width": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
            "brick_height": {"type": "float", "default": 0.25, "min": 0.05, "max": 0.5, "step": 0.025},
            "mortar_size": {"type": "float", "default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005},
            "mortar_smooth": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.02},
            "row_offset": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
            "brick_color_r": {"type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05},
            "brick_color_g": {"type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05},
            "brick_color_b": {"type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05},
            "mortar_color_r": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
            "mortar_color_g": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
            "mortar_color_b": {"type": "float", "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05},
            "color_variation": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.02},
            "blend_amount": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float brick_width, brick_height;
            uniform float mortar_size, mortar_smooth;
            uniform float row_offset;
            uniform float brick_color_r, brick_color_g, brick_color_b;
            uniform float mortar_color_r, mortar_color_g, mortar_color_b;
            uniform float color_variation;
            uniform float blend_amount;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;
                vec2 p = v_uv;

                // Calculate row
                float row = floor(p.y / brick_height);
                float offset = mod(row, 2.0) * row_offset * brick_width;

                // Calculate brick coordinates
                vec2 brick_uv;
                brick_uv.x = mod(p.x + offset, brick_width) / brick_width;
                brick_uv.y = mod(p.y, brick_height) / brick_height;

                // Calculate brick ID for color variation
                vec2 brick_id = vec2(floor((p.x + offset) / brick_width), row);

                // Mortar detection
                float mortar_x = smoothstep(mortar_size, mortar_size + mortar_smooth, brick_uv.x) *
                                 smoothstep(mortar_size, mortar_size + mortar_smooth, 1.0 - brick_uv.x);
                float mortar_y = smoothstep(mortar_size / brick_width * brick_height, (mortar_size + mortar_smooth) / brick_width * brick_height, brick_uv.y) *
                                 smoothstep(mortar_size / brick_width * brick_height, (mortar_size + mortar_smooth) / brick_width * brick_height, 1.0 - brick_uv.y);
                float mortar = 1.0 - mortar_x * mortar_y;

                // Brick and mortar colors
                vec3 brick_col = vec3(brick_color_r, brick_color_g, brick_color_b);
                vec3 mortar_col = vec3(mortar_color_r, mortar_color_g, mortar_color_b);

                // Add variation to brick color
                float var = (hash(brick_id) - 0.5) * 2.0 * color_variation;
                brick_col += var;

                vec3 pattern = mix(brick_col, mortar_col, mortar);
                vec3 result = mix(tex_color, pattern, blend_amount);

                f_color = vec4(result, 1.0);
            }
        """
    },
    "Gradient Texture": {
        "description": "Linear, radial, and spherical gradients.",
        "category": "Procedural",
        "uniforms": {
            "gradient_type": {"type": "float", "default": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
            "rotation": {"type": "float", "default": 0.0, "min": 0.0, "max": 6.28, "step": 0.1},
            "center_x": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "center_y": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "scale": {"type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
            "color1_r": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color1_g": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color1_b": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_g": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_b": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "blend_amount": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float gradient_type;
            uniform float rotation;
            uniform float center_x, center_y;
            uniform float scale;
            uniform float color1_r, color1_g, color1_b;
            uniform float color2_r, color2_g, color2_b;
            uniform float blend_amount;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;
                vec2 center = vec2(center_x, center_y);
                vec2 p = (v_uv - center) * scale;

                // Apply rotation for linear gradient
                float c = cos(rotation);
                float s = sin(rotation);
                vec2 rotated = vec2(p.x * c - p.y * s, p.x * s + p.y * c);

                float value;
                int gt = int(gradient_type);
                if(gt == 0) {
                    // Linear
                    value = rotated.x + 0.5;
                } else if(gt == 1) {
                    // Radial
                    value = length(p);
                } else if(gt == 2) {
                    // Quadratic sphere
                    value = 1.0 - length(p);
                } else if(gt == 3) {
                    // Diagonal
                    value = (rotated.x + rotated.y) * 0.5 + 0.5;
                } else {
                    // Angular
                    value = (atan(p.y, p.x) / 3.14159 + 1.0) * 0.5;
                }

                value = clamp(value, 0.0, 1.0);
                vec3 color1 = vec3(color1_r, color1_g, color1_b);
                vec3 color2 = vec3(color2_r, color2_g, color2_b);
                vec3 gradient = mix(color1, color2, value);

                vec3 result = mix(tex_color, gradient, blend_amount);
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Checker Texture": {
        "description": "Checkerboard patterns with customizable colors and scale.",
        "category": "Procedural",
        "uniforms": {
            "scale_x": {"type": "float", "default": 8.0, "min": 1.0, "max": 32.0, "step": 1.0},
            "scale_y": {"type": "float", "default": 8.0, "min": 1.0, "max": 32.0, "step": 1.0},
            "color1_r": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color1_g": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color1_b": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_g": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_b": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "blend_amount": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "softness": {"type": "float", "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float scale_x, scale_y;
            uniform float color1_r, color1_g, color1_b;
            uniform float color2_r, color2_g, color2_b;
            uniform float blend_amount;
            uniform float softness;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;

                vec2 p = v_uv * vec2(scale_x, scale_y);
                vec2 f = fract(p);

                float checker;
                if(softness > 0.0) {
                    float fx = smoothstep(0.5 - softness, 0.5 + softness, f.x);
                    float fy = smoothstep(0.5 - softness, 0.5 + softness, f.y);
                    checker = abs(fx - fy);
                } else {
                    checker = mod(floor(p.x) + floor(p.y), 2.0);
                }

                vec3 color1 = vec3(color1_r, color1_g, color1_b);
                vec3 color2 = vec3(color2_r, color2_g, color2_b);
                vec3 pattern = mix(color1, color2, checker);

                vec3 result = mix(tex_color, pattern, blend_amount);
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Magic Texture": {
        "description": "Psychedelic, trippy color patterns.",
        "category": "Procedural",
        "uniforms": {
            "scale": {"type": "float", "default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5},
            "turbulence": {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 1.0},
            "distortion": {"type": "float", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
            "color_shift": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "blend_amount": {"type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05},
            "time": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float scale;
            uniform float turbulence;
            uniform float distortion;
            uniform float color_shift;
            uniform float blend_amount;
            uniform float time;
            in vec2 v_uv;
            out vec4 f_color;

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;
                vec2 p = v_uv * scale + time * 0.05;

                float x = sin(p.x * turbulence + cos(p.y * turbulence * 0.7) * distortion);
                float y = cos(p.y * turbulence + sin(p.x * turbulence * 0.7) * distortion);
                float z = sin((p.x + p.y) * turbulence * 0.5);

                float hue = fract((x + y + z) * 0.33 + color_shift);
                vec3 magic = hsv2rgb(vec3(hue, 0.8, 0.9));

                vec3 result = mix(tex_color, magic, blend_amount);
                f_color = vec4(result, 1.0);
            }
        """
    },
    # ==================== COLOR MANIPULATION ====================
    "HSV Adjust": {
        "description": "Direct Hue, Saturation, Value color control.",
        "category": "Color",
        "uniforms": {
            "hue_shift": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.02},
            "saturation": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
            "value": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
            "colorize": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
            "colorize_hue": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float hue_shift;
            uniform float saturation;
            uniform float value;
            uniform float colorize;
            uniform float colorize_hue;
            in vec2 v_uv;
            out vec4 f_color;

            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec3 hsv = rgb2hsv(color);

                if(colorize > 0.5) {
                    hsv.x = colorize_hue;
                } else {
                    hsv.x = fract(hsv.x + hue_shift);
                }

                hsv.y *= saturation;
                hsv.z *= value;

                vec3 result = hsv2rgb(clamp(hsv, 0.0, 1.0));
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Color Ramp": {
        "description": "Map luminance to a custom color gradient.",
        "category": "Color",
        "uniforms": {
            "color1_r": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color1_g": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color1_b": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_r": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_g": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color2_b": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "color3_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color3_g": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "color3_b": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color4_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color4_g": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color4_b": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
            "pos2": {"type": "float", "default": 0.33, "min": 0.0, "max": 1.0, "step": 0.02},
            "pos3": {"type": "float", "default": 0.66, "min": 0.0, "max": 1.0, "step": 0.02},
            "blend_original": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float color1_r, color1_g, color1_b;
            uniform float color2_r, color2_g, color2_b;
            uniform float color3_r, color3_g, color3_b;
            uniform float color4_r, color4_g, color4_b;
            uniform float pos2, pos3;
            uniform float blend_original;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 tex_color = texture(u_texture, v_uv).rgb;
                float lum = dot(tex_color, vec3(0.299, 0.587, 0.114));

                vec3 c1 = vec3(color1_r, color1_g, color1_b);
                vec3 c2 = vec3(color2_r, color2_g, color2_b);
                vec3 c3 = vec3(color3_r, color3_g, color3_b);
                vec3 c4 = vec3(color4_r, color4_g, color4_b);

                vec3 ramp;
                if(lum < pos2) {
                    ramp = mix(c1, c2, lum / pos2);
                } else if(lum < pos3) {
                    ramp = mix(c2, c3, (lum - pos2) / (pos3 - pos2));
                } else {
                    ramp = mix(c3, c4, (lum - pos3) / (1.0 - pos3));
                }

                vec3 result = mix(ramp, tex_color * ramp, blend_original);
                f_color = vec4(result, 1.0);
            }
        """
    },
    "RGB Curves": {
        "description": "Individual R/G/B channel curve adjustments.",
        "category": "Color",
        "uniforms": {
            "r_black": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02},
            "r_white": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.02},
            "r_lift": {"type": "float", "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.02},
            "r_gamma": {"type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05},
            "g_black": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02},
            "g_white": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.02},
            "g_lift": {"type": "float", "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.02},
            "g_gamma": {"type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05},
            "b_black": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02},
            "b_white": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.02},
            "b_lift": {"type": "float", "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.02},
            "b_gamma": {"type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float r_black, r_white, r_lift, r_gamma;
            uniform float g_black, g_white, g_lift, g_gamma;
            uniform float b_black, b_white, b_lift, b_gamma;
            in vec2 v_uv;
            out vec4 f_color;

            float apply_curve(float v, float black, float white, float lift, float gamma) {
                v = (v - black) / (white - black);
                v = clamp(v, 0.0, 1.0);
                v = pow(v, gamma);
                v += lift;
                return clamp(v, 0.0, 1.0);
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;

                color.r = apply_curve(color.r, r_black, r_white, r_lift, r_gamma);
                color.g = apply_curve(color.g, g_black, g_white, g_lift, g_gamma);
                color.b = apply_curve(color.b, b_black, b_white, b_lift, b_gamma);

                f_color = vec4(color, 1.0);
            }
        """
    },
    "Blend Modes": {
        "description": "Screen, Overlay, Soft Light, Dodge, Burn, and more blend modes.",
        "category": "Color",
        "uniforms": {
            "blend_mode": {"type": "float", "default": 0.0, "min": 0.0, "max": 10.0, "step": 1.0},
            "blend_color_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "blend_color_g": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
            "blend_color_b": {"type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05},
            "opacity": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "use_luminance": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float blend_mode;
            uniform float blend_color_r, blend_color_g, blend_color_b;
            uniform float opacity;
            uniform float use_luminance;
            in vec2 v_uv;
            out vec4 f_color;

            vec3 blend_multiply(vec3 a, vec3 b) { return a * b; }
            vec3 blend_screen(vec3 a, vec3 b) { return 1.0 - (1.0 - a) * (1.0 - b); }
            vec3 blend_overlay(vec3 a, vec3 b) {
                return mix(2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b), step(0.5, a));
            }
            vec3 blend_soft_light(vec3 a, vec3 b) {
                return mix(2.0 * a * b + a * a * (1.0 - 2.0 * b), sqrt(a) * (2.0 * b - 1.0) + 2.0 * a * (1.0 - b), step(0.5, b));
            }
            vec3 blend_hard_light(vec3 a, vec3 b) {
                return mix(2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b), step(0.5, b));
            }
            vec3 blend_dodge(vec3 a, vec3 b) { return a / (1.0 - b + 0.001); }
            vec3 blend_burn(vec3 a, vec3 b) { return 1.0 - (1.0 - a) / (b + 0.001); }
            vec3 blend_difference(vec3 a, vec3 b) { return abs(a - b); }
            vec3 blend_exclusion(vec3 a, vec3 b) { return a + b - 2.0 * a * b; }
            vec3 blend_add(vec3 a, vec3 b) { return a + b; }
            vec3 blend_subtract(vec3 a, vec3 b) { return a - b; }

            void main() {
                vec3 base = texture(u_texture, v_uv).rgb;
                vec3 blend = vec3(blend_color_r, blend_color_g, blend_color_b);

                if(use_luminance > 0.5) {
                    float lum = dot(base, vec3(0.299, 0.587, 0.114));
                    blend = vec3(lum);
                }

                vec3 result;
                int mode = int(blend_mode);
                if(mode == 0) result = blend_multiply(base, blend);
                else if(mode == 1) result = blend_screen(base, blend);
                else if(mode == 2) result = blend_overlay(base, blend);
                else if(mode == 3) result = blend_soft_light(base, blend);
                else if(mode == 4) result = blend_hard_light(base, blend);
                else if(mode == 5) result = blend_dodge(base, blend);
                else if(mode == 6) result = blend_burn(base, blend);
                else if(mode == 7) result = blend_difference(base, blend);
                else if(mode == 8) result = blend_exclusion(base, blend);
                else if(mode == 9) result = blend_add(base, blend);
                else result = blend_subtract(base, blend);

                result = mix(base, clamp(result, 0.0, 1.0), opacity);
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Selective Color": {
        "description": "Adjust specific color ranges (reds, greens, blues, etc.).",
        "category": "Color",
        "uniforms": {
            "target_hue": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02},
            "hue_range": {"type": "float", "default": 0.1, "min": 0.02, "max": 0.5, "step": 0.02},
            "hue_shift": {"type": "float", "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.02},
            "sat_adjust": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
            "val_adjust": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float target_hue;
            uniform float hue_range;
            uniform float hue_shift;
            uniform float sat_adjust;
            uniform float val_adjust;
            in vec2 v_uv;
            out vec4 f_color;

            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec3 hsv = rgb2hsv(color);

                // Calculate how close the hue is to the target
                float hue_diff = min(abs(hsv.x - target_hue), 1.0 - abs(hsv.x - target_hue));
                float mask = smoothstep(hue_range, 0.0, hue_diff) * hsv.y; // Also consider saturation

                // Apply adjustments only to selected colors
                vec3 adjusted_hsv = hsv;
                adjusted_hsv.x = fract(hsv.x + hue_shift * mask);
                adjusted_hsv.y = clamp(hsv.y + sat_adjust * mask, 0.0, 1.0);
                adjusted_hsv.z = clamp(hsv.z + val_adjust * mask, 0.0, 1.0);

                vec3 result = hsv2rgb(adjusted_hsv);
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Channel Mixer": {
        "description": "Swap and blend RGB channels.",
        "category": "Color",
        "uniforms": {
            "rr": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05},
            "rg": {"type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05},
            "rb": {"type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05},
            "gr": {"type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05},
            "gg": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05},
            "gb": {"type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05},
            "br": {"type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05},
            "bg": {"type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05},
            "bb": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float rr, rg, rb;
            uniform float gr, gg, gb;
            uniform float br, bg, bb;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;

                mat3 mixer = mat3(
                    rr, gr, br,
                    rg, gg, bg,
                    rb, gb, bb
                );

                vec3 result = mixer * color;
                f_color = vec4(clamp(result, 0.0, 1.0), 1.0);
            }
        """
    },
    # ==================== DISTORTION EFFECTS ====================
    "Displacement": {
        "description": "Warp and distort image based on noise patterns.",
        "category": "Distortion",
        "uniforms": {
            "strength": {"type": "float", "default": 0.05, "min": 0.0, "max": 0.3, "step": 0.005},
            "scale": {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5},
            "octaves": {"type": "float", "default": 3.0, "min": 1.0, "max": 6.0, "step": 1.0},
            "time": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float strength;
            uniform float scale;
            uniform float octaves;
            uniform float time;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
                           mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
            }

            float fbm(vec2 p) {
                float value = 0.0;
                float amp = 0.5;
                for(int i = 0; i < int(octaves); i++) {
                    value += amp * noise(p);
                    p *= 2.0;
                    amp *= 0.5;
                }
                return value;
            }

            void main() {
                vec2 p = v_uv * scale + time * 0.1;
                vec2 offset = vec2(
                    fbm(p) - 0.5,
                    fbm(p + vec2(5.2, 1.3)) - 0.5
                ) * strength;

                vec2 uv = clamp(v_uv + offset, 0.0, 1.0);
                f_color = texture(u_texture, uv);
            }
        """
    },
    "Spherize": {
        "description": "Fisheye and barrel/pincushion distortion effects.",
        "category": "Distortion",
        "uniforms": {
            "strength": {"type": "float", "default": 0.5, "min": -1.0, "max": 1.0, "step": 0.05},
            "center_x": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "center_y": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "radius": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float strength;
            uniform float center_x, center_y;
            uniform float radius;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 center = vec2(center_x, center_y);
                vec2 p = v_uv - center;
                float dist = length(p);
                float r = dist / radius;

                vec2 uv;
                if(dist < radius) {
                    float power = (2.0 * r - r * r);
                    float bind = sqrt(1.0 - power * strength);
                    uv = center + normalize(p) * dist * bind;
                } else {
                    uv = v_uv;
                }

                uv = clamp(uv, 0.0, 1.0);
                f_color = texture(u_texture, uv);
            }
        """
    },
    "Twirl": {
        "description": "Spiral/swirl distortion effect.",
        "category": "Distortion",
        "uniforms": {
            "angle": {"type": "float", "default": 3.14, "min": -12.56, "max": 12.56, "step": 0.1},
            "center_x": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "center_y": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "radius": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float angle;
            uniform float center_x, center_y;
            uniform float radius;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 center = vec2(center_x, center_y);
                vec2 p = v_uv - center;
                float dist = length(p);

                vec2 uv;
                if(dist < radius) {
                    float percent = (radius - dist) / radius;
                    float theta = percent * percent * angle;
                    float s = sin(theta);
                    float c = cos(theta);
                    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
                }

                uv = center + p;
                uv = clamp(uv, 0.0, 1.0);
                f_color = texture(u_texture, uv);
            }
        """
    },
    "Wave Distortion": {
        "description": "Sine wave displacement effect.",
        "category": "Distortion",
        "uniforms": {
            "amplitude_x": {"type": "float", "default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005},
            "amplitude_y": {"type": "float", "default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005},
            "frequency_x": {"type": "float", "default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0},
            "frequency_y": {"type": "float", "default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0},
            "phase": {"type": "float", "default": 0.0, "min": 0.0, "max": 6.28, "step": 0.1},
            "time": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float amplitude_x, amplitude_y;
            uniform float frequency_x, frequency_y;
            uniform float phase;
            uniform float time;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 offset = vec2(
                    sin(v_uv.y * frequency_y + phase + time) * amplitude_x,
                    sin(v_uv.x * frequency_x + phase + time) * amplitude_y
                );

                vec2 uv = clamp(v_uv + offset, 0.0, 1.0);
                f_color = texture(u_texture, uv);
            }
        """
    },
    "Liquify": {
        "description": "Fluid-like smearing and warping effect.",
        "category": "Distortion",
        "uniforms": {
            "strength": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.02},
            "scale": {"type": "float", "default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5},
            "turbulence": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.5},
            "time": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float strength;
            uniform float scale;
            uniform float turbulence;
            uniform float time;
            in vec2 v_uv;
            out vec4 f_color;

            vec2 hash2(vec2 p) {
                p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
                return -1.0 + 2.0 * fract(sin(p) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                vec2 u = f * f * (3.0 - 2.0 * f);
                return mix(mix(dot(hash2(i), f),
                               dot(hash2(i + vec2(1,0)), f - vec2(1,0)), u.x),
                           mix(dot(hash2(i + vec2(0,1)), f - vec2(0,1)),
                               dot(hash2(i + vec2(1,1)), f - vec2(1,1)), u.x), u.y);
            }

            void main() {
                vec2 p = v_uv * scale;
                float t = time * 0.2;

                vec2 flow = vec2(
                    noise(p + vec2(t, 0.0)),
                    noise(p + vec2(0.0, t) + 5.2)
                );

                flow += vec2(
                    noise(p * turbulence + flow + t),
                    noise(p * turbulence + flow + t + 3.7)
                ) * 0.5;

                vec2 uv = v_uv + flow * strength;
                uv = clamp(uv, 0.0, 1.0);
                f_color = texture(u_texture, uv);
            }
        """
    },
    "Kaleidoscope": {
        "description": "Mirror and repeat patterns in radial symmetry.",
        "category": "Distortion",
        "uniforms": {
            "segments": {"type": "float", "default": 6.0, "min": 2.0, "max": 16.0, "step": 1.0},
            "rotation": {"type": "float", "default": 0.0, "min": 0.0, "max": 6.28, "step": 0.05},
            "center_x": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "center_y": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "zoom": {"type": "float", "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float segments;
            uniform float rotation;
            uniform float center_x, center_y;
            uniform float zoom;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 center = vec2(center_x, center_y);
                vec2 p = (v_uv - center) * zoom;

                float angle = atan(p.y, p.x) + rotation;
                float radius = length(p);

                float segment_angle = 3.14159 * 2.0 / segments;
                angle = mod(angle, segment_angle);
                if(angle > segment_angle * 0.5) {
                    angle = segment_angle - angle;
                }

                vec2 uv = center + vec2(cos(angle), sin(angle)) * radius;
                uv = fract(uv); // Wrap around
                f_color = texture(u_texture, uv);
            }
        """
    },
    # ==================== ADVANCED LIGHTING ====================
    "Screen Space AO": {
        "description": "Fake ambient occlusion from image depth for added depth.",
        "category": "Lighting",
        "uniforms": {
            "radius": {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5},
            "strength": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1},
            "samples": {"type": "float", "default": 8.0, "min": 4.0, "max": 16.0, "step": 1.0},
            "falloff": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1},
            "only_ao": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float radius;
            uniform float strength;
            uniform float samples;
            uniform float falloff;
            uniform float only_ao;
            in vec2 v_uv;
            out vec4 f_color;

            float getDepth(vec2 uv) {
                return dot(texture(u_texture, uv).rgb, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));
                float center_depth = getDepth(v_uv);

                float ao = 0.0;
                int s = int(samples);
                float golden_angle = 2.39996;

                for(int i = 0; i < s; i++) {
                    float angle = float(i) * golden_angle;
                    float r = (float(i) + 0.5) / float(s) * radius;
                    vec2 offset = vec2(cos(angle), sin(angle)) * r * texel;

                    float sample_depth = getDepth(v_uv + offset);
                    float diff = center_depth - sample_depth;
                    ao += smoothstep(0.0, 0.1, diff);
                }

                ao = 1.0 - (ao / float(s)) * strength;
                ao = pow(ao, falloff);

                vec3 result = only_ao > 0.5 ? vec3(ao) : color * ao;
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Rim Light": {
        "description": "Edge/fresnel glow effect for dramatic lighting.",
        "category": "Lighting",
        "uniforms": {
            "rim_color_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "rim_color_g": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
            "rim_color_b": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
            "rim_power": {"type": "float", "default": 2.0, "min": 0.5, "max": 8.0, "step": 0.5},
            "rim_strength": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1},
            "light_angle": {"type": "float", "default": 0.0, "min": 0.0, "max": 6.28, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float rim_color_r, rim_color_g, rim_color_b;
            uniform float rim_power;
            uniform float rim_strength;
            uniform float light_angle;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                // Estimate normal from gradient
                float hL = dot(texture(u_texture, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hR = dot(texture(u_texture, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hD = dot(texture(u_texture, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float hU = dot(texture(u_texture, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));

                vec3 normal = normalize(vec3(hL - hR, hD - hU, 0.5));
                vec3 view = vec3(0.0, 0.0, 1.0);

                // Directional rim
                vec2 light_dir_2d = vec2(cos(light_angle), sin(light_angle));
                vec3 light_dir = normalize(vec3(light_dir_2d, 0.5));

                float rim = 1.0 - max(dot(normal, view), 0.0);
                rim = pow(rim, rim_power);

                // Enhance rim on light-facing side
                float light_factor = max(dot(normal.xy, light_dir_2d), 0.0);
                rim *= (0.5 + light_factor * 0.5);

                vec3 rim_color = vec3(rim_color_r, rim_color_g, rim_color_b);
                vec3 result = color + rim_color * rim * rim_strength;

                f_color = vec4(result, 1.0);
            }
        """
    },
    "Matcap": {
        "description": "Material capture/lit sphere lighting effect.",
        "category": "Lighting",
        "uniforms": {
            "matcap_type": {"type": "float", "default": 0.0, "min": 0.0, "max": 5.0, "step": 1.0},
            "intensity": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
            "blend": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float matcap_type;
            uniform float intensity;
            uniform float blend;
            in vec2 v_uv;
            out vec4 f_color;

            vec3 getMatcapColor(vec2 n, int type) {
                // Generate various matcap styles procedurally
                float d = length(n);
                float f = 1.0 - d;

                if(type == 0) {
                    // Shiny metal
                    vec3 base = vec3(0.8, 0.85, 0.9);
                    float highlight = pow(max(1.0 - length(n - vec2(-0.3, 0.3)), 0.0), 4.0);
                    return base * (0.5 + f * 0.5) + vec3(highlight);
                } else if(type == 1) {
                    // Warm clay
                    vec3 light = vec3(1.0, 0.9, 0.8);
                    vec3 shadow = vec3(0.4, 0.25, 0.2);
                    return mix(shadow, light, f);
                } else if(type == 2) {
                    // Cool blue
                    vec3 light = vec3(0.8, 0.9, 1.0);
                    vec3 shadow = vec3(0.1, 0.2, 0.4);
                    float rim = pow(d, 3.0);
                    return mix(shadow, light, f) + vec3(0.2, 0.4, 0.8) * rim;
                } else if(type == 3) {
                    // Jade
                    vec3 base = vec3(0.2, 0.6, 0.4);
                    float subsurface = pow(f, 0.5);
                    return base * subsurface + vec3(0.8, 1.0, 0.9) * pow(f, 4.0);
                } else if(type == 4) {
                    // Gold
                    vec3 base = vec3(0.9, 0.7, 0.3);
                    float spec = pow(max(1.0 - length(n - vec2(-0.2, 0.3)), 0.0), 8.0);
                    return base * (0.4 + f * 0.6) + vec3(1.0, 0.95, 0.8) * spec;
                } else {
                    // X-ray/rim
                    float rim = pow(d, 2.0);
                    return vec3(rim * 0.5, rim * 0.7, rim);
                }
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                // Estimate normal
                float hL = dot(texture(u_texture, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hR = dot(texture(u_texture, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hD = dot(texture(u_texture, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float hU = dot(texture(u_texture, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));

                vec2 normal = vec2(hL - hR, hD - hU) * 5.0;
                normal = clamp(normal, -1.0, 1.0);

                vec3 matcap = getMatcapColor(normal, int(matcap_type)) * intensity;
                vec3 result = mix(color, matcap * color + matcap * 0.3, blend);

                f_color = vec4(result, 1.0);
            }
        """
    },
    "Gooch Shading": {
        "description": "Warm-to-cool artistic technical illustration shading.",
        "category": "Lighting",
        "uniforms": {
            "warm_r": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
            "warm_g": {"type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05},
            "warm_b": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05},
            "cool_r": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05},
            "cool_g": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05},
            "cool_b": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
            "diffuse_warm": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05},
            "diffuse_cool": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05},
            "light_x": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1},
            "light_y": {"type": "float", "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float warm_r, warm_g, warm_b;
            uniform float cool_r, cool_g, cool_b;
            uniform float diffuse_warm, diffuse_cool;
            uniform float light_x, light_y;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                // Estimate normal
                float hL = dot(texture(u_texture, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hR = dot(texture(u_texture, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
                float hD = dot(texture(u_texture, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float hU = dot(texture(u_texture, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));

                vec3 normal = normalize(vec3(hL - hR, hD - hU, 0.5));
                vec3 light = normalize(vec3(light_x, light_y, 1.0));

                float NdotL = dot(normal, light);
                float t = (NdotL + 1.0) * 0.5;

                vec3 warm = vec3(warm_r, warm_g, warm_b);
                vec3 cool = vec3(cool_r, cool_g, cool_b);

                vec3 k_cool = cool + diffuse_cool * color;
                vec3 k_warm = warm + diffuse_warm * color;

                vec3 result = mix(k_cool, k_warm, t);
                f_color = vec4(result, 1.0);
            }
        """
    },
    "X-Ray": {
        "description": "See-through transparency effect with edge emphasis.",
        "category": "Lighting",
        "uniforms": {
            "edge_strength": {"type": "float", "default": 1.5, "min": 0.0, "max": 3.0, "step": 0.1},
            "fill_opacity": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
            "color_r": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
            "color_g": {"type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05},
            "color_b": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "invert": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float edge_strength;
            uniform float fill_opacity;
            uniform float color_r, color_g, color_b;
            uniform float invert;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                // Sobel edge detection
                float tl = dot(texture(u_texture, v_uv + vec2(-texel.x, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float t  = dot(texture(u_texture, v_uv + vec2(0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float tr = dot(texture(u_texture, v_uv + vec2(texel.x, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float l  = dot(texture(u_texture, v_uv + vec2(-texel.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
                float r  = dot(texture(u_texture, v_uv + vec2(texel.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
                float bl = dot(texture(u_texture, v_uv + vec2(-texel.x, -texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float b  = dot(texture(u_texture, v_uv + vec2(0, -texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float br = dot(texture(u_texture, v_uv + vec2(texel.x, -texel.y)).rgb, vec3(0.299, 0.587, 0.114));

                float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
                float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
                float edge = sqrt(gx*gx + gy*gy) * edge_strength;

                vec3 xray_color = vec3(color_r, color_g, color_b);
                float lum = dot(color, vec3(0.299, 0.587, 0.114));

                vec3 fill = xray_color * lum * fill_opacity;
                vec3 edges = xray_color * edge;

                vec3 result = fill + edges;

                if(invert > 0.5) {
                    result = vec3(1.0) - result;
                }

                f_color = vec4(result, 1.0);
            }
        """
    },
    # ==================== ARTISTIC/STYLIZATION ====================
    "Kuwahara": {
        "description": "Painterly smoothing filter that preserves edges.",
        "category": "Artistic",
        "uniforms": {
            "radius": {"type": "float", "default": 4.0, "min": 1.0, "max": 10.0, "step": 1.0},
            "sharpness": {"type": "float", "default": 8.0, "min": 1.0, "max": 20.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float radius;
            uniform float sharpness;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));
                int r = int(radius);

                vec3 mean[4];
                vec3 sigma[4];

                for(int i = 0; i < 4; i++) {
                    mean[i] = vec3(0.0);
                    sigma[i] = vec3(0.0);
                }

                vec2 offsets[4] = vec2[4](
                    vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1)
                );

                for(int i = 0; i < 4; i++) {
                    vec2 dir = offsets[i];
                    int count = 0;

                    for(int x = 0; x <= r; x++) {
                        for(int y = 0; y <= r; y++) {
                            vec2 offset = vec2(float(x) * dir.x, float(y) * dir.y) * texel;
                            vec3 c = texture(u_texture, v_uv + offset).rgb;
                            mean[i] += c;
                            sigma[i] += c * c;
                            count++;
                        }
                    }

                    mean[i] /= float(count);
                    sigma[i] = abs(sigma[i] / float(count) - mean[i] * mean[i]);
                }

                // Find region with minimum variance
                float min_sigma = dot(sigma[0], vec3(1.0));
                vec3 result = mean[0];

                for(int i = 1; i < 4; i++) {
                    float s = dot(sigma[i], vec3(1.0));
                    if(s < min_sigma) {
                        min_sigma = s;
                        result = mean[i];
                    }
                }

                f_color = vec4(result, 1.0);
            }
        """
    },
    "Cross Hatch": {
        "description": "Pen and ink hatching illustration style.",
        "category": "Artistic",
        "uniforms": {
            "density": {"type": "float", "default": 10.0, "min": 5.0, "max": 30.0, "step": 1.0},
            "thickness": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
            "angle1": {"type": "float", "default": 0.785, "min": 0.0, "max": 3.14, "step": 0.1},
            "angle2": {"type": "float", "default": 2.356, "min": 0.0, "max": 3.14, "step": 0.1},
            "levels": {"type": "float", "default": 4.0, "min": 2.0, "max": 8.0, "step": 1.0},
            "ink_color_r": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
            "ink_color_g": {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05},
            "ink_color_b": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "paper_color_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "paper_color_g": {"type": "float", "default": 0.98, "min": 0.0, "max": 1.0, "step": 0.05},
            "paper_color_b": {"type": "float", "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float density;
            uniform float thickness;
            uniform float angle1, angle2;
            uniform float levels;
            uniform float ink_color_r, ink_color_g, ink_color_b;
            uniform float paper_color_r, paper_color_g, paper_color_b;
            in vec2 v_uv;
            out vec4 f_color;

            float hatch(vec2 uv, float angle, float dens) {
                float c = cos(angle);
                float s = sin(angle);
                vec2 rotated = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
                return abs(sin(rotated.x * dens * 3.14159));
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                float lum = dot(color, vec3(0.299, 0.587, 0.114));

                // Quantize luminance
                float quantized = floor(lum * levels) / levels;
                float darkness = 1.0 - quantized;

                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 uv = v_uv * tex_size / 10.0;

                float h1 = hatch(uv, angle1, density);
                float h2 = hatch(uv, angle2, density);

                float pattern = 1.0;

                if(darkness > 0.25) pattern = min(pattern, smoothstep(thickness, thickness + 0.2, h1));
                if(darkness > 0.5) pattern = min(pattern, smoothstep(thickness, thickness + 0.2, h2));
                if(darkness > 0.75) {
                    float h3 = hatch(uv, angle1 + 0.5, density * 1.5);
                    pattern = min(pattern, smoothstep(thickness * 0.7, thickness * 0.7 + 0.2, h3));
                }

                vec3 ink = vec3(ink_color_r, ink_color_g, ink_color_b);
                vec3 paper = vec3(paper_color_r, paper_color_g, paper_color_b);

                vec3 result = mix(ink, paper, pattern);
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Stippling": {
        "description": "Dot-based shading technique.",
        "category": "Artistic",
        "uniforms": {
            "dot_size": {"type": "float", "default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5},
            "density": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1},
            "randomness": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
            "ink_color_r": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "ink_color_g": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "ink_color_b": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "paper_color_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "paper_color_g": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "paper_color_b": {"type": "float", "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float dot_size;
            uniform float density;
            uniform float randomness;
            uniform float ink_color_r, ink_color_g, ink_color_b;
            uniform float paper_color_r, paper_color_g, paper_color_b;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                float lum = dot(color, vec3(0.299, 0.587, 0.114));
                float darkness = 1.0 - lum;

                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 cell_size = vec2(dot_size) / tex_size;
                vec2 cell = floor(v_uv / cell_size);
                vec2 cell_uv = fract(v_uv / cell_size);

                // Random offset within cell
                vec2 offset = (hash(cell) * 2.0 - 1.0) * randomness * 0.3;
                vec2 center = vec2(0.5) + offset;

                float dist = length(cell_uv - center);

                // Dot radius based on darkness
                float max_radius = 0.45 * density;
                float dot_radius = darkness * max_radius;

                // Add some variation
                dot_radius *= 0.8 + hash(cell + 0.5) * 0.4;

                float dot = 1.0 - smoothstep(dot_radius - 0.1, dot_radius, dist);

                vec3 ink = vec3(ink_color_r, ink_color_g, ink_color_b);
                vec3 paper = vec3(paper_color_r, paper_color_g, paper_color_b);

                vec3 result = mix(paper, ink, dot);
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Woodcut": {
        "description": "Bold black and white woodblock print style.",
        "category": "Artistic",
        "uniforms": {
            "threshold": {"type": "float", "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05},
            "line_density": {"type": "float", "default": 40.0, "min": 10.0, "max": 100.0, "step": 5.0},
            "line_thickness": {"type": "float", "default": 0.4, "min": 0.1, "max": 0.8, "step": 0.05},
            "distortion": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.02},
            "invert": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float threshold;
            uniform float line_density;
            uniform float line_thickness;
            uniform float distortion;
            uniform float invert;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                float lum = dot(color, vec3(0.299, 0.587, 0.114));

                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 p = v_uv * tex_size;

                // Add wood grain distortion
                float noise = hash(floor(p / 10.0)) * distortion;
                p.x += sin(p.y * 0.1) * noise * 20.0;

                // Create line pattern that varies with luminance
                float line = abs(sin(p.y / tex_size.y * line_density * 3.14159));

                // Threshold based on luminance
                float t = threshold + (1.0 - lum) * 0.3;
                float pattern = smoothstep(t - line_thickness * 0.5, t + line_thickness * 0.5, line + lum * 0.3);

                if(invert > 0.5) pattern = 1.0 - pattern;

                f_color = vec4(vec3(pattern), 1.0);
            }
        """
    },
    "Stained Glass": {
        "description": "Colored segmented regions like stained glass windows.",
        "category": "Artistic",
        "uniforms": {
            "cell_size": {"type": "float", "default": 20.0, "min": 5.0, "max": 50.0, "step": 1.0},
            "edge_width": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.5},
            "edge_color_r": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
            "edge_color_g": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
            "edge_color_b": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
            "saturation_boost": {"type": "float", "default": 1.3, "min": 0.5, "max": 2.0, "step": 0.1},
            "light_variation": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float cell_size;
            uniform float edge_width;
            uniform float edge_color_r, edge_color_g, edge_color_b;
            uniform float saturation_boost;
            uniform float light_variation;
            in vec2 v_uv;
            out vec4 f_color;

            vec2 hash2(vec2 p) {
                p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
                return fract(sin(p) * 43758.5453);
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 p = v_uv * tex_size / cell_size;
                vec2 cell = floor(p);
                vec2 f = fract(p);

                // Voronoi for cell boundaries
                float min_dist = 8.0;
                vec2 closest_cell = cell;

                for(int j = -1; j <= 1; j++) {
                    for(int i = -1; i <= 1; i++) {
                        vec2 neighbor = cell + vec2(i, j);
                        vec2 point = hash2(neighbor);
                        vec2 diff = neighbor + point - p;
                        float d = dot(diff, diff);
                        if(d < min_dist) {
                            min_dist = d;
                            closest_cell = neighbor;
                        }
                    }
                }

                // Get color from cell center
                vec2 cell_center = (closest_cell + hash2(closest_cell)) * cell_size / tex_size;
                cell_center = clamp(cell_center, 0.0, 1.0);
                vec3 cell_color = texture(u_texture, cell_center).rgb;

                // Boost saturation
                float lum = dot(cell_color, vec3(0.299, 0.587, 0.114));
                cell_color = mix(vec3(lum), cell_color, saturation_boost);

                // Add light variation
                float variation = hash2(closest_cell).x * light_variation;
                cell_color *= (1.0 - light_variation * 0.5 + variation);

                // Edge detection
                float edge = smoothstep(0.0, edge_width * edge_width / (cell_size * cell_size), min_dist);
                vec3 edge_col = vec3(edge_color_r, edge_color_g, edge_color_b);

                vec3 result = mix(edge_col, cell_color, edge);
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Palette Swap": {
        "description": "Remap colors to a limited color palette.",
        "category": "Artistic",
        "uniforms": {
            "palette_size": {"type": "float", "default": 8.0, "min": 2.0, "max": 16.0, "step": 1.0},
            "palette_type": {"type": "float", "default": 0.0, "min": 0.0, "max": 5.0, "step": 1.0},
            "dither_amount": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float palette_size;
            uniform float palette_type;
            uniform float dither_amount;
            in vec2 v_uv;
            out vec4 f_color;

            vec3 getPaletteColor(int index, int palette) {
                // Various preset palettes
                if(palette == 0) {
                    // GameBoy
                    vec3 gb[4] = vec3[4](
                        vec3(0.06, 0.22, 0.06),
                        vec3(0.19, 0.38, 0.19),
                        vec3(0.55, 0.67, 0.06),
                        vec3(0.61, 0.74, 0.06)
                    );
                    return gb[index % 4];
                } else if(palette == 1) {
                    // CGA
                    vec3 cga[4] = vec3[4](
                        vec3(0.0, 0.0, 0.0),
                        vec3(0.33, 1.0, 1.0),
                        vec3(1.0, 0.33, 1.0),
                        vec3(1.0, 1.0, 1.0)
                    );
                    return cga[index % 4];
                } else if(palette == 2) {
                    // Sepia
                    float t = float(index) / 7.0;
                    return vec3(t * 0.9 + 0.1, t * 0.7 + 0.05, t * 0.4);
                } else if(palette == 3) {
                    // Neon
                    vec3 neon[6] = vec3[6](
                        vec3(0.0, 0.0, 0.0),
                        vec3(1.0, 0.0, 0.5),
                        vec3(0.0, 1.0, 1.0),
                        vec3(1.0, 1.0, 0.0),
                        vec3(0.5, 0.0, 1.0),
                        vec3(1.0, 1.0, 1.0)
                    );
                    return neon[index % 6];
                } else if(palette == 4) {
                    // Earth tones
                    vec3 earth[5] = vec3[5](
                        vec3(0.2, 0.15, 0.1),
                        vec3(0.4, 0.3, 0.2),
                        vec3(0.6, 0.5, 0.35),
                        vec3(0.75, 0.65, 0.5),
                        vec3(0.9, 0.85, 0.75)
                    );
                    return earth[index % 5];
                } else {
                    // Rainbow
                    float hue = float(index) / palette_size;
                    vec3 rgb = abs(mod(hue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0;
                    return clamp(rgb, 0.0, 1.0);
                }
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;

                // Bayer dithering
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                ivec2 pos = ivec2(v_uv * tex_size) % 4;
                float bayer[16] = float[16](
                    0.0/16.0, 8.0/16.0, 2.0/16.0, 10.0/16.0,
                    12.0/16.0, 4.0/16.0, 14.0/16.0, 6.0/16.0,
                    3.0/16.0, 11.0/16.0, 1.0/16.0, 9.0/16.0,
                    15.0/16.0, 7.0/16.0, 13.0/16.0, 5.0/16.0
                );
                float dither = bayer[pos.y * 4 + pos.x] - 0.5;

                color += dither * dither_amount * 0.2;

                // Find closest palette color
                float lum = dot(color, vec3(0.299, 0.587, 0.114));
                int index = int(lum * (palette_size - 1.0) + 0.5);
                index = clamp(index, 0, int(palette_size) - 1);

                vec3 result = getPaletteColor(index, int(palette_type));
                f_color = vec4(result, 1.0);
            }
        """
    },
    # ==================== COMPOSITING EFFECTS ====================
    "Lens Flare": {
        "description": "Sun and light flare effects.",
        "category": "Compositing",
        "uniforms": {
            "flare_x": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.02},
            "flare_y": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.02},
            "intensity": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1},
            "glow_size": {"type": "float", "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05},
            "streaks": {"type": "float", "default": 6.0, "min": 0.0, "max": 12.0, "step": 1.0},
            "chromatic": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
            "ghosts": {"type": "float", "default": 3.0, "min": 0.0, "max": 6.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float flare_x, flare_y;
            uniform float intensity;
            uniform float glow_size;
            uniform float streaks;
            uniform float chromatic;
            uniform float ghosts;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 center = vec2(flare_x, flare_y);
                vec2 p = v_uv - center;
                float dist = length(p);
                float angle = atan(p.y, p.x);

                // Main glow
                float glow = exp(-dist * dist / (glow_size * glow_size * 0.5));

                // Streaks
                float streak = 0.0;
                if(streaks > 0.0) {
                    streak = pow(abs(sin(angle * streaks)), 10.0) * glow * 0.5;
                }

                // Lens ghosts
                vec3 ghost_color = vec3(0.0);
                for(int i = 0; i < int(ghosts); i++) {
                    float t = (float(i) + 1.0) / (ghosts + 1.0);
                    vec2 ghost_pos = center - p * (t * 2.0);
                    float ghost_dist = length(v_uv - ghost_pos);
                    float ghost_size = 0.05 + t * 0.1;
                    float g = exp(-ghost_dist * ghost_dist / (ghost_size * ghost_size));

                    // Chromatic separation
                    vec3 gc = vec3(
                        exp(-(ghost_dist - chromatic * 0.02) * (ghost_dist - chromatic * 0.02) / (ghost_size * ghost_size)),
                        g,
                        exp(-(ghost_dist + chromatic * 0.02) * (ghost_dist + chromatic * 0.02) / (ghost_size * ghost_size))
                    );
                    ghost_color += gc * 0.3;
                }

                // Chromatic main glow
                vec3 flare_color = vec3(
                    exp(-pow(dist - chromatic * 0.05, 2.0) / (glow_size * glow_size * 0.5)),
                    glow,
                    exp(-pow(dist + chromatic * 0.05, 2.0) / (glow_size * glow_size * 0.5))
                );

                flare_color += streak;
                flare_color += ghost_color;

                vec3 result = color + flare_color * intensity;
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Light Streaks": {
        "description": "Anamorphic horizontal light streaks.",
        "category": "Compositing",
        "uniforms": {
            "threshold": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
            "streak_length": {"type": "float", "default": 0.3, "min": 0.05, "max": 0.8, "step": 0.05},
            "intensity": {"type": "float", "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1},
            "falloff": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.5},
            "samples": {"type": "float", "default": 16.0, "min": 4.0, "max": 32.0, "step": 4.0},
            "color_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "color_g": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
            "color_b": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float threshold;
            uniform float streak_length;
            uniform float intensity;
            uniform float falloff;
            uniform float samples;
            uniform float color_r, color_g, color_b;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                float lum = dot(color, vec3(0.299, 0.587, 0.114));

                vec3 streak = vec3(0.0);
                int s = int(samples);

                for(int i = -s; i <= s; i++) {
                    float t = float(i) / float(s);
                    vec2 offset = vec2(t * streak_length, 0.0);
                    vec3 sample_color = texture(u_texture, clamp(v_uv + offset, 0.0, 1.0)).rgb;
                    float sample_lum = dot(sample_color, vec3(0.299, 0.587, 0.114));

                    if(sample_lum > threshold) {
                        float weight = pow(1.0 - abs(t), falloff);
                        streak += (sample_color - threshold) * weight;
                    }
                }

                streak /= float(s * 2 + 1);
                vec3 tint = vec3(color_r, color_g, color_b);

                vec3 result = color + streak * tint * intensity;
                f_color = vec4(result, 1.0);
            }
        """
    },
    "God Rays": {
        "description": "Volumetric light shafts / sun beams.",
        "category": "Compositing",
        "uniforms": {
            "light_x": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.02},
            "light_y": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.02},
            "intensity": {"type": "float", "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1},
            "decay": {"type": "float", "default": 0.95, "min": 0.8, "max": 1.0, "step": 0.01},
            "samples": {"type": "float", "default": 50.0, "min": 10.0, "max": 100.0, "step": 10.0},
            "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float light_x, light_y;
            uniform float intensity;
            uniform float decay;
            uniform float samples;
            uniform float threshold;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 light_pos = vec2(light_x, light_y);
                vec2 delta = (v_uv - light_pos) / samples;

                vec2 uv = v_uv;
                vec3 rays = vec3(0.0);
                float illumination_decay = 1.0;

                for(int i = 0; i < int(samples); i++) {
                    uv -= delta;
                    vec3 sample_color = texture(u_texture, clamp(uv, 0.0, 1.0)).rgb;
                    float sample_lum = dot(sample_color, vec3(0.299, 0.587, 0.114));

                    if(sample_lum > threshold) {
                        rays += (sample_color - threshold * 0.5) * illumination_decay;
                    }

                    illumination_decay *= decay;
                }

                rays /= samples;

                vec3 result = color + rays * intensity;
                f_color = vec4(result, 1.0);
            }
        """
    },
    "Lens Dirt": {
        "description": "Dirty/dusty lens overlay effect.",
        "category": "Compositing",
        "uniforms": {
            "dirt_amount": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
            "dirt_scale": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.5},
            "bloom_threshold": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
            "bloom_intensity": {"type": "float", "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float dirt_amount;
            uniform float dirt_scale;
            uniform float bloom_threshold;
            uniform float bloom_intensity;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
                           mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
            }

            float fbm(vec2 p) {
                float value = 0.0;
                float amplitude = 0.5;
                for(int i = 0; i < 5; i++) {
                    value += amplitude * noise(p);
                    p *= 2.0;
                    amplitude *= 0.5;
                }
                return value;
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;

                // Generate dirt pattern
                vec2 p = v_uv * dirt_scale;
                float dirt = fbm(p * 3.0);
                dirt *= fbm(p * 7.0 + 5.2);
                dirt = pow(dirt, 0.5) * dirt_amount;

                // Simple bloom for bright areas
                vec3 bloom = vec3(0.0);
                float lum = dot(color, vec3(0.299, 0.587, 0.114));
                if(lum > bloom_threshold) {
                    bloom = (color - bloom_threshold) * bloom_intensity;
                }

                // Dirt reveals bloom
                vec3 result = color + bloom * dirt * 2.0;

                // Add slight dirt tint
                result = mix(result, result * vec3(1.0, 0.95, 0.9), dirt * 0.3);

                f_color = vec4(result, 1.0);
            }
        """
    },
    "Bokeh": {
        "description": "Custom bokeh shapes (circles, hexagons, hearts, stars).",
        "category": "Compositing",
        "uniforms": {
            "bokeh_shape": {"type": "float", "default": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
            "blur_amount": {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 1.0},
            "threshold": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
            "bokeh_size": {"type": "float", "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1},
            "rotation": {"type": "float", "default": 0.0, "min": 0.0, "max": 6.28, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float bokeh_shape;
            uniform float blur_amount;
            uniform float threshold;
            uniform float bokeh_size;
            uniform float rotation;
            in vec2 v_uv;
            out vec4 f_color;

            float sdCircle(vec2 p, float r) {
                return length(p) - r;
            }

            float sdHexagon(vec2 p, float r) {
                p = abs(p);
                return max(p.x - r * 0.866, p.y + p.x * 0.5 - r);
            }

            float sdHeart(vec2 p, float r) {
                p.x = abs(p.x);
                if(p.y + p.x > r) return length(p - vec2(0.25, 0.75) * r) - r * 0.25;
                return length(p - vec2(0.0, 1.0) * r) - r * 0.5;
            }

            float sdStar(vec2 p, float r, int n) {
                float an = 3.14159 / float(n);
                float en = 3.14159 / 3.0;
                vec2 acs = vec2(cos(an), sin(an));
                vec2 ecs = vec2(cos(en), sin(en));
                float bn = mod(atan(p.x, p.y), 2.0 * an) - an;
                p = length(p) * vec2(cos(bn), abs(sin(bn)));
                p -= r * acs;
                p += ecs * clamp(-dot(p, ecs), 0.0, r * acs.y / ecs.y);
                return length(p) * sign(p.x);
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                vec3 bokeh = vec3(0.0);
                float total_weight = 0.0;
                int samples = int(blur_amount * 2.0);

                float c = cos(rotation);
                float s = sin(rotation);
                mat2 rot = mat2(c, -s, s, c);

                for(int x = -samples; x <= samples; x++) {
                    for(int y = -samples; y <= samples; y++) {
                        vec2 offset = vec2(float(x), float(y)) * texel * blur_amount;
                        vec2 p = offset / (blur_amount * texel * bokeh_size);
                        p = rot * p;

                        float dist;
                        int shape = int(bokeh_shape);
                        if(shape == 0) dist = sdCircle(p, 1.0);
                        else if(shape == 1) dist = sdHexagon(p, 1.0);
                        else if(shape == 2) dist = sdHeart(p * vec2(1.0, -1.0) + vec2(0.0, 0.5), 0.5);
                        else if(shape == 3) dist = sdStar(p, 1.0, 5);
                        else dist = sdStar(p, 1.0, 6);

                        float weight = 1.0 - smoothstep(-0.1, 0.1, dist);

                        if(weight > 0.0) {
                            vec3 sample_color = texture(u_texture, v_uv + offset).rgb;
                            float sample_lum = dot(sample_color, vec3(0.299, 0.587, 0.114));

                            if(sample_lum > threshold) {
                                bokeh += sample_color * weight * (sample_lum - threshold);
                            }
                            bokeh += sample_color * weight * 0.05;
                            total_weight += weight;
                        }
                    }
                }

                if(total_weight > 0.0) {
                    bokeh /= total_weight;
                }

                f_color = vec4(bokeh, 1.0);
            }
        """
    },
    # ==================== EDGE EFFECTS ====================
    "Dilate": {
        "description": "Expand/grow bright regions.",
        "category": "Edge",
        "uniforms": {
            "radius": {"type": "float", "default": 2.0, "min": 1.0, "max": 10.0, "step": 1.0},
            "shape": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float radius;
            uniform float shape;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));
                int r = int(radius);
                vec3 max_color = vec3(0.0);

                for(int x = -r; x <= r; x++) {
                    for(int y = -r; y <= r; y++) {
                        if(shape > 0.5) {
                            // Circular
                            if(x*x + y*y > r*r) continue;
                        }
                        vec2 offset = vec2(float(x), float(y)) * texel;
                        vec3 sample_color = texture(u_texture, v_uv + offset).rgb;
                        max_color = max(max_color, sample_color);
                    }
                }

                f_color = vec4(max_color, 1.0);
            }
        """
    },
    "Erode": {
        "description": "Shrink bright regions / expand dark regions.",
        "category": "Edge",
        "uniforms": {
            "radius": {"type": "float", "default": 2.0, "min": 1.0, "max": 10.0, "step": 1.0},
            "shape": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float radius;
            uniform float shape;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));
                int r = int(radius);
                vec3 min_color = vec3(1.0);

                for(int x = -r; x <= r; x++) {
                    for(int y = -r; y <= r; y++) {
                        if(shape > 0.5) {
                            if(x*x + y*y > r*r) continue;
                        }
                        vec2 offset = vec2(float(x), float(y)) * texel;
                        vec3 sample_color = texture(u_texture, v_uv + offset).rgb;
                        min_color = min(min_color, sample_color);
                    }
                }

                f_color = vec4(min_color, 1.0);
            }
        """
    },
    "Laplacian Edge": {
        "description": "Alternative edge detection using Laplacian operator.",
        "category": "Edge",
        "uniforms": {
            "strength": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1},
            "threshold": {"type": "float", "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.02},
            "invert": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
            "colored": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float strength;
            uniform float threshold;
            uniform float invert;
            uniform float colored;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                // Laplacian kernel
                vec3 sum = vec3(0.0);
                sum += texture(u_texture, v_uv + vec2(-texel.x, texel.y)).rgb;
                sum += texture(u_texture, v_uv + vec2(0, texel.y)).rgb;
                sum += texture(u_texture, v_uv + vec2(texel.x, texel.y)).rgb;
                sum += texture(u_texture, v_uv + vec2(-texel.x, 0)).rgb;
                sum -= 8.0 * texture(u_texture, v_uv).rgb;
                sum += texture(u_texture, v_uv + vec2(texel.x, 0)).rgb;
                sum += texture(u_texture, v_uv + vec2(-texel.x, -texel.y)).rgb;
                sum += texture(u_texture, v_uv + vec2(0, -texel.y)).rgb;
                sum += texture(u_texture, v_uv + vec2(texel.x, -texel.y)).rgb;

                vec3 edge = abs(sum) * strength;

                if(colored < 0.5) {
                    float gray = dot(edge, vec3(0.299, 0.587, 0.114));
                    edge = vec3(gray);
                }

                edge = smoothstep(vec3(threshold), vec3(threshold + 0.1), edge);

                if(invert > 0.5) {
                    edge = vec3(1.0) - edge;
                }

                f_color = vec4(edge, 1.0);
            }
        """
    },
    "Canny Edge": {
        "description": "Multi-stage edge detection with hysteresis.",
        "category": "Edge",
        "uniforms": {
            "low_threshold": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.02},
            "high_threshold": {"type": "float", "default": 0.3, "min": 0.1, "max": 1.0, "step": 0.02},
            "blur_radius": {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.5},
            "invert": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float low_threshold;
            uniform float high_threshold;
            uniform float blur_radius;
            uniform float invert;
            in vec2 v_uv;
            out vec4 f_color;

            float getLum(vec2 uv) {
                return dot(texture(u_texture, uv).rgb, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                // Optional Gaussian blur
                float blurred = 0.0;
                if(blur_radius > 0.0) {
                    float total = 0.0;
                    int r = int(blur_radius);
                    for(int x = -r; x <= r; x++) {
                        for(int y = -r; y <= r; y++) {
                            float weight = exp(-float(x*x + y*y) / (2.0 * blur_radius * blur_radius));
                            blurred += getLum(v_uv + vec2(x, y) * texel) * weight;
                            total += weight;
                        }
                    }
                    blurred /= total;
                } else {
                    blurred = getLum(v_uv);
                }

                // Sobel gradient
                float gx = -getLum(v_uv + vec2(-texel.x, texel.y)) - 2.0 * getLum(v_uv + vec2(-texel.x, 0)) - getLum(v_uv + vec2(-texel.x, -texel.y))
                          + getLum(v_uv + vec2(texel.x, texel.y)) + 2.0 * getLum(v_uv + vec2(texel.x, 0)) + getLum(v_uv + vec2(texel.x, -texel.y));
                float gy = -getLum(v_uv + vec2(-texel.x, texel.y)) - 2.0 * getLum(v_uv + vec2(0, texel.y)) - getLum(v_uv + vec2(texel.x, texel.y))
                          + getLum(v_uv + vec2(-texel.x, -texel.y)) + 2.0 * getLum(v_uv + vec2(0, -texel.y)) + getLum(v_uv + vec2(texel.x, -texel.y));

                float magnitude = sqrt(gx * gx + gy * gy);

                // Double threshold with hysteresis
                float edge = 0.0;
                if(magnitude >= high_threshold) {
                    edge = 1.0;
                } else if(magnitude >= low_threshold) {
                    // Check if connected to strong edge
                    edge = 0.5;
                }

                if(invert > 0.5) edge = 1.0 - edge;

                f_color = vec4(vec3(edge), 1.0);
            }
        """
    },
    "Double Edge Glow": {
        "description": "Inner and outer edge glow effect.",
        "category": "Edge",
        "uniforms": {
            "inner_color_r": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "inner_color_g": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "inner_color_b": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "outer_color_r": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "outer_color_g": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "outer_color_b": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "inner_width": {"type": "float", "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5},
            "outer_width": {"type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5},
            "intensity": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
        },
        "frag": """
            #version 330 core
            uniform sampler2D u_texture;
            uniform float inner_color_r, inner_color_g, inner_color_b;
            uniform float outer_color_r, outer_color_g, outer_color_b;
            uniform float inner_width, outer_width;
            uniform float intensity;
            in vec2 v_uv;
            out vec4 f_color;

            float getEdge(vec2 uv, vec2 texel) {
                float tl = dot(texture(u_texture, uv + vec2(-texel.x, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float t  = dot(texture(u_texture, uv + vec2(0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float tr = dot(texture(u_texture, uv + vec2(texel.x, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float l  = dot(texture(u_texture, uv + vec2(-texel.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
                float r  = dot(texture(u_texture, uv + vec2(texel.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
                float bl = dot(texture(u_texture, uv + vec2(-texel.x, -texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float b  = dot(texture(u_texture, uv + vec2(0, -texel.y)).rgb, vec3(0.299, 0.587, 0.114));
                float br = dot(texture(u_texture, uv + vec2(texel.x, -texel.y)).rgb, vec3(0.299, 0.587, 0.114));

                float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
                float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;

                return sqrt(gx*gx + gy*gy);
            }

            void main() {
                vec3 color = texture(u_texture, v_uv).rgb;
                vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));

                // Inner glow (erode then edge)
                float inner_edge = 0.0;
                if(inner_width > 0.0) {
                    for(float r = 1.0; r <= inner_width; r += 1.0) {
                        float weight = 1.0 - r / inner_width;
                        inner_edge += getEdge(v_uv, texel * r) * weight;
                    }
                    inner_edge /= inner_width;
                }

                // Outer glow (dilate then edge)
                float outer_edge = 0.0;
                if(outer_width > 0.0) {
                    for(float r = 1.0; r <= outer_width; r += 1.0) {
                        float weight = 1.0 - r / outer_width;
                        outer_edge += getEdge(v_uv, texel * r) * weight;
                    }
                    outer_edge /= outer_width;
                }

                vec3 inner_col = vec3(inner_color_r, inner_color_g, inner_color_b);
                vec3 outer_col = vec3(outer_color_r, outer_color_g, outer_color_b);

                vec3 glow = inner_col * inner_edge + outer_col * outer_edge;
                vec3 result = color + glow * intensity;

                f_color = vec4(result, 1.0);
            }
        """
    },
}

# AI Models for shader effect generation
AI_MODELS = {
    "Local (Built-in)": {
        "description": "Built-in pattern matching - works offline, instant results",
        "type": "local",
        "api_key": None
    },
    "GPT-4 Vision": {
        "description": "OpenAI's most capable model for understanding visual effects",
        "type": "openai",
        "model": "gpt-4-turbo",
        "api_key_env": "OPENAI_API_KEY"
    },
    "GPT-3.5 Turbo": {
        "description": "Fast and cost-effective OpenAI model",
        "type": "openai",
        "model": "gpt-3.5-turbo",
        "api_key_env": "OPENAI_API_KEY"
    },
    "Claude 3 Opus": {
        "description": "Anthropic's most powerful model for nuanced effects",
        "type": "anthropic",
        "model": "claude-3-opus-20240229",
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "Claude 3 Sonnet": {
        "description": "Balanced performance and speed from Anthropic",
        "type": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "Gemini Pro": {
        "description": "Google's multimodal AI for creative effects",
        "type": "google",
        "model": "gemini-pro",
        "api_key_env": "GOOGLE_API_KEY"
    }
}

# Effect keywords mapping for local AI interpretation
EFFECT_KEYWORDS = {
    # Edge/Outline effects
    "outline": {"shader": "Inverted Hull Outline", "params": {"outline_width": 2.0, "depth_influence": 1.0, "normal_influence": 1.0}},
    "inverted hull": {"shader": "Inverted Hull Outline", "params": {"outline_width": 2.5, "inner_outline": 0.4}},
    "hull": {"shader": "Inverted Hull Outline", "params": {"outline_width": 2.0, "silhouette_only": 0.0}},
    "silhouette": {"shader": "Inverted Hull Outline", "params": {"outline_width": 3.0, "silhouette_only": 1.0, "depth_influence": 1.0}},
    "contour": {"shader": "Inverted Hull Outline", "params": {"outline_width": 1.5, "normal_influence": 0.8}},
    "black outline": {"shader": "Inverted Hull Outline", "params": {"outline_width": 2.5, "outline_color_r": 0.0, "outline_color_g": 0.0, "outline_color_b": 0.0}},
    "white outline": {"shader": "Inverted Hull Outline", "params": {"outline_width": 2.0, "outline_color_r": 1.0, "outline_color_g": 1.0, "outline_color_b": 1.0}},
    "colored outline": {"shader": "Inverted Hull Outline", "params": {"outline_width": 2.0, "color_blend": 0.7}},
    "thick outline": {"shader": "Inverted Hull Outline", "params": {"outline_width": 5.0, "inner_outline": 0.2}},
    "thin outline": {"shader": "Inverted Hull Outline", "params": {"outline_width": 1.0, "outline_falloff": 1.5}},
    "edge": {"shader": "Sobel Edge", "params": {"edge_strength": 1.2, "threshold": 0.15}},
    "edges": {"shader": "Sobel Edge", "params": {"edge_strength": 1.2, "threshold": 0.15}},
    "sobel": {"shader": "Sobel Edge", "params": {"edge_strength": 1.5, "threshold": 0.1}},
    "sketch": {"shader": "Sketch Lines", "params": {"line_density": 1.5, "darkness": 1.2}},
    "pencil": {"shader": "Sketch Lines", "params": {"line_density": 1.2, "darkness": 1.0, "levels": 4.0}},
    "lineart": {"shader": "Inverted Hull Outline", "params": {"outline_width": 1.5, "inner_outline": 0.5, "outline_falloff": 1.2}},

    # Glow/Bloom effects
    "glow": {"shader": "Bloom", "params": {"intensity": 0.6, "threshold": 0.5, "radius": 15.0}},
    "bloom": {"shader": "Bloom", "params": {"intensity": 0.5, "threshold": 0.6, "radius": 12.0}},
    "dreamy": {"shader": "Dreamy Glow", "params": {"intensity": 0.5, "radius": 18.0, "softness": 0.4}},
    "soft": {"shader": "Dreamy Glow", "params": {"intensity": 0.3, "radius": 12.0, "softness": 0.5}},
    "ethereal": {"shader": "Dreamy Glow", "params": {"intensity": 0.6, "radius": 20.0, "threshold": 0.4}},
    "neon": {"shader": "Cyberpunk", "params": {"neon_intensity": 1.5, "glow": 0.7, "contrast": 1.3}},

    # Color effects
    "vintage": {"shader": "Vintage Film", "params": {"fade": 0.2, "warmth": 0.4, "grain": 0.1, "vignette": 0.5}},
    "retro": {"shader": "Vintage Film", "params": {"fade": 0.25, "warmth": 0.35, "saturation": 0.7}},
    "sepia": {"shader": "Sepia", "params": {"intensity": 0.8, "contrast": 1.1}},
    "noir": {"shader": "Noir", "params": {"contrast": 2.2, "brightness": -0.05, "vignette": 0.7}},
    "black and white": {"shader": "Noir", "params": {"contrast": 1.5, "brightness": 0.0}},
    "bw": {"shader": "Noir", "params": {"contrast": 1.5, "brightness": 0.0}},
    "monochrome": {"shader": "Noir", "params": {"contrast": 1.8, "grain": 0.05}},
    "cinematic": {"shader": "Cross Process", "params": {"intensity": 0.5, "contrast": 1.25, "saturation": 1.1}},
    "dramatic": {"shader": "Cross Process", "params": {"intensity": 0.7, "contrast": 1.4, "saturation": 1.2}},
    "warm": {"shader": "Color Grade", "params": {"temperature": 0.3, "saturation": 1.1}},
    "cool": {"shader": "Color Grade", "params": {"temperature": -0.3, "saturation": 1.0}},
    "cold": {"shader": "Color Grade", "params": {"temperature": -0.4, "tint": 0.1}},
    "vibrant": {"shader": "Color Grade", "params": {"saturation": 1.4, "vibrance": 0.3, "contrast": 1.1}},
    "saturated": {"shader": "Color Grade", "params": {"saturation": 1.5, "vibrance": 0.2}},
    "desaturated": {"shader": "Color Grade", "params": {"saturation": 0.4, "contrast": 1.1}},
    "muted": {"shader": "Color Grade", "params": {"saturation": 0.6, "contrast": 0.95}},
    "negative": {"shader": "Negative", "params": {"intensity": 1.0}},
    "invert": {"shader": "Negative", "params": {"intensity": 1.0}},

    # Blur effects
    "blur": {"shader": "Blur", "params": {"radius": 8.0, "quality": 2.0}},
    "blurry": {"shader": "Blur", "params": {"radius": 10.0, "quality": 2.0}},
    "bokeh": {"shader": "Depth of Field", "params": {"aperture": 0.5, "blur_amount": 12.0, "bokeh_shape": 1.0}},
    "depth of field": {"shader": "Depth of Field", "params": {"focus_point": 0.5, "aperture": 0.4, "blur_amount": 10.0}},
    "dof": {"shader": "Depth of Field", "params": {"focus_point": 0.5, "aperture": 0.4, "blur_amount": 8.0}},
    "tilt shift": {"shader": "Depth of Field", "params": {"focus_point": 0.5, "aperture": 0.6, "blur_amount": 15.0}},
    "motion blur": {"shader": "Motion Blur", "params": {"blur_amount": 5.0, "blur_samples": 12.0}},
    "motion": {"shader": "Motion Blur", "params": {"blur_amount": 4.0, "blur_samples": 10.0}},

    # Stylized effects
    "cartoon": {"shader": "Toon Shader", "params": {"levels": 4.0, "edge_threshold": 0.12, "saturation": 1.3}},
    "toon": {"shader": "Toon Shader", "params": {"levels": 4.0, "edge_threshold": 0.15, "edge_width": 1.0}},
    "cel": {"shader": "Toon Shader", "params": {"levels": 3.0, "edge_threshold": 0.1, "edge_width": 1.5}},
    "anime": {"shader": "Comic Book", "params": {"edge_strength": 1.0, "color_levels": 5.0, "saturation": 1.3}},
    "comic": {"shader": "Comic Book", "params": {"edge_strength": 1.2, "color_levels": 4.0, "saturation": 1.2}},
    "manga": {"shader": "Comic Book", "params": {"edge_strength": 1.5, "color_levels": 3.0, "saturation": 0.3}},
    "poster": {"shader": "Posterize", "params": {"levels": 4.0, "saturation": 1.3, "outline": 0.4}},
    "posterize": {"shader": "Posterize", "params": {"levels": 5.0, "saturation": 1.2}},
    "pop art": {"shader": "Posterize", "params": {"levels": 3.0, "saturation": 1.6, "outline": 0.6}},

    # Painting effects
    "oil": {"shader": "Oil Painting", "params": {"radius": 4.0, "sharpness": 10.0, "saturation": 1.2}},
    "oil painting": {"shader": "Oil Painting", "params": {"radius": 5.0, "sharpness": 8.0, "saturation": 1.3}},
    "paint": {"shader": "Oil Painting", "params": {"radius": 4.0, "sharpness": 12.0}},
    "watercolor": {"shader": "Watercolor", "params": {"edge_strength": 0.7, "blur_amount": 3.0, "wetness": 0.5}},
    "watercolour": {"shader": "Watercolor", "params": {"edge_strength": 0.7, "blur_amount": 3.0, "wetness": 0.5}},
    "impressionist": {"shader": "Oil Painting", "params": {"radius": 6.0, "sharpness": 6.0, "saturation": 1.4}},

    # Pixel/Retro effects
    "pixel": {"shader": "Pixelation", "params": {"pixel_size": 8.0}},
    "pixelate": {"shader": "Pixelation", "params": {"pixel_size": 6.0}},
    "pixelated": {"shader": "Pixelation", "params": {"pixel_size": 8.0}},
    "8bit": {"shader": "Pixelation", "params": {"pixel_size": 10.0, "color_depth": 8.0}},
    "8-bit": {"shader": "Pixelation", "params": {"pixel_size": 10.0, "color_depth": 8.0}},
    "retro game": {"shader": "Pixelation", "params": {"pixel_size": 6.0, "color_depth": 16.0}},
    "dither": {"shader": "Dithering (Retro)", "params": {"dither_size": 4.0, "color_levels": 4.0}},
    "dithering": {"shader": "Dithering (Retro)", "params": {"dither_size": 3.0, "color_levels": 5.0}},
    "halftone": {"shader": "Halftone", "params": {"dot_size": 4.0, "contrast": 1.2}},
    "dots": {"shader": "Halftone", "params": {"dot_size": 5.0, "angle": 0.5}},
    "mosaic": {"shader": "Mosaic", "params": {"tile_size": 15.0, "gap": 1.0}},
    "tiles": {"shader": "Mosaic", "params": {"tile_size": 20.0, "gap": 2.0}},
    "ascii": {"shader": "ASCII Art", "params": {"char_size": 8.0, "contrast": 1.3}},

    # Tech/Glitch effects
    "glitch": {"shader": "Glitch", "params": {"intensity": 0.05, "block_size": 0.04, "color_shift": 0.02}},
    "vhs": {"shader": "VHS Tape", "params": {"scan_intensity": 0.4, "noise_amount": 0.15, "tracking_error": 0.03}},
    "crt": {"shader": "CRT Monitor", "params": {"scanline_intensity": 0.4, "curvature": 0.1, "glow": 0.3}},
    "scanlines": {"shader": "CRT Monitor", "params": {"scanline_intensity": 0.5, "pixel_size": 1.0}},
    "chromatic": {"shader": "Chromatic Aberration", "params": {"strength": 0.008, "radial": 0.5}},
    "chromatic aberration": {"shader": "Chromatic Aberration", "params": {"strength": 0.01, "radial": 0.6}},
    "rgb split": {"shader": "Chromatic Aberration", "params": {"strength": 0.015, "radial": 0.3}},
    "cyberpunk": {"shader": "Cyberpunk", "params": {"neon_intensity": 1.3, "hue_shift": 0.5, "scanlines": 0.2}},
    "matrix": {"shader": "Night Vision", "params": {"brightness": 2.0, "scanlines": 0.3, "noise": 0.1}},

    # Film/Photo effects
    "film": {"shader": "Film Grain", "params": {"intensity": 0.15, "size": 1.5, "saturation": 0.9}},
    "grain": {"shader": "Film Grain", "params": {"intensity": 0.12, "size": 1.2}},
    "grainy": {"shader": "Film Grain", "params": {"intensity": 0.2, "size": 1.8}},
    "vignette": {"shader": "Vignette", "params": {"intensity": 0.7, "radius": 0.6, "softness": 0.4}},
    "sharp": {"shader": "Sharpen", "params": {"amount": 1.5, "radius": 1.0}},
    "sharpen": {"shader": "Sharpen", "params": {"amount": 1.2, "radius": 1.0}},
    "crisp": {"shader": "Sharpen", "params": {"amount": 1.8, "radius": 0.8}},
    "emboss": {"shader": "Emboss", "params": {"strength": 2.0, "blend": 0.5}},
    "embossed": {"shader": "Emboss", "params": {"strength": 2.5, "blend": 0.4}},
    "thermal": {"shader": "Thermal", "params": {"sensitivity": 1.0, "noise": 0.05}},
    "heat": {"shader": "Thermal", "params": {"sensitivity": 1.2, "palette": 0.0}},
    "infrared": {"shader": "Thermal", "params": {"sensitivity": 0.9, "palette": 1.0}},
    "night vision": {"shader": "Night Vision", "params": {"brightness": 2.5, "noise": 0.15, "scanlines": 0.3}},
    "nightvision": {"shader": "Night Vision", "params": {"brightness": 2.5, "noise": 0.15, "scanlines": 0.3}},

    # Lighting effects
    "pbr": {"shader": "PBR (Physically Based)", "params": {"metallic": 0.3, "roughness": 0.5}},
    "metallic": {"shader": "PBR (Physically Based)", "params": {"metallic": 0.8, "roughness": 0.3}},
    "metal": {"shader": "Anisotropic (Brushed Metal)", "params": {"anisotropy": 0.7, "metallic": 0.9}},
    "brushed metal": {"shader": "Anisotropic (Brushed Metal)", "params": {"anisotropy": 0.8, "rotation": 0.0}},
    "subsurface": {"shader": "Subsurface Scattering", "params": {"scatter_strength": 0.5, "translucency": 0.3}},
    "skin": {"shader": "Subsurface Scattering", "params": {"scatter_strength": 0.4, "scatter_color_r": 1.0, "scatter_color_g": 0.5}},
    "wax": {"shader": "Subsurface Scattering", "params": {"scatter_strength": 0.6, "translucency": 0.5}},
    "lambert": {"shader": "Lambert (Matte Diffuse)", "params": {"ambient": 0.2, "wrap": 0.3}},
    "matte": {"shader": "Lambert (Matte Diffuse)", "params": {"ambient": 0.25, "wrap": 0.2}},
    "specular": {"shader": "Blinn-Phong (Specular)", "params": {"specular_strength": 0.6, "shininess": 32.0}},
    "shiny": {"shader": "Blinn-Phong (Specular)", "params": {"specular_strength": 0.8, "shininess": 64.0}},
    "glossy": {"shader": "Blinn-Phong (Specular)", "params": {"specular_strength": 0.7, "shininess": 48.0}},

    # Environmental effects
    "fog": {"shader": "Volumetric Fog", "params": {"fog_density": 0.4, "light_scatter": 0.3}},
    "foggy": {"shader": "Volumetric Fog", "params": {"fog_density": 0.5, "noise_amount": 0.3}},
    "mist": {"shader": "Volumetric Fog", "params": {"fog_density": 0.3, "fog_color_r": 0.9, "fog_color_g": 0.95}},
    "haze": {"shader": "Volumetric Fog", "params": {"fog_density": 0.25, "fog_start": 0.3}},
    "water": {"shader": "Water/Waves", "params": {"wave_scale": 15.0, "refraction": 0.03, "caustics": 0.4}},
    "underwater": {"shader": "Water/Waves", "params": {"wave_scale": 10.0, "water_tint": 0.4, "caustics": 0.5}},
    "waves": {"shader": "Water/Waves", "params": {"wave_scale": 20.0, "wave_height": 0.03}},
    "ripple": {"shader": "Water/Waves", "params": {"ripple_freq": 15.0, "refraction": 0.04}},
    "noise": {"shader": "Perlin Noise", "params": {"noise_scale": 5.0, "blend_amount": 0.4}},
    "perlin": {"shader": "Perlin Noise", "params": {"noise_scale": 4.0, "octaves": 4.0}},
    "parallax": {"shader": "Parallax/Normal Map", "params": {"height_scale": 0.04, "normal_strength": 1.0}},
    "bump": {"shader": "Parallax/Normal Map", "params": {"height_scale": 0.03, "normal_strength": 0.8}},
    "normal map": {"shader": "Parallax/Normal Map", "params": {"normal_strength": 1.2, "specular": 0.4}},

    # Texture/Pattern Generation
    "voronoi": {"shader": "Voronoi Texture", "params": {"scale": 5.0, "randomness": 1.0}},
    "cells": {"shader": "Voronoi Texture", "params": {"scale": 8.0, "randomness": 0.8}},
    "cellular": {"shader": "Voronoi Texture", "params": {"scale": 6.0, "randomness": 0.9}},
    "musgrave": {"shader": "Musgrave Texture", "params": {"scale": 3.0, "octaves": 4.0, "lacunarity": 2.0}},
    "fractal": {"shader": "Musgrave Texture", "params": {"scale": 4.0, "octaves": 6.0, "dimension": 2.0}},
    "terrain": {"shader": "Musgrave Texture", "params": {"scale": 2.0, "octaves": 5.0, "gain": 0.5}},
    "wave texture": {"shader": "Wave Texture", "params": {"scale": 10.0, "distortion": 0.5, "rings": 0.0}},
    "ripples": {"shader": "Wave Texture", "params": {"scale": 15.0, "distortion": 0.3, "rings": 1.0}},
    "concentric": {"shader": "Wave Texture", "params": {"scale": 8.0, "rings": 1.0, "distortion": 0.2}},
    "brick": {"shader": "Brick Texture", "params": {"scale": 3.0, "mortar_size": 0.05, "offset": 0.5}},
    "bricks": {"shader": "Brick Texture", "params": {"scale": 4.0, "mortar_size": 0.03, "offset": 0.5}},
    "wall": {"shader": "Brick Texture", "params": {"scale": 2.5, "mortar_size": 0.04, "offset": 0.5}},
    "gradient": {"shader": "Gradient Texture", "params": {"gradient_type": 0.0}},
    "linear gradient": {"shader": "Gradient Texture", "params": {"gradient_type": 0.0}},
    "radial gradient": {"shader": "Gradient Texture", "params": {"gradient_type": 1.0}},
    "checker": {"shader": "Checker Texture", "params": {"scale": 8.0}},
    "checkerboard": {"shader": "Checker Texture", "params": {"scale": 10.0}},
    "grid": {"shader": "Checker Texture", "params": {"scale": 6.0}},
    "magic": {"shader": "Magic Texture", "params": {"scale": 5.0, "turbulence": 5.0}},
    "psychedelic": {"shader": "Magic Texture", "params": {"scale": 3.0, "turbulence": 8.0}},
    "trippy": {"shader": "Magic Texture", "params": {"scale": 4.0, "turbulence": 10.0}},

    # Color Manipulation
    "hsv": {"shader": "HSV Adjust", "params": {"hue_shift": 0.0, "saturation": 1.0, "value": 1.0}},
    "hue shift": {"shader": "HSV Adjust", "params": {"hue_shift": 0.5, "saturation": 1.0, "value": 1.0}},
    "color shift": {"shader": "HSV Adjust", "params": {"hue_shift": 0.3, "saturation": 1.1, "value": 1.0}},
    "color ramp": {"shader": "Color Ramp", "params": {"position": 0.5}},
    "ramp": {"shader": "Color Ramp", "params": {"position": 0.5}},
    "curves": {"shader": "RGB Curves", "params": {"red_mult": 1.0, "green_mult": 1.0, "blue_mult": 1.0}},
    "rgb curves": {"shader": "RGB Curves", "params": {"red_mult": 1.0, "green_mult": 1.0, "blue_mult": 1.0}},
    "blend": {"shader": "Blend Modes", "params": {"blend_mode": 0.0, "blend_amount": 0.5}},
    "multiply": {"shader": "Blend Modes", "params": {"blend_mode": 1.0, "blend_amount": 0.7}},
    "screen": {"shader": "Blend Modes", "params": {"blend_mode": 2.0, "blend_amount": 0.6}},
    "overlay": {"shader": "Blend Modes", "params": {"blend_mode": 3.0, "blend_amount": 0.5}},
    "selective color": {"shader": "Selective Color", "params": {"target_hue": 0.0, "hue_range": 0.1, "saturation_adjust": 1.5}},
    "color select": {"shader": "Selective Color", "params": {"target_hue": 0.0, "hue_range": 0.15, "saturation_adjust": 1.3}},
    "channel mixer": {"shader": "Channel Mixer", "params": {"red_red": 1.0, "red_green": 0.0, "red_blue": 0.0}},
    "channels": {"shader": "Channel Mixer", "params": {"red_red": 1.0, "green_green": 1.0, "blue_blue": 1.0}},

    # Distortion Effects
    "displacement": {"shader": "Displacement", "params": {"strength": 0.1, "scale": 5.0}},
    "displace": {"shader": "Displacement", "params": {"strength": 0.08, "scale": 4.0}},
    "spherize": {"shader": "Spherize", "params": {"strength": 0.5, "radius": 0.5}},
    "bulge": {"shader": "Spherize", "params": {"strength": 0.4, "radius": 0.6}},
    "fisheye": {"shader": "Spherize", "params": {"strength": 0.7, "radius": 0.8}},
    "twirl": {"shader": "Twirl", "params": {"angle": 3.14, "radius": 0.5}},
    "swirl": {"shader": "Twirl", "params": {"angle": 4.0, "radius": 0.6}},
    "spin": {"shader": "Twirl", "params": {"angle": 6.28, "radius": 0.4}},
    "wave distortion": {"shader": "Wave Distortion", "params": {"amplitude": 0.05, "frequency": 10.0}},
    "wavy": {"shader": "Wave Distortion", "params": {"amplitude": 0.03, "frequency": 8.0}},
    "wobble": {"shader": "Wave Distortion", "params": {"amplitude": 0.04, "frequency": 12.0}},
    "liquify": {"shader": "Liquify", "params": {"strength": 0.3, "scale": 3.0}},
    "melt": {"shader": "Liquify", "params": {"strength": 0.5, "scale": 2.0}},
    "warp": {"shader": "Liquify", "params": {"strength": 0.4, "scale": 4.0}},
    "kaleidoscope": {"shader": "Kaleidoscope", "params": {"segments": 6.0, "rotation": 0.0}},
    "mirror": {"shader": "Kaleidoscope", "params": {"segments": 2.0, "rotation": 0.0}},
    "symmetry": {"shader": "Kaleidoscope", "params": {"segments": 4.0, "rotation": 0.0}},

    # Advanced Lighting
    "ssao": {"shader": "Screen Space AO", "params": {"radius": 0.5, "intensity": 1.0, "samples": 16.0}},
    "ambient occlusion": {"shader": "Screen Space AO", "params": {"radius": 0.6, "intensity": 1.2, "samples": 24.0}},
    "ao": {"shader": "Screen Space AO", "params": {"radius": 0.4, "intensity": 0.8, "samples": 12.0}},
    "rim light": {"shader": "Rim Light", "params": {"rim_power": 3.0, "rim_intensity": 1.0}},
    "rim": {"shader": "Rim Light", "params": {"rim_power": 2.5, "rim_intensity": 0.8}},
    "fresnel": {"shader": "Rim Light", "params": {"rim_power": 4.0, "rim_intensity": 1.2}},
    "backlighting": {"shader": "Rim Light", "params": {"rim_power": 2.0, "rim_intensity": 1.5}},
    "matcap": {"shader": "Matcap", "params": {"rotation": 0.0}},
    "material capture": {"shader": "Matcap", "params": {"rotation": 0.0}},
    "gooch": {"shader": "Gooch Shading", "params": {"warm_intensity": 0.6, "cool_intensity": 0.4}},
    "technical": {"shader": "Gooch Shading", "params": {"warm_intensity": 0.5, "cool_intensity": 0.5}},
    "illustrative": {"shader": "Gooch Shading", "params": {"warm_intensity": 0.7, "cool_intensity": 0.3}},
    "xray": {"shader": "X-Ray", "params": {"edge_intensity": 1.0, "transparency": 0.5}},
    "x-ray": {"shader": "X-Ray", "params": {"edge_intensity": 1.0, "transparency": 0.5}},
    "transparent": {"shader": "X-Ray", "params": {"edge_intensity": 0.8, "transparency": 0.7}},

    # Artistic/Stylization
    "kuwahara": {"shader": "Kuwahara", "params": {"radius": 4.0}},
    "painterly": {"shader": "Kuwahara", "params": {"radius": 6.0}},
    "smooth painting": {"shader": "Kuwahara", "params": {"radius": 5.0}},
    "crosshatch": {"shader": "Cross Hatch", "params": {"density": 1.0, "thickness": 1.0}},
    "cross hatch": {"shader": "Cross Hatch", "params": {"density": 1.2, "thickness": 0.8}},
    "hatching": {"shader": "Cross Hatch", "params": {"density": 0.8, "thickness": 1.2}},
    "stipple": {"shader": "Stippling", "params": {"density": 1.0, "dot_size": 2.0}},
    "stippling": {"shader": "Stippling", "params": {"density": 1.2, "dot_size": 1.5}},
    "pointillism": {"shader": "Stippling", "params": {"density": 1.5, "dot_size": 2.5}},
    "woodcut": {"shader": "Woodcut", "params": {"contrast": 1.5, "line_width": 1.0}},
    "linocut": {"shader": "Woodcut", "params": {"contrast": 1.8, "line_width": 1.2}},
    "engraving": {"shader": "Woodcut", "params": {"contrast": 2.0, "line_width": 0.8}},
    "stained glass": {"shader": "Stained Glass", "params": {"cell_size": 20.0, "edge_width": 2.0}},
    "glass": {"shader": "Stained Glass", "params": {"cell_size": 25.0, "edge_width": 1.5}},
    "mosaic art": {"shader": "Stained Glass", "params": {"cell_size": 15.0, "edge_width": 3.0}},
    "palette swap": {"shader": "Palette Swap", "params": {"palette_index": 0.0}},
    "recolor": {"shader": "Palette Swap", "params": {"palette_index": 1.0}},
    "color palette": {"shader": "Palette Swap", "params": {"palette_index": 2.0}},

    # Compositing Effects
    "lens flare": {"shader": "Lens Flare", "params": {"intensity": 0.8, "threshold": 0.7}},
    "flare": {"shader": "Lens Flare", "params": {"intensity": 0.6, "threshold": 0.6}},
    "sun flare": {"shader": "Lens Flare", "params": {"intensity": 1.0, "threshold": 0.8}},
    "light streaks": {"shader": "Light Streaks", "params": {"intensity": 0.7, "threshold": 0.6, "streak_length": 0.3}},
    "streaks": {"shader": "Light Streaks", "params": {"intensity": 0.5, "streak_length": 0.4}},
    "anamorphic": {"shader": "Light Streaks", "params": {"intensity": 0.8, "streak_length": 0.5}},
    "god rays": {"shader": "God Rays", "params": {"intensity": 0.8, "decay": 0.95, "samples": 50.0}},
    "sunbeams": {"shader": "God Rays", "params": {"intensity": 1.0, "decay": 0.9, "samples": 64.0}},
    "volumetric light": {"shader": "God Rays", "params": {"intensity": 0.6, "decay": 0.97, "samples": 40.0}},
    "lens dirt": {"shader": "Lens Dirt", "params": {"intensity": 0.5, "threshold": 0.6}},
    "dirty lens": {"shader": "Lens Dirt", "params": {"intensity": 0.7, "threshold": 0.5}},
    "dust": {"shader": "Lens Dirt", "params": {"intensity": 0.4, "threshold": 0.7}},
    "bokeh shapes": {"shader": "Bokeh", "params": {"size": 10.0, "threshold": 0.7, "intensity": 0.8}},
    "hexagonal bokeh": {"shader": "Bokeh", "params": {"size": 12.0, "threshold": 0.6, "shape": 6.0}},
    "circular bokeh": {"shader": "Bokeh", "params": {"size": 8.0, "threshold": 0.8, "shape": 0.0}},

    # Edge Effects
    "dilate": {"shader": "Dilate", "params": {"radius": 2.0}},
    "expand": {"shader": "Dilate", "params": {"radius": 3.0}},
    "grow": {"shader": "Dilate", "params": {"radius": 1.0}},
    "erode": {"shader": "Erode", "params": {"radius": 2.0}},
    "shrink": {"shader": "Erode", "params": {"radius": 3.0}},
    "contract": {"shader": "Erode", "params": {"radius": 1.0}},
    "laplacian": {"shader": "Laplacian Edge", "params": {"intensity": 1.0}},
    "laplacian edge": {"shader": "Laplacian Edge", "params": {"intensity": 1.2}},
    "second derivative": {"shader": "Laplacian Edge", "params": {"intensity": 0.8}},
    "canny": {"shader": "Canny Edge", "params": {"low_threshold": 0.1, "high_threshold": 0.3}},
    "canny edge": {"shader": "Canny Edge", "params": {"low_threshold": 0.15, "high_threshold": 0.4}},
    "precise edges": {"shader": "Canny Edge", "params": {"low_threshold": 0.08, "high_threshold": 0.25}},
    "double edge": {"shader": "Double Edge Glow", "params": {"inner_width": 1.0, "outer_width": 3.0, "glow_intensity": 0.8}},
    "edge glow": {"shader": "Double Edge Glow", "params": {"inner_width": 1.5, "outer_width": 4.0, "glow_intensity": 1.0}},
    "neon edge": {"shader": "Double Edge Glow", "params": {"inner_width": 2.0, "outer_width": 5.0, "glow_intensity": 1.2}},

    # Intensity modifiers (applied as multipliers)
    "strong": {"multiplier": 1.5},
    "intense": {"multiplier": 1.6},
    "heavy": {"multiplier": 1.7},
    "extreme": {"multiplier": 2.0},
    "light": {"multiplier": 0.5},
    "subtle": {"multiplier": 0.4},
    "slight": {"multiplier": 0.3},
    "gentle": {"multiplier": 0.5},
    "mild": {"multiplier": 0.6},
    "thick": {"multiplier": 1.4},
    "thin": {"multiplier": 0.6},
    "bold": {"multiplier": 1.3},
    "faint": {"multiplier": 0.4},
}

# Quick preset prompts
QUICK_AI_PROMPTS = {
    "Outline": "Apply edge detection with bold black outlines",
    "Glow": "Add a soft dreamy glow effect with bloom",
    "Vintage": "Apply vintage film look with grain and warm tones",
    "Dramatic": "Create dramatic cinematic look with high contrast and vignette",
    "Cartoon": "Make it look like a cartoon with cel-shading and outlines",
    "Neon": "Add neon glow cyberpunk style effect",
    "Painterly": "Apply oil painting artistic effect",
    "Noir": "Black and white film noir with high contrast",
}

# Simple passthrough shader for blitting textures to screen/FBO
PASSTHROUGH_SHADER = """
#version 330 core
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(u_texture, v_uv);
}
"""

# Blend modes for layer compositing
BLEND_MODES = {
    "normal": 0, "multiply": 1, "screen": 2, "overlay": 3,
    "soft_light": 4, "hard_light": 5, "add": 6, "subtract": 7,
    "difference": 8, "exclusion": 9, "color_dodge": 10, "color_burn": 11,
}
BLEND_MODE_NAMES = list(BLEND_MODES.keys())

# GPU compositing shader for layer blending
COMPOSITING_SHADER = """
#version 330 core
uniform sampler2D u_base;
uniform sampler2D u_overlay;
uniform float u_opacity;
uniform int u_blend_mode;
in vec2 v_uv;
out vec4 fragColor;

vec3 blend_normal(vec3 b, vec3 l)    { return l; }
vec3 blend_multiply(vec3 b, vec3 l)  { return b * l; }
vec3 blend_screen(vec3 b, vec3 l)    { return 1.0 - (1.0 - b) * (1.0 - l); }
vec3 blend_overlay(vec3 b, vec3 l) {
    return mix(2.0 * b * l, 1.0 - 2.0 * (1.0 - b) * (1.0 - l), step(0.5, b));
}
vec3 blend_soft_light(vec3 b, vec3 l) {
    return mix(
        2.0 * b * l + b * b * (1.0 - 2.0 * l),
        sqrt(b) * (2.0 * l - 1.0) + 2.0 * b * (1.0 - l),
        step(0.5, l)
    );
}
vec3 blend_hard_light(vec3 b, vec3 l) {
    return mix(2.0 * b * l, 1.0 - 2.0 * (1.0 - b) * (1.0 - l), step(0.5, l));
}
vec3 blend_add(vec3 b, vec3 l)        { return min(b + l, 1.0); }
vec3 blend_subtract(vec3 b, vec3 l)   { return max(b - l, 0.0); }
vec3 blend_difference(vec3 b, vec3 l) { return abs(b - l); }
vec3 blend_exclusion(vec3 b, vec3 l)  { return b + l - 2.0 * b * l; }
vec3 blend_color_dodge(vec3 b, vec3 l) {
    return min(b / max(1.0 - l, 0.001), 1.0);
}
vec3 blend_color_burn(vec3 b, vec3 l) {
    return 1.0 - min((1.0 - b) / max(l, 0.001), 1.0);
}

void main() {
    vec4 base = texture(u_base, v_uv);
    vec4 over = texture(u_overlay, v_uv);
    vec3 blended;
    if      (u_blend_mode == 0)  blended = blend_normal(base.rgb, over.rgb);
    else if (u_blend_mode == 1)  blended = blend_multiply(base.rgb, over.rgb);
    else if (u_blend_mode == 2)  blended = blend_screen(base.rgb, over.rgb);
    else if (u_blend_mode == 3)  blended = blend_overlay(base.rgb, over.rgb);
    else if (u_blend_mode == 4)  blended = blend_soft_light(base.rgb, over.rgb);
    else if (u_blend_mode == 5)  blended = blend_hard_light(base.rgb, over.rgb);
    else if (u_blend_mode == 6)  blended = blend_add(base.rgb, over.rgb);
    else if (u_blend_mode == 7)  blended = blend_subtract(base.rgb, over.rgb);
    else if (u_blend_mode == 8)  blended = blend_difference(base.rgb, over.rgb);
    else if (u_blend_mode == 9)  blended = blend_exclusion(base.rgb, over.rgb);
    else if (u_blend_mode == 10) blended = blend_color_dodge(base.rgb, over.rgb);
    else if (u_blend_mode == 11) blended = blend_color_burn(base.rgb, over.rgb);
    else                         blended = blend_normal(base.rgb, over.rgb);
    vec3 result = mix(base.rgb, blended, u_opacity * over.a);
    fragColor = vec4(result, 1.0);
}
"""

VERTEX_SHADER = """
#version 330 core
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
"""

# 3D Vertex shader with model-view-projection
VERTEX_SHADER_3D = """
#version 330 core
in vec3 in_vert;
in vec2 in_uv;
in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec2 v_uv;
out vec3 v_normal;
out vec3 v_position;

void main() {
    vec4 world_pos = u_model * vec4(in_vert, 1.0);
    gl_Position = u_projection * u_view * world_pos;
    v_uv = in_uv;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_position = world_pos.xyz;
}
"""

# 3D Fragment shader that samples texture like 2D but with lighting
FRAG_SHADER_3D_WRAPPER = """
#version 330 core
uniform sampler2D u_texture;
{uniforms}

in vec2 v_uv;
in vec3 v_normal;
in vec3 v_position;

out vec4 f_color;

void main() {{
    // First apply the 2D shader effect
    {effect_code}

    // Then apply basic lighting
    vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(v_normal);
    float diff = max(dot(normal, light_dir), 0.0);
    float ambient = 0.3;
    float lighting = ambient + diff * 0.7;

    f_color.rgb *= lighting;
}}
"""


def load_obj(filepath):
    """Load a simple OBJ file and return vertices, uvs, normals, and indices."""
    vertices = []
    uvs = []
    normals = []

    temp_verts = []
    temp_uvs = []
    temp_normals = []
    faces = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'v':
                    temp_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vt':
                    temp_uvs.append([float(parts[1]), float(parts[2])])
                elif parts[0] == 'vn':
                    temp_normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f':
                    face = []
                    for p in parts[1:]:
                        indices = p.split('/')
                        v_idx = int(indices[0]) - 1
                        t_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else 0
                        n_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else 0
                        face.append((v_idx, t_idx, n_idx))
                    faces.append(face)

        # Triangulate and build vertex data
        for face in faces:
            # Triangulate (fan triangulation for convex polygons)
            for i in range(1, len(face) - 1):
                for idx in [face[0], face[i], face[i + 1]]:
                    v_idx, t_idx, n_idx = idx
                    vertices.extend(temp_verts[v_idx])
                    if temp_uvs:
                        uvs.extend(temp_uvs[t_idx] if t_idx < len(temp_uvs) else [0, 0])
                    else:
                        uvs.extend([0, 0])
                    if temp_normals:
                        normals.extend(temp_normals[n_idx] if n_idx < len(temp_normals) else [0, 1, 0])
                    else:
                        normals.extend([0, 1, 0])

        return np.array(vertices, dtype=np.float32), np.array(uvs, dtype=np.float32), np.array(normals, dtype=np.float32)
    except Exception as e:
        print(f"Error loading OBJ: {e}")
        return None, None, None


def create_sphere(radius=1.0, segments=32, rings=16):
    """Create a UV sphere for 3D preview."""
    vertices = []
    uvs = []
    normals = []

    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        for seg in range(segments + 1):
            theta = 2 * math.pi * seg / segments

            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)

            nx, ny, nz = x/radius, y/radius, z/radius
            u = seg / segments
            v = ring / rings

            vertices.extend([x, y, z])
            normals.extend([nx, ny, nz])
            uvs.extend([u, v])

    # Generate indices for triangle strip
    indices = []
    for ring in range(rings):
        for seg in range(segments):
            curr = ring * (segments + 1) + seg
            next_ring = (ring + 1) * (segments + 1) + seg

            indices.extend([curr, next_ring, curr + 1])
            indices.extend([curr + 1, next_ring, next_ring + 1])

    # Convert indexed to non-indexed
    verts_arr = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    uvs_arr = np.array(uvs, dtype=np.float32).reshape(-1, 2)
    norms_arr = np.array(normals, dtype=np.float32).reshape(-1, 3)

    out_verts = []
    out_uvs = []
    out_norms = []

    for idx in indices:
        out_verts.extend(verts_arr[idx])
        out_uvs.extend(uvs_arr[idx])
        out_norms.extend(norms_arr[idx])

    return np.array(out_verts, dtype=np.float32), np.array(out_uvs, dtype=np.float32), np.array(out_norms, dtype=np.float32)


def create_cube():
    """Create a simple cube for 3D preview."""
    # Vertices for a unit cube centered at origin
    v = [
        # Front face
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        # Back face
        [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
        # Top face
        [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
        # Bottom face
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
        # Right face
        [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5],
        # Left face
        [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
    ]

    uv = [
        [0, 0], [1, 0], [1, 1], [0, 1],
        [1, 0], [1, 1], [0, 1], [0, 0],
        [0, 1], [0, 0], [1, 0], [1, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [1, 0], [1, 1], [0, 1], [0, 0],
        [0, 0], [1, 0], [1, 1], [0, 1],
    ]

    n = [
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
        [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
    ]

    # Convert quads to triangles
    indices = []
    for i in range(6):
        base = i * 4
        indices.extend([base, base+1, base+2, base, base+2, base+3])

    vertices = []
    uvs = []
    normals = []
    for idx in indices:
        vertices.extend(v[idx])
        uvs.extend(uv[idx])
        normals.extend(n[idx])

    return np.array(vertices, dtype=np.float32), np.array(uvs, dtype=np.float32), np.array(normals, dtype=np.float32)


class ShaderCanvas(QOpenGLWidget):
    """OpenGL widget supporting both 2D image and 3D model rendering."""

    # Signals
    colorPicked = QtCore.pyqtSignal(int, int, int, int)  # RGBA color picked
    textureLoaded = QtCore.pyqtSignal(str)  # Emitted when a texture is loaded (path)

    def __init__(self):
        super().__init__()
        self.program = None
        self.program_3d = None
        self.vao = None
        self.vbo = None
        self.vao_3d = None
        self.vbo_3d = None
        self.texture_id = None
        self.image_data = None
        self.image_size = (256, 256)
        self.current_preset = "Original"
        self.params = {}
        self.image_path = None

        # 3D mode properties
        self.mode_3d = False
        self.model_vertices = None
        self.model_vertex_count = 0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.auto_rotate = False
        self.rotation_speed = 1.0
        self.zoom = 2.5
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Multiple 3D objects support
        self.objects_3d = []  # List of {"vao": ..., "vbo": ..., "count": ..., "transform": ...}

        # Skybox/HDRI
        self.skybox_texture = None
        self.skybox_enabled = False

        # View modes
        self.wireframe_mode = False
        self.show_normals = False

        # Mouse interaction
        self.mouse_pressed = False
        self.last_mouse_pos = None
        self.mouse_button = None
        self.color_picker_mode = False

        # Multiple light sources for 3D
        self.lights = [
            {"pos": [2.0, 2.0, 2.0], "color": [1.0, 1.0, 1.0], "intensity": 1.0},  # Key light
            {"pos": [-2.0, 1.0, 1.0], "color": [0.3, 0.5, 1.0], "intensity": 0.5},  # Fill light
            {"pos": [0.0, -1.0, 2.0], "color": [1.0, 0.8, 0.6], "intensity": 0.3},  # Rim light
        ]
        self.ambient_intensity = 0.2
        self.specular_power = 32.0
        self.specular_intensity = 0.5

        # Undo/Redo system
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 50

        # Animation timer
        self.animation_timer = QtCore.QTimer()
        self.animation_timer.timeout.connect(self._animate)
        self.animation_timer.setInterval(16)  # ~60 FPS
        self.initialized = False
        self.setMinimumSize(400, 400)

        # Animation keyframes
        self.keyframes = []  # List of {"time": ..., "rotation": ..., "zoom": ...}
        self.keyframe_time = 0.0
        self.playing_keyframes = False
        self.keyframe_timer = QtCore.QTimer()
        self.keyframe_timer.timeout.connect(self._update_keyframe)
        self.keyframe_timer.setInterval(16)

        # Post-processing shader stack
        self.post_process_stack = []  # List of shader names to chain
        self.post_programs = {}  # Compiled post-processing programs
        self.fbo = None
        self.fbo_texture = None
        self.fbo_size = (0, 0)

        # LUT and overlays
        self.current_lut = None
        self.overlay_texture = None
        self.overlay_blend_mode = "multiply"
        self.overlay_opacity = 0.5

        # GIF animation support for 2D mode
        self.gif_frames = []  # List of numpy arrays for each frame
        self.gif_frame_durations = []  # Duration of each frame in ms
        self.gif_current_frame = 0
        self.gif_playing = False
        self.gif_timer = QtCore.QTimer()
        self.gif_timer.timeout.connect(self._advance_gif_frame)
        self.is_gif = False

        # Shader layer system v2 (compositing model)
        self.shader_layers = []           # Layer data list from panel signal
        self._prev_layer_data = []        # Previous frame's data for dirty comparison
        self.layer_programs = {}          # Cached compiled shader programs
        self.layer_cache = {}             # {layer_id: {'fbo': int, 'texture': int, 'dirty': True}}
        self.accum_fbo_a = None           # Accumulation FBO A (ping-pong)
        self.accum_fbo_a_texture = None
        self.accum_fbo_b = None           # Accumulation FBO B
        self.accum_fbo_b_texture = None
        self.accum_fbo_size = (0, 0)
        self.compositing_program = None
        self.passthrough_program = None  # Simple texture blit shader

        # Bake chain (interactive multi-pass shader chaining)
        self._bake_chain = []            # List of {'shader': name, 'params': {...}} for each baked pass
        self._original_image_data = None # numpy array of original image (for reset)
        self._original_image_path = None # path to original image (for reset)

        # Enable mouse tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # Set OpenGL format
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
        self.setFormat(fmt)

    def initializeGL(self):
        print("Initializing OpenGL...")
        print(f"OpenGL Version: {GL.glGetString(GL.GL_VERSION).decode()}")
        print(f"GLSL Version: {GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode()}")

        # Create VAO
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Create VBO with quad vertices
        vertices = np.array([
            # x, y, u, v
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32)

        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

        # Create placeholder texture (checkerboard)
        self._create_placeholder_texture()

        # Initialize shader params
        self._init_params()

        # Compile shader
        self._compile_shader()

        self.initialized = True
        print("OpenGL initialized successfully!")

    def _create_placeholder_texture(self):
        """Create a checkerboard placeholder texture."""
        size = 256
        data = np.zeros((size, size, 4), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                if (x // 32 + y // 32) % 2 == 0:
                    data[y, x] = [80, 80, 80, 255]
                else:
                    data[y, x] = [50, 50, 50, 255]

        self.image_size = (size, size)
        self._upload_texture(data)

    def _upload_texture(self, data):
        """Upload numpy array as texture."""
        if self.texture_id is not None:
            GL.glDeleteTextures([self.texture_id])

        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        height, width = data.shape[:2]
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)

        print(f"Texture uploaded: {width}x{height}")

    def _create_fbo(self, width, height):
        """Create a framebuffer object for post-processing."""
        if self.fbo is not None and self.fbo_size == (width, height):
            return  # Already correct size

        # Delete old FBO
        if self.fbo is not None:
            GL.glDeleteFramebuffers(1, [self.fbo])
            GL.glDeleteTextures([self.fbo_texture])

        # Create FBO
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        # Create texture for FBO
        self.fbo_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                       GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        # Attach texture to FBO
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self.fbo_texture, 0)

        # Check FBO status
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            print("FBO creation failed!")
            self.fbo = None
            return

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.fbo_size = (width, height)
        print(f"FBO created: {width}x{height}")

    def _compile_post_shader(self, shader_name):
        """Compile a shader for post-processing use."""
        if shader_name in self.post_programs:
            return self.post_programs[shader_name]

        if shader_name not in SHADERS:
            return None

        # Compile vertex shader
        vert_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vert_shader, VERTEX_SHADER)
        GL.glCompileShader(vert_shader)
        if not GL.glGetShaderiv(vert_shader, GL.GL_COMPILE_STATUS):
            return None

        # Compile fragment shader
        frag_src = SHADERS[shader_name]["frag"]
        frag_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(frag_shader, frag_src)
        GL.glCompileShader(frag_shader)
        if not GL.glGetShaderiv(frag_shader, GL.GL_COMPILE_STATUS):
            GL.glDeleteShader(vert_shader)
            return None

        # Link program
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vert_shader)
        GL.glAttachShader(program, frag_shader)
        GL.glLinkProgram(program)

        GL.glDeleteShader(vert_shader)
        GL.glDeleteShader(frag_shader)

        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            GL.glDeleteProgram(program)
            return None

        self.post_programs[shader_name] = program
        return program

    def add_post_effect(self, shader_name):
        """Add a shader to the post-processing stack."""
        if shader_name in SHADERS and shader_name not in self.post_process_stack:
            self.post_process_stack.append(shader_name)
            self.makeCurrent()
            self._compile_post_shader(shader_name)
            self.update()

    def remove_post_effect(self, shader_name):
        """Remove a shader from the post-processing stack."""
        if shader_name in self.post_process_stack:
            self.post_process_stack.remove(shader_name)
            self.update()

    def clear_post_effects(self):
        """Clear all post-processing effects."""
        self.post_process_stack.clear()
        self.update()

    def get_post_effect_params(self, shader_name):
        """Get the parameters for a post-processing shader."""
        if shader_name in SHADERS:
            return SHADERS[shader_name].get("uniforms", {})
        return {}

    def set_shader_layers(self, layers_data):
        """Set the shader layer stack for multi-layer rendering."""
        # Just store the data - shader compilation will happen lazily during paintGL
        self.shader_layers = layers_data

    def bake_current_pass(self):
        """Render the current shader effect and commit it as the new source texture.
        Records the shader+params in the bake chain for later batch processing.
        Returns the chain entry dict, or None on failure.
        """
        import numpy as np
        if self.program is None or self.texture_id is None:
            return None

        self.makeCurrent()
        w, h = self.image_size

        # Render current effect
        image_data = self._export_2d_single(self.program, self.params, w, h)
        if image_data is None:
            return None

        # Record this pass in the chain
        entry = {'shader': self.current_preset, 'params': dict(self.params)}
        self._bake_chain.append(entry)

        # Upload rendered result as new source texture
        self._upload_texture(np.ascontiguousarray(np.flipud(image_data)))

        # Reset to Original shader so viewport shows baked result cleanly
        self.current_preset = "Original"
        self._init_params()
        self._compile_shader()
        self.update()

        return entry

    def reset_chain(self):
        """Reset to the original image and clear the bake chain."""
        self._bake_chain.clear()
        if self._original_image_path:
            self.load_texture(self._original_image_path)
            self.current_preset = "Original"
            self._init_params()
            self._compile_shader()
            self.update()

    def _compile_layer_shader(self, shader_name):
        """Compile a shader for use in layered rendering."""
        # If already compiled for layers, return cached
        if shader_name in self.layer_programs:
            return self.layer_programs[shader_name]

        if shader_name not in SHADERS:
            return None

        try:
            frag_src = SHADERS[shader_name]["frag"]

            vert_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
            GL.glShaderSource(vert_shader, VERTEX_SHADER)
            GL.glCompileShader(vert_shader)
            if not GL.glGetShaderiv(vert_shader, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(vert_shader).decode()
                print(f"Vertex shader error for layer {shader_name}: {error}")
                GL.glDeleteShader(vert_shader)
                return None

            frag_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
            GL.glShaderSource(frag_shader, frag_src)
            GL.glCompileShader(frag_shader)
            if not GL.glGetShaderiv(frag_shader, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(frag_shader).decode()
                print(f"Fragment shader error for layer {shader_name}: {error}")
                GL.glDeleteShader(vert_shader)
                GL.glDeleteShader(frag_shader)
                return None

            program = GL.glCreateProgram()
            GL.glAttachShader(program, vert_shader)
            GL.glAttachShader(program, frag_shader)

            # Bind attribute locations to match VAO setup
            GL.glBindAttribLocation(program, 0, "in_vert")
            GL.glBindAttribLocation(program, 1, "in_uv")

            GL.glLinkProgram(program)

            GL.glDeleteShader(vert_shader)
            GL.glDeleteShader(frag_shader)

            if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
                error = GL.glGetProgramInfoLog(program).decode()
                print(f"Program link error for layer {shader_name}: {error}")
                GL.glDeleteProgram(program)
                return None

            self.layer_programs[shader_name] = program
            return program

        except Exception as e:
            import traceback
            print(f"Exception compiling layer shader {shader_name}: {e}")
            traceback.print_exc()
            return None

    def _init_params(self):
        """Initialize parameters from shader definition."""
        shader_def = SHADERS.get(self.current_preset, {})
        uniforms = shader_def.get("uniforms", {})
        self.params = {}
        for name, props in uniforms.items():
            self.params[name] = props.get("default", 0.0)

    def _compile_shader(self):
        """Compile the current shader program."""
        if self.program is not None:
            try:
                GL.glDeleteProgram(self.program)
            except Exception as e:
                print(f"Warning: Could not delete old program {self.program}: {e}")
            self.program = None

        # Compile vertex shader
        vert_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vert_shader, VERTEX_SHADER)
        GL.glCompileShader(vert_shader)
        if not GL.glGetShaderiv(vert_shader, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(vert_shader).decode()
            print(f"Vertex shader error: {error}")
            return

        # Compile fragment shader
        frag_src = SHADERS[self.current_preset]["frag"]
        frag_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(frag_shader, frag_src)
        GL.glCompileShader(frag_shader)
        if not GL.glGetShaderiv(frag_shader, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(frag_shader).decode()
            print(f"Fragment shader error: {error}")
            return

        # Link program
        self.program = GL.glCreateProgram()
        GL.glAttachShader(self.program, vert_shader)
        GL.glAttachShader(self.program, frag_shader)
        GL.glLinkProgram(self.program)

        if not GL.glGetProgramiv(self.program, GL.GL_LINK_STATUS):
            error = GL.glGetProgramInfoLog(self.program).decode()
            print(f"Program link error: {error}")
            return

        GL.glDeleteShader(vert_shader)
        GL.glDeleteShader(frag_shader)

        print(f"Compiled shader: {self.current_preset}")

        # Setup vertex attributes
        GL.glUseProgram(self.program)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)

        # in_vert attribute (location 0)
        pos_loc = GL.glGetAttribLocation(self.program, "in_vert")
        if pos_loc >= 0:
            GL.glEnableVertexAttribArray(pos_loc)
            GL.glVertexAttribPointer(pos_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)

        # in_uv attribute (location 1)
        uv_loc = GL.glGetAttribLocation(self.program, "in_uv")
        if uv_loc >= 0:
            GL.glEnableVertexAttribArray(uv_loc)
            GL.glVertexAttribPointer(uv_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 16,
                                     GL.ctypes.c_void_p(8))

    def _compile_shader_3d(self):
        """Compile the 3D shader program with current effect."""
        if self.program_3d is not None:
            GL.glDeleteProgram(self.program_3d)

        # Compile vertex shader
        vert_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vert_shader, VERTEX_SHADER_3D)
        GL.glCompileShader(vert_shader)
        if not GL.glGetShaderiv(vert_shader, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(vert_shader).decode()
            print(f"3D Vertex shader error: {error}")
            return

        # Create 3D fragment shader with the current effect
        # We need to adapt the 2D fragment shader for 3D with lighting
        shader_def = SHADERS.get(self.current_preset, SHADERS["Original"])
        frag_2d = shader_def["frag"]

        # Build a 3D-compatible fragment shader with multiple light sources
        frag_src = f"""
#version 330 core
uniform sampler2D u_texture;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

// Light uniforms
uniform vec3 u_light1_pos;
uniform vec3 u_light1_color;
uniform float u_light1_intensity;
uniform vec3 u_light2_pos;
uniform vec3 u_light2_color;
uniform float u_light2_intensity;
uniform vec3 u_light3_pos;
uniform vec3 u_light3_color;
uniform float u_light3_intensity;
uniform float u_ambient_intensity;
uniform float u_specular_power;
uniform float u_specular_intensity;
uniform vec3 u_camera_pos;

in vec2 v_uv;
in vec3 v_normal;
in vec3 v_position;

out vec4 f_color;

// Uniforms from shader
{self._extract_uniforms(frag_2d)}

vec3 calculateLight(vec3 lightPos, vec3 lightColor, float intensity, vec3 normal, vec3 fragPos, vec3 viewDir) {{
    if(intensity <= 0.0) return vec3(0.0);

    vec3 lightDir = normalize(lightPos - fragPos);

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * intensity;

    // Specular (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), u_specular_power);
    vec3 specular = spec * lightColor * u_specular_intensity * intensity;

    // Attenuation
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

    return (diffuse + specular) * attenuation;
}}

void main() {{
    // Sample texture with shader effect
    vec4 base_color = texture(u_texture, v_uv);

    // Apply basic color adjustments based on shader params
    vec3 color = base_color.rgb;

    {self._generate_3d_effect_code()}

    // Calculate lighting
    vec3 normal = normalize(v_normal);
    vec3 viewDir = normalize(u_camera_pos - v_position);

    // Ambient
    vec3 ambient = color * u_ambient_intensity;

    // Add contributions from all lights
    vec3 lighting = ambient;
    lighting += calculateLight(u_light1_pos, u_light1_color, u_light1_intensity, normal, v_position, viewDir) * color;
    lighting += calculateLight(u_light2_pos, u_light2_color, u_light2_intensity, normal, v_position, viewDir) * color;
    lighting += calculateLight(u_light3_pos, u_light3_color, u_light3_intensity, normal, v_position, viewDir) * color;

    f_color = vec4(clamp(lighting, 0.0, 1.0), base_color.a);
}}
"""

        frag_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(frag_shader, frag_src)
        GL.glCompileShader(frag_shader)
        if not GL.glGetShaderiv(frag_shader, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(frag_shader).decode()
            print(f"3D Fragment shader error: {error}")
            print("Shader source:")
            for i, line in enumerate(frag_src.split('\n'), 1):
                print(f"{i}: {line}")
            return

        # Link program
        self.program_3d = GL.glCreateProgram()
        GL.glAttachShader(self.program_3d, vert_shader)
        GL.glAttachShader(self.program_3d, frag_shader)
        GL.glLinkProgram(self.program_3d)

        if not GL.glGetProgramiv(self.program_3d, GL.GL_LINK_STATUS):
            error = GL.glGetProgramInfoLog(self.program_3d).decode()
            print(f"3D Program link error: {error}")
            return

        GL.glDeleteShader(vert_shader)
        GL.glDeleteShader(frag_shader)

        print(f"Compiled 3D shader: {self.current_preset}")

    def _extract_uniforms(self, frag_shader):
        """Extract uniform declarations from a fragment shader."""
        lines = []
        for line in frag_shader.split('\n'):
            stripped = line.strip()
            if stripped.startswith('uniform float'):
                # Skip texture uniform
                if 'u_texture' not in stripped:
                    lines.append(stripped)
        return '\n'.join(lines)

    def _generate_3d_effect_code(self):
        """Generate simple effect code for 3D mode based on current shader."""
        preset = self.current_preset

        if preset == "Original":
            return """
    // Basic adjustments
    color *= pow(2.0, exposure);
    color += brightness;
    color = (color - 0.5) * contrast + 0.5;
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(gray), color, saturation);
    color = pow(max(color, vec3(0.0)), vec3(1.0 / gamma));
"""
        elif preset == "Toon Shader":
            return """
    // Toon quantization
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(gray), color, saturation_boost);
    color = floor(color * color_bands + 0.5) / color_bands;
    color += brightness_boost;
"""
        elif preset == "Pixelation":
            return """
    // Pixelation effect
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(gray), color, saturation_boost);
    float levels = color_depth;
    color = floor(color * levels + 0.5) / levels;
"""
        elif preset == "Sepia":
            return """
    // Sepia tone
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    vec3 sepia = vec3(1.2, 1.0, 0.8) * gray;
    color = mix(color, sepia, sepia_intensity);
"""
        elif preset == "Color Grade":
            return """
    // Color grading
    color = pow(color, vec3(1.0 / gamma));
    color = color * vec3(red_gain, green_gain, blue_gain);
    color += temperature * vec3(0.1, 0.0, -0.1);
    color += tint * vec3(0.0, 0.1, 0.0);
"""
        elif preset == "Noir":
            return """
    // Noir effect
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    gray = pow(gray, film_gamma);
    color = vec3(gray);
"""
        elif preset == "Cyberpunk":
            return """
    // Cyberpunk effect
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(gray), color, saturation);
    color *= neon_intensity;
"""
        else:
            return """
    // Default: pass through with basic adjustments
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
"""

    def load_texture(self, path):
        """Load an image file as texture. Supports animated GIFs."""
        try:
            self.makeCurrent()

            # Invalidate all layer caches (source image is changing)
            self._invalidate_all_layer_caches()

            # Stop any existing GIF animation
            self.gif_timer.stop()
            self.gif_playing = False
            self.gif_frames = []
            self.gif_frame_durations = []
            self.gif_current_frame = 0
            self.is_gif = False

            img = Image.open(path)
            self.image_path = path

            # Check if this is an animated GIF
            is_animated_gif = path.lower().endswith('.gif') and hasattr(img, 'n_frames') and img.n_frames > 1

            if is_animated_gif:
                # Load all frames from the GIF
                self.is_gif = True
                self.gif_frames = []
                self.gif_frame_durations = []

                for frame_idx in range(img.n_frames):
                    img.seek(frame_idx)
                    # Get frame duration (default to 100ms if not specified)
                    duration = img.info.get('duration', 100)
                    if duration == 0:
                        duration = 100  # Some GIFs have 0 duration, default to 100ms

                    # Convert frame to RGBA
                    frame = img.convert("RGBA")
                    frame_data = np.array(frame)
                    frame_data = np.flipud(frame_data)

                    self.gif_frames.append(frame_data)
                    self.gif_frame_durations.append(duration)

                self.image_size = img.size
                print(f"Loaded animated GIF: {img.n_frames} frames")

                # Upload first frame
                self._upload_texture(self.gif_frames[0])

                # Start GIF animation
                self.gif_current_frame = 0
                self.gif_playing = True
                self.gif_timer.setInterval(self.gif_frame_durations[0])
                self.gif_timer.start()
            else:
                # Regular image (or single-frame GIF)
                img = img.convert("RGBA")
                self.image_size = img.size

                # Convert to numpy array (flip for OpenGL)
                data = np.array(img)
                data = np.flipud(data)

                self._upload_texture(data)

            # Store original image for bake chain reset (only on fresh loads, not bake uploads)
            if not self._bake_chain:
                self._original_image_data = data.copy()
                self._original_image_path = path

            self.update()
            self.textureLoaded.emit(path)
            return True
        except Exception as e:
            print(f"Error loading texture: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _advance_gif_frame(self):
        """Advance to the next frame of an animated GIF."""
        if not self.gif_frames or not self.gif_playing:
            return

        # Move to next frame
        self.gif_current_frame = (self.gif_current_frame + 1) % len(self.gif_frames)

        # Upload the new frame
        self.makeCurrent()
        self._upload_texture(self.gif_frames[self.gif_current_frame])

        # Set timer for next frame duration
        next_duration = self.gif_frame_durations[self.gif_current_frame]
        self.gif_timer.setInterval(next_duration)

        self.update()

    def toggle_gif_playback(self):
        """Toggle GIF animation playback."""
        if not self.is_gif or not self.gif_frames:
            return

        self.gif_playing = not self.gif_playing
        if self.gif_playing:
            self.gif_timer.start()
        else:
            self.gif_timer.stop()

    def set_gif_frame(self, frame_index):
        """Set GIF to a specific frame."""
        if not self.is_gif or not self.gif_frames:
            return

        frame_index = max(0, min(frame_index, len(self.gif_frames) - 1))
        self.gif_current_frame = frame_index

        self.makeCurrent()
        self._upload_texture(self.gif_frames[self.gif_current_frame])
        self.update()

    def set_preset(self, name):
        """Switch to a different shader preset."""
        if name not in SHADERS:
            return
        self.current_preset = name
        self._init_params()
        self.makeCurrent()
        self._compile_shader()
        if self.mode_3d:
            self._compile_shader_3d()
        self.update()

    def set_param(self, name, value):
        """Update a shader parameter."""
        self.params[name] = value
        self.update()

    def paintGL(self):
        if not self.initialized:
            return

        try:
            # Clear any pending OpenGL errors from previous operations
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass

            # Get viewport size
            ratio = self.devicePixelRatio()
            w = int(self.width() * ratio)
            h = int(self.height() * ratio)

            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.12, 0.12, 0.12, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            if self.mode_3d:
                self._paint_3d(w, h)
            else:
                self._paint_2d()
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _paint_2d(self):
        """Render 2D image with shader effect."""
        if self.program is None or self.texture_id is None:
            return

        GL.glDisable(GL.GL_DEPTH_TEST)

        # Check if we have shader layers to render
        if self.shader_layers:
            enabled_layers = [l for l in self.shader_layers if l.get('enabled', True)]
            if enabled_layers:
                self._render_with_layers(enabled_layers)
                return

        # If we have post-processing effects, render to FBO first
        if self.post_process_stack:
            ratio = self.devicePixelRatio()
            w = int(self.width() * ratio)
            h = int(self.height() * ratio)
            self._create_fbo(w, h)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            GL.glViewport(0, 0, w, h)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self.program)
        GL.glBindVertexArray(self.vao)

        # Bind texture
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        # Set texture uniform
        tex_loc = GL.glGetUniformLocation(self.program, "u_texture")
        if tex_loc >= 0:
            GL.glUniform1i(tex_loc, 0)

        # Set other uniforms
        for name, value in self.params.items():
            loc = GL.glGetUniformLocation(self.program, name)
            if loc >= 0:
                GL.glUniform1f(loc, float(value))

        # Draw quad
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # Apply post-processing stack
        if self.post_process_stack:
            self._apply_post_processing()

    # --- Layer System v2: FBO Management, Compositing, Rendering ---

    def _create_fbo_pair(self, width, height):
        """Create a single FBO + texture pair. Returns (fbo, texture) or (None, None)."""
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                       GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, texture, 0)

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glDeleteFramebuffers(1, [fbo])
            GL.glDeleteTextures([texture])
            return None, None

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return fbo, texture

    def _flush_fbo_to_texture(self, fbo, texture, width, height):
        """Copy FBO content to its attached texture using glCopyTexSubImage2D.

        Workaround for NVIDIA driver issue where rendering to an FBO-attached
        texture doesn't make the content available for subsequent sampling.
        Uses glCopyTexSubImage2D (not glCopyTexImage2D) to avoid reallocating
        the texture storage, which would break the FBO attachment.
        The FBO must still be bound when calling this method.
        """
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glCopyTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height)

    def _ensure_accum_fbos(self, width, height):
        """Create or resize the two accumulation FBOs for compositing."""
        if self.accum_fbo_size == (width, height) and self.accum_fbo_a is not None:
            return True

        # Delete old
        for fbo_attr, tex_attr in [('accum_fbo_a', 'accum_fbo_a_texture'),
                                    ('accum_fbo_b', 'accum_fbo_b_texture')]:
            fbo = getattr(self, fbo_attr)
            tex = getattr(self, tex_attr)
            if fbo is not None:
                GL.glDeleteFramebuffers(1, [fbo])
                GL.glDeleteTextures([tex])

        self.accum_fbo_a, self.accum_fbo_a_texture = self._create_fbo_pair(width, height)
        if self.accum_fbo_a is None:
            return False

        self.accum_fbo_b, self.accum_fbo_b_texture = self._create_fbo_pair(width, height)
        if self.accum_fbo_b is None:
            GL.glDeleteFramebuffers(1, [self.accum_fbo_a])
            GL.glDeleteTextures([self.accum_fbo_a_texture])
            self.accum_fbo_a = None
            return False

        self.accum_fbo_size = (width, height)
        return True

    def _ensure_layer_cache_fbo(self, layer_id, width, height):
        """Ensure a cache FBO exists for the given layer. Returns (fbo, texture)."""
        if layer_id in self.layer_cache:
            entry = self.layer_cache[layer_id]
            if entry.get('size') == (width, height):
                return entry['fbo'], entry['texture']
            # Size changed, delete old
            GL.glDeleteFramebuffers(1, [entry['fbo']])
            GL.glDeleteTextures([entry['texture']])

        fbo, texture = self._create_fbo_pair(width, height)
        if fbo is None:
            return None, None

        self.layer_cache[layer_id] = {
            'fbo': fbo, 'texture': texture, 'dirty': True, 'size': (width, height)
        }
        return fbo, texture

    def _cleanup_layer_cache(self, active_layer_ids):
        """Delete cached FBOs for layers that no longer exist."""
        to_remove = [lid for lid in self.layer_cache if lid not in active_layer_ids]
        for lid in to_remove:
            entry = self.layer_cache.pop(lid)
            GL.glDeleteFramebuffers(1, [entry['fbo']])
            GL.glDeleteTextures([entry['texture']])

    def _invalidate_all_layer_caches(self):
        """Mark all cached layers as dirty (e.g., when source image changes)."""
        for entry in self.layer_cache.values():
            entry['dirty'] = True

    def _compile_compositing_shader(self):
        """Compile the GPU compositing shader for layer blending."""
        try:
            vert_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
            GL.glShaderSource(vert_shader, VERTEX_SHADER)
            GL.glCompileShader(vert_shader)
            if not GL.glGetShaderiv(vert_shader, GL.GL_COMPILE_STATUS):
                GL.glDeleteShader(vert_shader)
                return

            frag_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
            GL.glShaderSource(frag_shader, COMPOSITING_SHADER)
            GL.glCompileShader(frag_shader)
            if not GL.glGetShaderiv(frag_shader, GL.GL_COMPILE_STATUS):
                GL.glDeleteShader(vert_shader)
                GL.glDeleteShader(frag_shader)
                return

            self.compositing_program = GL.glCreateProgram()
            GL.glAttachShader(self.compositing_program, vert_shader)
            GL.glAttachShader(self.compositing_program, frag_shader)
            GL.glBindAttribLocation(self.compositing_program, 0, "in_vert")
            GL.glBindAttribLocation(self.compositing_program, 1, "in_uv")
            GL.glLinkProgram(self.compositing_program)

            GL.glDeleteShader(vert_shader)
            GL.glDeleteShader(frag_shader)

            if not GL.glGetProgramiv(self.compositing_program, GL.GL_LINK_STATUS):
                GL.glDeleteProgram(self.compositing_program)
                self.compositing_program = None
        except Exception:
            self.compositing_program = None

    def _ensure_passthrough_program(self):
        """Compile a simple passthrough shader for blitting textures."""
        if self.passthrough_program is not None:
            return self.passthrough_program
        try:
            vert_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
            GL.glShaderSource(vert_shader, VERTEX_SHADER)
            GL.glCompileShader(vert_shader)
            if not GL.glGetShaderiv(vert_shader, GL.GL_COMPILE_STATUS):
                GL.glDeleteShader(vert_shader)
                return None

            frag_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
            GL.glShaderSource(frag_shader, PASSTHROUGH_SHADER)
            GL.glCompileShader(frag_shader)
            if not GL.glGetShaderiv(frag_shader, GL.GL_COMPILE_STATUS):
                GL.glDeleteShader(vert_shader)
                GL.glDeleteShader(frag_shader)
                return None

            self.passthrough_program = GL.glCreateProgram()
            GL.glAttachShader(self.passthrough_program, vert_shader)
            GL.glAttachShader(self.passthrough_program, frag_shader)
            GL.glBindAttribLocation(self.passthrough_program, 0, "in_vert")
            GL.glBindAttribLocation(self.passthrough_program, 1, "in_uv")
            GL.glLinkProgram(self.passthrough_program)

            GL.glDeleteShader(vert_shader)
            GL.glDeleteShader(frag_shader)

            if not GL.glGetProgramiv(self.passthrough_program, GL.GL_LINK_STATUS):
                GL.glDeleteProgram(self.passthrough_program)
                self.passthrough_program = None
                return None
            return self.passthrough_program
        except Exception:
            self.passthrough_program = None
            return None

    def _update_layer_dirty_flags(self, layers):
        """Compare layers against previous frame and mark changed layers dirty."""
        prev_map = {l.get('id'): l for l in self._prev_layer_data}

        for layer in layers:
            lid = layer.get('id')
            if lid is None:
                continue
            prev = prev_map.get(lid)
            if prev is None:
                # New layer
                if lid in self.layer_cache:
                    self.layer_cache[lid]['dirty'] = True
            elif (layer.get('shader') != prev.get('shader') or
                  layer.get('params') != prev.get('params')):
                # Shader or params changed — need to re-render this layer's cache
                if lid in self.layer_cache:
                    self.layer_cache[lid]['dirty'] = True

        # Store current as previous for next frame
        import copy
        self._prev_layer_data = copy.deepcopy(layers)

    def _render_layer_to_cache(self, layer, width, height):
        """Render a layer's shader effect on the original image to its cache FBO."""
        lid = layer.get('id')
        fbo, texture = self._ensure_layer_cache_fbo(lid, width, height)
        if fbo is None:
            return

        shader_name = layer.get('shader', 'Original')
        program = self.layer_programs.get(shader_name)
        if program is None:
            program = self._compile_layer_shader(shader_name)
        if program is None:
            program = self.program
        if program is None:
            return

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(program)
        GL.glBindVertexArray(self.vao)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        tex_loc = GL.glGetUniformLocation(program, "u_texture")
        if tex_loc >= 0:
            GL.glUniform1i(tex_loc, 0)

        res_loc = GL.glGetUniformLocation(program, "u_resolution")
        if res_loc >= 0:
            GL.glUniform2f(res_loc, float(width), float(height))

        for name, value in layer.get('params', {}).items():
            loc = GL.glGetUniformLocation(program, name)
            if loc >= 0:
                GL.glUniform1f(loc, float(value))

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        self._flush_fbo_to_texture(fbo, texture, width, height)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self.layer_cache[lid]['dirty'] = False

    def _render_texture_to_screen(self, texture_id):
        """Render a texture to the default framebuffer using the passthrough shader."""
        program = self._ensure_passthrough_program()
        if program is None:
            return

        GL.glUseProgram(program)
        GL.glBindVertexArray(self.vao)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

        tex_loc = GL.glGetUniformLocation(program, "u_texture")
        if tex_loc >= 0:
            GL.glUniform1i(tex_loc, 0)

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    def _render_with_layers(self, layers):
        """Render composited layer stack to screen."""
        try:
            self._render_with_layers_composite(layers)
        except Exception:
            import traceback
            traceback.print_exc()
            if layers:
                # Fallback: render first layer directly
                shader_name = layers[0].get('shader', 'Original')
                program = self.layer_programs.get(shader_name) or self._compile_layer_shader(shader_name) or self.program
                if program:
                    GL.glUseProgram(program)
                    GL.glBindVertexArray(self.vao)
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
                    tex_loc = GL.glGetUniformLocation(program, "u_texture")
                    if tex_loc >= 0:
                        GL.glUniform1i(tex_loc, 0)
                    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    def _render_with_layers_composite(self, layers):
        """Compositing pipeline: render each layer independently, then blend bottom-to-top."""
        if self.texture_id is None or not layers:
            return

        ratio = self.devicePixelRatio()
        viewport_w = int(self.width() * ratio)
        viewport_h = int(self.height() * ratio)
        fbo_w, fbo_h = self.image_size

        # Single layer optimization
        if len(layers) == 1:
            layer = layers[0]
            lid = layer.get('id')
            if lid is not None:
                if lid not in self.layer_cache or self.layer_cache.get(lid, {}).get('dirty', True):
                    self._render_layer_to_cache(layer, fbo_w, fbo_h)
                if lid in self.layer_cache:
                    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
                    GL.glViewport(0, 0, viewport_w, viewport_h)
                    self._render_texture_to_screen(self.layer_cache[lid]['texture'])
                    return
            # Fallback for layers without id
            self._render_texture_to_screen(self.texture_id)
            return

        # Multi-layer compositing
        active_ids = {l.get('id') for l in layers if l.get('id') is not None}
        self._update_layer_dirty_flags(layers)
        self._cleanup_layer_cache(active_ids)

        if not self._ensure_accum_fbos(fbo_w, fbo_h):
            self._render_texture_to_screen(self.texture_id)
            return

        if self.compositing_program is None:
            self._compile_compositing_shader()
        if self.compositing_program is None:
            self._render_texture_to_screen(self.texture_id)
            return

        # Phase 1: Render dirty layers to their cache FBOs
        for layer in layers:
            lid = layer.get('id')
            if lid is None:
                continue
            cache = self.layer_cache.get(lid)
            if cache is None or cache.get('dirty', True):
                self._render_layer_to_cache(layer, fbo_w, fbo_h)

        # Phase 2: Composite bottom-to-top using ping-pong accumulation
        accum_fbos = [self.accum_fbo_a, self.accum_fbo_b]
        accum_textures = [self.accum_fbo_a_texture, self.accum_fbo_b_texture]
        read_idx = 0  # Which accum buffer holds current result
        first_enabled = True

        for layer in layers:
            lid = layer.get('id')
            if lid is None or lid not in self.layer_cache:
                continue

            layer_tex = self.layer_cache[lid]['texture']

            if first_enabled:
                # Base layer: copy to accumulation buffer A
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, accum_fbos[0])
                GL.glViewport(0, 0, fbo_w, fbo_h)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)

                opacity = layer.get('opacity', 1.0)
                if opacity >= 1.0:
                    self._render_texture_to_screen_fbo(layer_tex, fbo_w, fbo_h)
                else:
                    # For base layer with < 1.0 opacity, blend with black
                    GL.glEnable(GL.GL_BLEND)
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                    self._render_texture_with_alpha(layer_tex, opacity, fbo_w, fbo_h)
                    GL.glDisable(GL.GL_BLEND)

                self._flush_fbo_to_texture(accum_fbos[0], accum_textures[0], fbo_w, fbo_h)
                read_idx = 0
                first_enabled = False
            else:
                # Composite this layer onto accumulation
                write_idx = 1 - read_idx

                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, accum_fbos[write_idx])
                GL.glViewport(0, 0, fbo_w, fbo_h)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)

                GL.glUseProgram(self.compositing_program)
                GL.glBindVertexArray(self.vao)

                # u_base = accumulation result (texture unit 0)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, accum_textures[read_idx])
                base_loc = GL.glGetUniformLocation(self.compositing_program, "u_base")
                if base_loc >= 0:
                    GL.glUniform1i(base_loc, 0)

                # u_overlay = this layer's cache (texture unit 1)
                GL.glActiveTexture(GL.GL_TEXTURE1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, layer_tex)
                overlay_loc = GL.glGetUniformLocation(self.compositing_program, "u_overlay")
                if overlay_loc >= 0:
                    GL.glUniform1i(overlay_loc, 1)

                # Set blend mode and opacity
                opacity_loc = GL.glGetUniformLocation(self.compositing_program, "u_opacity")
                if opacity_loc >= 0:
                    GL.glUniform1f(opacity_loc, float(layer.get('opacity', 1.0)))

                mode_loc = GL.glGetUniformLocation(self.compositing_program, "u_blend_mode")
                if mode_loc >= 0:
                    blend_mode = layer.get('blend_mode', 'normal')
                    GL.glUniform1i(mode_loc, BLEND_MODES.get(blend_mode, 0))

                GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                self._flush_fbo_to_texture(accum_fbos[write_idx], accum_textures[write_idx], fbo_w, fbo_h)

                read_idx = write_idx

        # Phase 3: Render final accumulation to screen
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, viewport_w, viewport_h)

        if not first_enabled:
            self._render_texture_to_screen(accum_textures[read_idx])
        else:
            self._render_texture_to_screen(self.texture_id)

        # Reset texture unit to 0
        GL.glActiveTexture(GL.GL_TEXTURE0)

    def _render_texture_to_screen_fbo(self, texture_id, width, height):
        """Render a texture to the currently bound FBO (not the screen)."""
        program = self._ensure_passthrough_program()
        if program is None:
            return

        GL.glUseProgram(program)
        GL.glBindVertexArray(self.vao)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

        tex_loc = GL.glGetUniformLocation(program, "u_texture")
        if tex_loc >= 0:
            GL.glUniform1i(tex_loc, 0)

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    def _render_texture_with_alpha(self, texture_id, opacity, width, height):
        """Render a texture with a given opacity to the currently bound FBO.
        Uses the compositing shader to blend the texture against a transparent/black base."""
        # For base layer opacity < 1.0, just use passthrough and let the alpha handle it
        program = self._ensure_passthrough_program()
        if program is None:
            return

        GL.glUseProgram(program)
        GL.glBindVertexArray(self.vao)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

        tex_loc = GL.glGetUniformLocation(program, "u_texture")
        if tex_loc >= 0:
            GL.glUniform1i(tex_loc, 0)

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    def _apply_post_processing(self):
        """Apply the post-processing shader stack."""
        ratio = self.devicePixelRatio()
        w = int(self.width() * ratio)
        h = int(self.height() * ratio)

        # We need ping-pong rendering for multiple effects
        current_texture = self.fbo_texture

        for i, shader_name in enumerate(self.post_process_stack):
            is_last = (i == len(self.post_process_stack) - 1)

            if is_last:
                # Render to screen
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            else:
                # Would need second FBO for true ping-pong, for now just render to screen
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

            GL.glViewport(0, 0, w, h)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            program = self._compile_post_shader(shader_name)
            if program is None:
                continue

            GL.glUseProgram(program)
            GL.glBindVertexArray(self.vao)

            # Bind FBO texture
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, current_texture)

            tex_loc = GL.glGetUniformLocation(program, "u_texture")
            if tex_loc >= 0:
                GL.glUniform1i(tex_loc, 0)

            # Set default uniforms for the post shader
            shader_def = SHADERS.get(shader_name, {})
            for name, props in shader_def.get("uniforms", {}).items():
                loc = GL.glGetUniformLocation(program, name)
                if loc >= 0:
                    GL.glUniform1f(loc, props.get("default", 0.0))

            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    def _paint_3d(self, w, h):
        """Render 3D model with shader effect."""
        if self.program_3d is None or self.vao_3d is None:
            return

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.program_3d)
        GL.glBindVertexArray(self.vao_3d)

        # Bind texture if available
        GL.glActiveTexture(GL.GL_TEXTURE0)
        if self.texture_id is not None:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        # Set texture uniform
        tex_loc = GL.glGetUniformLocation(self.program_3d, "u_texture")
        if tex_loc >= 0:
            GL.glUniform1i(tex_loc, 0)

        # Set transform matrices
        aspect = w / h if h > 0 else 1.0
        model = self._make_rotation_matrix()
        view = self._make_view_matrix()
        projection = self._make_perspective_matrix(45.0, aspect, 0.1, 100.0)

        model_loc = GL.glGetUniformLocation(self.program_3d, "u_model")
        view_loc = GL.glGetUniformLocation(self.program_3d, "u_view")
        proj_loc = GL.glGetUniformLocation(self.program_3d, "u_projection")

        if model_loc >= 0:
            GL.glUniformMatrix4fv(model_loc, 1, GL.GL_TRUE, model)
        if view_loc >= 0:
            GL.glUniformMatrix4fv(view_loc, 1, GL.GL_TRUE, view)
        if proj_loc >= 0:
            GL.glUniformMatrix4fv(proj_loc, 1, GL.GL_TRUE, projection)

        # Set light uniforms
        for i, light in enumerate(self.lights):
            pos_loc = GL.glGetUniformLocation(self.program_3d, f"u_light{i+1}_pos")
            color_loc = GL.glGetUniformLocation(self.program_3d, f"u_light{i+1}_color")
            intensity_loc = GL.glGetUniformLocation(self.program_3d, f"u_light{i+1}_intensity")
            if pos_loc >= 0:
                GL.glUniform3f(pos_loc, *light["pos"])
            if color_loc >= 0:
                GL.glUniform3f(color_loc, *light["color"])
            if intensity_loc >= 0:
                GL.glUniform1f(intensity_loc, light["intensity"])

        # Set additional lighting uniforms
        ambient_loc = GL.glGetUniformLocation(self.program_3d, "u_ambient_intensity")
        if ambient_loc >= 0:
            GL.glUniform1f(ambient_loc, self.ambient_intensity)

        spec_power_loc = GL.glGetUniformLocation(self.program_3d, "u_specular_power")
        if spec_power_loc >= 0:
            GL.glUniform1f(spec_power_loc, self.specular_power)

        spec_int_loc = GL.glGetUniformLocation(self.program_3d, "u_specular_intensity")
        if spec_int_loc >= 0:
            GL.glUniform1f(spec_int_loc, self.specular_intensity)

        # Camera position for specular
        cam_pos_loc = GL.glGetUniformLocation(self.program_3d, "u_camera_pos")
        if cam_pos_loc >= 0:
            GL.glUniform3f(cam_pos_loc, 0.0, 0.0, self.zoom)

        # Set shader-specific uniforms
        for name, value in self.params.items():
            loc = GL.glGetUniformLocation(self.program_3d, name)
            if loc >= 0:
                GL.glUniform1f(loc, float(value))

        # Draw model
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.model_vertex_count)

    def resizeGL(self, w, h):
        ratio = self.devicePixelRatio()
        GL.glViewport(0, 0, int(w * ratio), int(h * ratio))

    # --- 3D Mode Methods ---

    def _animate(self):
        """Animation timer callback for auto-rotation."""
        if self.auto_rotate:
            self.rotation_y += self.rotation_speed
            if self.rotation_y > 360:
                self.rotation_y -= 360
            self.update()

    def set_auto_rotate(self, enabled):
        """Enable or disable auto-rotation."""
        self.auto_rotate = enabled
        if enabled:
            self.animation_timer.start()
        else:
            self.animation_timer.stop()

    def load_3d_model(self, path):
        """Load a 3D model (OBJ, GLTF, GLB)."""
        ext = os.path.splitext(path)[1].lower()

        verts, uvs, norms = None, None, None

        if ext == '.obj':
            verts, uvs, norms = load_obj(path)
        elif ext in ['.gltf', '.glb']:
            verts, uvs, norms = load_gltf(path)
        else:
            print(f"Unsupported format: {ext}")
            return False

        if verts is None:
            return False

        self._setup_3d_model(verts, uvs, norms)
        self.mode_3d = True
        self.image_path = path
        self._compile_shader_3d()
        self.update()
        return True

    def load_primitive(self, primitive_type):
        """Load a primitive shape (cube, sphere)."""
        if primitive_type == "sphere":
            verts, uvs, norms = create_sphere()
        elif primitive_type == "cube":
            verts, uvs, norms = create_cube()
        else:
            return False

        self._setup_3d_model(verts, uvs, norms)
        self.mode_3d = True
        self._compile_shader_3d()
        self.update()
        return True

    def _setup_3d_model(self, verts, uvs, norms):
        """Setup VAO/VBO for 3D model."""
        self.makeCurrent()

        # Interleave vertex data: pos(3) + uv(2) + normal(3) = 8 floats per vertex
        vertex_count = len(verts) // 3
        self.model_vertex_count = vertex_count

        interleaved = []
        for i in range(vertex_count):
            interleaved.extend([
                verts[i*3], verts[i*3+1], verts[i*3+2],
                uvs[i*2], uvs[i*2+1],
                norms[i*3], norms[i*3+1], norms[i*3+2]
            ])

        data = np.array(interleaved, dtype=np.float32)

        if self.vao_3d is None:
            self.vao_3d = GL.glGenVertexArrays(1)
        if self.vbo_3d is None:
            self.vbo_3d = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao_3d)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_3d)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)

        stride = 8 * 4  # 8 floats * 4 bytes

        # Position
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glEnableVertexAttribArray(0)

        # UV
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)

        # Normal
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(20))
        GL.glEnableVertexAttribArray(2)

        self.model_vertices = data
        print(f"Loaded 3D model: {vertex_count} vertices")

    def set_2d_mode(self):
        """Switch back to 2D mode."""
        self.mode_3d = False
        self.set_auto_rotate(False)
        self.update()

    def export_to_image(self, width=None, height=None, layers=None):
        """Export the current render to an image array.
        If layers is provided, the full shader layer chain is applied.
        Renders at original image size for maximum quality.
        """
        self.makeCurrent()

        # Always export at original image size
        if width is None:
            width = self.image_size[0]
        if height is None:
            height = self.image_size[1]

        if self.mode_3d:
            # 3D export path - render to FBO at requested size
            return self._export_3d(width, height)

        # 2D export path
        GL.glDisable(GL.GL_DEPTH_TEST)
        enabled_layers = [l for l in (layers or []) if l.get('enabled', True)]

        if enabled_layers:
            return self._export_2d_with_layers(enabled_layers, width, height)
        else:
            return self._export_2d_single(self.program, self.params, width, height)

    def _export_2d_single(self, program, params, width, height):
        """Render a single shader pass to an FBO and read back pixels."""
        # Create FBO
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, tex, 0)

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            print("[EXPORT] FBO incomplete")
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glDeleteFramebuffers(1, [fbo])
            GL.glDeleteTextures([tex])
            return None

        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(program)
        GL.glBindVertexArray(self.vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        tex_loc = GL.glGetUniformLocation(program, "u_texture")
        if tex_loc >= 0:
            GL.glUniform1i(tex_loc, 0)
        res_loc = GL.glGetUniformLocation(program, "u_resolution")
        if res_loc >= 0:
            GL.glUniform2f(res_loc, float(width), float(height))
        for name, value in params.items():
            loc = GL.glGetUniformLocation(program, name)
            if loc >= 0:
                GL.glUniform1f(loc, float(value))

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glFinish()

        pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        image_data = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4).copy()
        image_data = np.flipud(image_data)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glDeleteFramebuffers(1, [fbo])
        GL.glDeleteTextures([tex])
        return image_data

    def _export_2d_with_layers(self, layers, width, height):
        """Composite layer stack at export resolution and read back pixels."""
        import numpy as np

        if not layers:
            return self._export_2d_single(self.program, self.params, width, height)

        # Single layer: simple export
        if len(layers) == 1:
            layer = layers[0]
            shader_name = layer.get('shader', 'Original')
            prog = self.layer_programs.get(shader_name) or self._compile_layer_shader(shader_name)
            if not prog:
                prog = self.program
            return self._export_2d_single(prog, layer.get('params', {}), width, height)

        # Multi-layer: create temporary FBOs at export resolution
        # Per-layer FBOs
        layer_fbos = []
        for layer in layers:
            fbo, tex = self._create_fbo_pair(width, height)
            if fbo is None:
                # Cleanup on failure
                for f, t in layer_fbos:
                    GL.glDeleteFramebuffers(1, [f])
                    GL.glDeleteTextures([t])
                return None
            layer_fbos.append((fbo, tex))

        # Accumulation FBOs
        accum_a_fbo, accum_a_tex = self._create_fbo_pair(width, height)
        accum_b_fbo, accum_b_tex = self._create_fbo_pair(width, height)
        if accum_a_fbo is None or accum_b_fbo is None:
            for f, t in layer_fbos:
                GL.glDeleteFramebuffers(1, [f])
                GL.glDeleteTextures([t])
            if accum_a_fbo:
                GL.glDeleteFramebuffers(1, [accum_a_fbo])
                GL.glDeleteTextures([accum_a_tex])
            return None

        if self.compositing_program is None:
            self._compile_compositing_shader()

        # Phase 1: Render each layer
        for i, layer in enumerate(layers):
            shader_name = layer.get('shader', 'Original')
            prog = self.layer_programs.get(shader_name) or self._compile_layer_shader(shader_name)
            if not prog:
                prog = self.program
            if not prog:
                continue

            fbo, tex = layer_fbos[i]
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glViewport(0, 0, width, height)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            GL.glUseProgram(prog)
            GL.glBindVertexArray(self.vao)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

            tex_loc = GL.glGetUniformLocation(prog, "u_texture")
            if tex_loc >= 0:
                GL.glUniform1i(tex_loc, 0)
            res_loc = GL.glGetUniformLocation(prog, "u_resolution")
            if res_loc >= 0:
                GL.glUniform2f(res_loc, float(width), float(height))

            for name, value in layer.get('params', {}).items():
                loc = GL.glGetUniformLocation(prog, name)
                if loc >= 0:
                    GL.glUniform1f(loc, float(value))

            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            self._flush_fbo_to_texture(fbo, tex, width, height)

        # Phase 2: Composite
        accum_fbos = [accum_a_fbo, accum_b_fbo]
        accum_textures = [accum_a_tex, accum_b_tex]
        read_idx = 0
        first_enabled = True

        for i, layer in enumerate(layers):
            _, layer_tex = layer_fbos[i]

            if first_enabled:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, accum_fbos[0])
                GL.glViewport(0, 0, width, height)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                self._render_texture_to_screen_fbo(layer_tex, width, height)
                self._flush_fbo_to_texture(accum_fbos[0], accum_textures[0], width, height)
                read_idx = 0
                first_enabled = False
            else:
                write_idx = 1 - read_idx

                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, accum_fbos[write_idx])
                GL.glViewport(0, 0, width, height)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)

                GL.glUseProgram(self.compositing_program)
                GL.glBindVertexArray(self.vao)

                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, accum_textures[read_idx])
                base_loc = GL.glGetUniformLocation(self.compositing_program, "u_base")
                if base_loc >= 0:
                    GL.glUniform1i(base_loc, 0)

                GL.glActiveTexture(GL.GL_TEXTURE1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, layer_tex)
                overlay_loc = GL.glGetUniformLocation(self.compositing_program, "u_overlay")
                if overlay_loc >= 0:
                    GL.glUniform1i(overlay_loc, 1)

                opacity_loc = GL.glGetUniformLocation(self.compositing_program, "u_opacity")
                if opacity_loc >= 0:
                    GL.glUniform1f(opacity_loc, float(layer.get('opacity', 1.0)))

                mode_loc = GL.glGetUniformLocation(self.compositing_program, "u_blend_mode")
                if mode_loc >= 0:
                    blend_mode = layer.get('blend_mode', 'normal')
                    GL.glUniform1i(mode_loc, BLEND_MODES.get(blend_mode, 0))

                GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                self._flush_fbo_to_texture(accum_fbos[write_idx], accum_textures[write_idx], width, height)
                read_idx = write_idx

        # Phase 3: Read pixels from final accumulation
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, accum_fbos[read_idx])
        pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        image_data = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4).copy()
        image_data = np.flipud(image_data)

        # Cleanup
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        for f, t in layer_fbos:
            GL.glDeleteFramebuffers(1, [f])
            GL.glDeleteTextures([t])
        GL.glDeleteFramebuffers(1, [accum_a_fbo])
        GL.glDeleteTextures([accum_a_tex])
        GL.glDeleteFramebuffers(1, [accum_b_fbo])
        GL.glDeleteTextures([accum_b_tex])

        return image_data

    def _export_3d(self, width, height):
        """Export 3D scene to image."""
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, tex, 0)
        depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                     GL.GL_RENDERBUFFER, depth_rb)

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            print("[EXPORT] 3D FBO incomplete")
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glDeleteFramebuffers(1, [fbo])
            GL.glDeleteTextures([tex])
            GL.glDeleteRenderbuffers(1, [depth_rb])
            return None

        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.12, 0.12, 0.12, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._paint_3d(width, height)
        GL.glFinish()

        pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        image_data = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4).copy()
        image_data = np.flipud(image_data)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glDeleteFramebuffers(1, [fbo])
        GL.glDeleteTextures([tex])
        GL.glDeleteRenderbuffers(1, [depth_rb])
        return image_data

    def _make_rotation_matrix(self):
        """Create rotation matrix from Euler angles."""
        rx, ry, rz = math.radians(self.rotation_x), math.radians(self.rotation_y), math.radians(self.rotation_z)

        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        # Combined rotation matrix (ZYX order)
        return np.array([
            [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz, 0],
            [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz, 0],
            [-sy, sx*cy, cx*cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def _make_perspective_matrix(self, fov, aspect, near, far):
        """Create perspective projection matrix."""
        f = 1.0 / math.tan(math.radians(fov) / 2)
        return np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), 2*far*near/(near-far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

    def _make_view_matrix(self):
        """Create view matrix (camera looking at origin with pan)."""
        return np.array([
            [1, 0, 0, self.pan_x],
            [0, 1, 0, self.pan_y],
            [0, 0, 1, -self.zoom],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    # --- MOUSE CONTROLS ---
    def mousePressEvent(self, event):
        """Handle mouse press for orbit/pan controls and color picker."""
        self.mouse_pressed = True
        self.last_mouse_pos = event.position()
        self.mouse_button = event.button()

        if self.color_picker_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pick_color(event.position())
            self.color_picker_mode = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.mouse_pressed = False
        self.mouse_button = None

    def mouseMoveEvent(self, event):
        """Handle mouse move for orbit/pan controls."""
        if not self.mouse_pressed or self.last_mouse_pos is None:
            return

        if not self.mode_3d:
            return

        pos = event.position()
        dx = pos.x() - self.last_mouse_pos.x()
        dy = pos.y() - self.last_mouse_pos.y()

        if self.mouse_button == QtCore.Qt.MouseButton.LeftButton:
            # Orbit rotation
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            self.rotation_x = max(-90, min(90, self.rotation_x))
        elif self.mouse_button == QtCore.Qt.MouseButton.MiddleButton:
            # Pan
            self.pan_x += dx * 0.01
            self.pan_y -= dy * 0.01
        elif self.mouse_button == QtCore.Qt.MouseButton.RightButton:
            # Zoom
            self.zoom -= dy * 0.02
            self.zoom = max(0.5, min(20.0, self.zoom))

        self.last_mouse_pos = pos
        self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        if self.mode_3d:
            delta = event.angleDelta().y() / 120.0
            self.zoom -= delta * 0.3
            self.zoom = max(0.5, min(20.0, self.zoom))
            self.update()

    def _pick_color(self, pos):
        """Pick color from the rendered image at the given position."""
        self.makeCurrent()

        # Read pixel at position
        x = int(pos.x() * self.devicePixelRatio())
        y = int((self.height() - pos.y()) * self.devicePixelRatio())

        pixel = GL.glReadPixels(x, y, 1, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        if pixel:
            r, g, b, a = pixel[0], pixel[1], pixel[2], pixel[3]
            self.colorPicked.emit(r, g, b, a)

    def enable_color_picker(self):
        """Enable color picker mode."""
        self.color_picker_mode = True
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    # --- KEYFRAME ANIMATION ---
    def add_keyframe(self, time):
        """Add a keyframe at the specified time."""
        keyframe = {
            "time": time,
            "rotation_x": self.rotation_x,
            "rotation_y": self.rotation_y,
            "rotation_z": self.rotation_z,
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y
        }
        # Insert sorted by time
        self.keyframes = [k for k in self.keyframes if k["time"] != time]
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k["time"])
        return keyframe

    def remove_keyframe(self, time):
        """Remove keyframe at specified time."""
        self.keyframes = [k for k in self.keyframes if k["time"] != time]

    def play_keyframes(self):
        """Start playing keyframe animation."""
        if not self.keyframes:
            return
        self.keyframe_time = 0.0
        self.playing_keyframes = True
        self.keyframe_timer.start()

    def stop_keyframes(self):
        """Stop keyframe animation."""
        self.playing_keyframes = False
        self.keyframe_timer.stop()

    def _update_keyframe(self):
        """Update animation based on keyframes."""
        if not self.keyframes or not self.playing_keyframes:
            return

        self.keyframe_time += 0.016  # ~60fps

        # Find surrounding keyframes
        prev_kf = None
        next_kf = None
        for kf in self.keyframes:
            if kf["time"] <= self.keyframe_time:
                prev_kf = kf
            if kf["time"] > self.keyframe_time and next_kf is None:
                next_kf = kf
                break

        if prev_kf is None:
            prev_kf = self.keyframes[0]
        if next_kf is None:
            # Loop or stop
            if self.keyframes[-1]["time"] < self.keyframe_time:
                self.keyframe_time = 0.0
                return

        # Interpolate
        if prev_kf == next_kf or next_kf is None:
            t = 0.0
        else:
            t = (self.keyframe_time - prev_kf["time"]) / (next_kf["time"] - prev_kf["time"])
            t = max(0, min(1, t))

        # Smooth interpolation (ease in/out)
        t = t * t * (3 - 2 * t)

        if next_kf:
            self.rotation_x = prev_kf["rotation_x"] + t * (next_kf["rotation_x"] - prev_kf["rotation_x"])
            self.rotation_y = prev_kf["rotation_y"] + t * (next_kf["rotation_y"] - prev_kf["rotation_y"])
            self.rotation_z = prev_kf["rotation_z"] + t * (next_kf["rotation_z"] - prev_kf["rotation_z"])
            self.zoom = prev_kf["zoom"] + t * (next_kf["zoom"] - prev_kf["zoom"])
            self.pan_x = prev_kf["pan_x"] + t * (next_kf["pan_x"] - prev_kf["pan_x"])
            self.pan_y = prev_kf["pan_y"] + t * (next_kf["pan_y"] - prev_kf["pan_y"])

        self.update()

    # --- WIREFRAME/NORMAL MODES ---
    def set_wireframe_mode(self, enabled):
        """Toggle wireframe rendering."""
        self.wireframe_mode = enabled
        self.update()

    def set_show_normals(self, enabled):
        """Toggle normal visualization."""
        self.show_normals = enabled
        self.update()


class ParameterSlider(QtWidgets.QWidget):
    """A labeled slider for parameter control with tooltip."""
    valueChanged = QtCore.pyqtSignal(str, float)

    def __init__(self, name, props):
        super().__init__()
        self.name = name
        self.props = props

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        display_name = name.replace('_', ' ').title()
        self.label = QtWidgets.QLabel(f"{display_name}: {props['default']:.2f}")
        self.label.setStyleSheet("color: #ccc; font-size: 11px;")
        layout.addWidget(self.label)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(self._value_to_slider(props['default']))
        self.slider.valueChanged.connect(self._on_change)

        tooltip = f"{display_name}\nRange: {props['min']:.2f} - {props['max']:.2f}\nDefault: {props['default']:.2f}"
        if 'description' in props:
            tooltip = f"{props['description']}\n\n{tooltip}"
        self.slider.setToolTip(tooltip)
        self.setToolTip(tooltip)

        layout.addWidget(self.slider)

    def _value_to_slider(self, value):
        min_v, max_v = self.props['min'], self.props['max']
        return int((value - min_v) / (max_v - min_v) * 1000)

    def _slider_to_value(self, pos):
        min_v, max_v = self.props['min'], self.props['max']
        return min_v + (pos / 1000) * (max_v - min_v)

    def _on_change(self, pos):
        value = self._slider_to_value(pos)
        display_name = self.name.replace('_', ' ').title()
        self.label.setText(f"{display_name}: {value:.2f}")
        self.valueChanged.emit(self.name, value)

    def set_value(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(self._value_to_slider(value))
        self.slider.blockSignals(False)
        display_name = self.name.replace('_', ' ').title()
        self.label.setText(f"{display_name}: {value:.2f}")


class ShaderLayerWidget(QtWidgets.QFrame):
    """Widget representing a single shader layer in the compositing stack."""

    layerChanged = QtCore.pyqtSignal()
    layerRemoved = QtCore.pyqtSignal(object)
    layerMoved = QtCore.pyqtSignal(object, int)
    layerSelected = QtCore.pyqtSignal(object)
    layerDuplicated = QtCore.pyqtSignal(object)

    # Short labels for blend modes
    _BLEND_ABBREV = {
        "normal": "N", "multiply": "M", "screen": "S", "overlay": "O",
        "soft_light": "SL", "hard_light": "HL", "add": "+", "subtract": "-",
        "difference": "D", "exclusion": "E", "color_dodge": "CD", "color_burn": "CB",
    }

    def __init__(self, layer_id, parent_panel=None, is_base_layer=False):
        super().__init__()
        self.layer_id = layer_id
        self.layer_name = f"Layer_{layer_id}"
        self.shader_name = "Original"
        self.params = {}
        self.enabled = True
        self.opacity = 1.0
        self.blend_mode = "normal"
        self.is_base_layer = is_base_layer
        self.is_selected = False
        self._initializing = True

        self._update_style()
        self._setup_ui()
        self._initializing = False

    def _update_style(self):
        if self.is_selected:
            border_color = "#6a9fda"
            bg_color = "#3a4a5a"
        else:
            border_color = "#444"
            bg_color = "#2d2d2d"

        base_indicator = "border-left: 3px solid #5a9a5a;" if self.is_base_layer else ""

        self.setStyleSheet(f"""
            ShaderLayerWidget {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                margin: 2px;
                {base_indicator}
            }}
        """)

    def set_selected(self, selected):
        self.is_selected = selected
        self._update_style()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.layerSelected.emit(self)

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # Visibility toggle
        self.visible_cb = QtWidgets.QCheckBox()
        self.visible_cb.setChecked(True)
        self.visible_cb.setFixedWidth(20)
        self.visible_cb.setToolTip("Enable/Disable Layer")
        self.visible_cb.toggled.connect(self._on_visibility_changed)
        layout.addWidget(self.visible_cb)

        # Editable layer name
        self.name_edit = QtWidgets.QLineEdit(self.layer_name)
        self.name_edit.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: none;
                color: #ddd;
                font-size: 11px;
                padding: 2px;
            }
            QLineEdit:focus {
                background-color: #3a3a3a;
                border: 1px solid #555;
            }
        """)
        self.name_edit.editingFinished.connect(self._on_name_changed)
        layout.addWidget(self.name_edit, 1)

        # Shader indicator
        self.shader_label = QtWidgets.QLabel(f"({self.shader_name})")
        self.shader_label.setStyleSheet("color: #888; font-size: 10px;")
        self.shader_label.setToolTip("Shader - change in Inspector")
        layout.addWidget(self.shader_label)

        # Blend mode abbreviation
        self.blend_label = QtWidgets.QLabel(self._BLEND_ABBREV.get(self.blend_mode, "N"))
        self.blend_label.setStyleSheet("color: #7aa; font-size: 9px; font-weight: bold;")
        self.blend_label.setFixedWidth(20)
        self.blend_label.setToolTip("Blend Mode - change in Inspector")
        layout.addWidget(self.blend_label)

        # Move up
        self.up_btn = QtWidgets.QPushButton("▲")
        self.up_btn.setFixedSize(20, 20)
        self.up_btn.setStyleSheet("background-color: #3a3a5a; font-size: 8px;")
        self.up_btn.setToolTip("Move Up")
        self.up_btn.clicked.connect(lambda: self.layerMoved.emit(self, -1))
        layout.addWidget(self.up_btn)

        # Move down
        self.down_btn = QtWidgets.QPushButton("▼")
        self.down_btn.setFixedSize(20, 20)
        self.down_btn.setStyleSheet("background-color: #3a3a5a; font-size: 8px;")
        self.down_btn.setToolTip("Move Down")
        self.down_btn.clicked.connect(lambda: self.layerMoved.emit(self, 1))
        layout.addWidget(self.down_btn)

        # Duplicate button
        self.dup_btn = QtWidgets.QPushButton("⧉")
        self.dup_btn.setFixedSize(20, 20)
        self.dup_btn.setStyleSheet("background-color: #3a5a3a; font-size: 10px;")
        self.dup_btn.setToolTip("Duplicate Layer")
        self.dup_btn.clicked.connect(lambda: self.layerDuplicated.emit(self))
        layout.addWidget(self.dup_btn)

        # Remove button (hidden for base layer)
        self.remove_btn = QtWidgets.QPushButton("×")
        self.remove_btn.setFixedSize(20, 20)
        self.remove_btn.setStyleSheet("background-color: #5a2d2d;")
        self.remove_btn.setToolTip("Remove Layer")
        self.remove_btn.clicked.connect(lambda: self.layerRemoved.emit(self))
        if self.is_base_layer:
            self.remove_btn.hide()
        layout.addWidget(self.remove_btn)

    def _on_visibility_changed(self, checked):
        self.enabled = checked
        if not self._initializing:
            self.layerChanged.emit()

    def _on_name_changed(self):
        self.layer_name = self.name_edit.text()
        if not self._initializing:
            self.layerChanged.emit()

    def set_shader(self, shader_name):
        self.shader_name = shader_name
        self.shader_label.setText(f"({shader_name})")
        shader_def = SHADERS.get(shader_name, {})
        uniforms = shader_def.get("uniforms", {})
        self.params = {name: props.get("default", 0.0) for name, props in uniforms.items()}
        if not self._initializing:
            self.layerChanged.emit()

    def set_opacity(self, opacity):
        self.opacity = opacity
        if not self._initializing:
            self.layerChanged.emit()

    def set_blend_mode(self, mode):
        self.blend_mode = mode
        self.blend_label.setText(self._BLEND_ABBREV.get(mode, "N"))
        if not self._initializing:
            self.layerChanged.emit()

    def set_param(self, name, value):
        self.params[name] = value
        if not self._initializing:
            self.layerChanged.emit()

    def get_layer_data(self):
        return {
            'id': self.layer_id,
            'shader': self.shader_name,
            'params': self.params.copy(),
            'enabled': self.enabled,
            'opacity': self.opacity,
            'blend_mode': self.blend_mode,
            'name': self.layer_name,
            'is_base': self.is_base_layer,
        }


class ShaderLayerPanel(QtWidgets.QWidget):
    """Panel for managing multiple shader layers with compositing."""

    layersChanged = QtCore.pyqtSignal(object)
    layerSelected = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.layers = []
        self.next_layer_id = 0
        self.selected_layer = None
        self.base_layer = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Shader Layers")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #fff;")
        header.addWidget(title)

        header.addStretch()

        # Duplicate button
        dup_btn = QtWidgets.QPushButton("Duplicate")
        dup_btn.setStyleSheet("background-color: #3a5a3a; padding: 4px 8px;")
        dup_btn.clicked.connect(self.duplicate_layer)
        header.addWidget(dup_btn)

        # Add layer button
        add_btn = QtWidgets.QPushButton("+ Add Layer")
        add_btn.setStyleSheet("background-color: #3a6a3a; padding: 4px 8px;")
        add_btn.clicked.connect(self.add_layer)
        header.addWidget(add_btn)

        layout.addLayout(header)

        # Scroll area for layers
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setMaximumHeight(300)

        self.layers_container = QtWidgets.QWidget()
        self.layers_layout = QtWidgets.QVBoxLayout(self.layers_container)
        self.layers_layout.setContentsMargins(0, 0, 0, 0)
        self.layers_layout.setSpacing(2)
        self.layers_layout.addStretch()

        scroll.setWidget(self.layers_container)
        layout.addWidget(scroll)

        # Info label
        self.info_label = QtWidgets.QLabel("Add layers to stack multiple shader effects")
        self.info_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

    def _connect_layer_signals(self, layer):
        layer.layerChanged.connect(self._on_layer_changed)
        layer.layerRemoved.connect(self._remove_layer)
        layer.layerMoved.connect(self._move_layer)
        layer.layerSelected.connect(self._on_layer_selected)
        layer.layerDuplicated.connect(self._on_layer_duplicated)

    def create_base_layer(self, shader_name="Original"):
        self.clear_layers()

        self.base_layer = ShaderLayerWidget(self.next_layer_id, self, is_base_layer=True)
        self.next_layer_id += 1
        self.base_layer.set_shader(shader_name)
        self._connect_layer_signals(self.base_layer)

        self.layers_layout.insertWidget(self.layers_layout.count() - 1, self.base_layer)
        self.layers.append(self.base_layer)

        self._select_layer(self.base_layer)
        self._update_info_label()
        self._emit_layers()
        return self.base_layer

    def add_layer(self, shader_name=None):
        if not isinstance(shader_name, str):
            shader_name = "Original"

        if not self.base_layer:
            self.create_base_layer("Original")

        layer = ShaderLayerWidget(self.next_layer_id, self, is_base_layer=False)
        self.next_layer_id += 1
        layer.set_shader(shader_name)
        self._connect_layer_signals(layer)

        self.layers_layout.insertWidget(self.layers_layout.count() - 1, layer)
        self.layers.append(layer)

        self._select_layer(layer)
        self._update_info_label()
        self._emit_layers()
        return layer

    def duplicate_layer(self):
        if not self.selected_layer:
            return
        source = self.selected_layer

        new_layer = ShaderLayerWidget(self.next_layer_id, self, is_base_layer=False)
        self.next_layer_id += 1
        new_layer._initializing = True
        new_layer.set_shader(source.shader_name)
        new_layer.opacity = source.opacity
        new_layer.blend_mode = source.blend_mode
        new_layer.blend_label.setText(new_layer._BLEND_ABBREV.get(source.blend_mode, "N"))
        for name, value in source.params.items():
            new_layer.params[name] = value
        new_layer.layer_name = f"{source.layer_name} (copy)"
        new_layer.name_edit.setText(new_layer.layer_name)
        new_layer._initializing = False

        self._connect_layer_signals(new_layer)

        idx = self.layers.index(source)
        self.layers.insert(idx + 1, new_layer)
        self._rebuild_layout()
        self._select_layer(new_layer)
        self._update_info_label()
        self._emit_layers()

    def _on_layer_duplicated(self, layer):
        self.selected_layer = layer
        self.duplicate_layer()

    def _on_layer_selected(self, layer):
        self._select_layer(layer)

    def _select_layer(self, layer):
        for l in self.layers:
            l.set_selected(False)
        if layer:
            layer.set_selected(True)
            self.selected_layer = layer
            self.layerSelected.emit(layer.get_layer_data())

    def _remove_layer(self, layer):
        if layer in self.layers:
            was_selected = (layer == self.selected_layer)
            self.layers.remove(layer)
            self.layers_layout.removeWidget(layer)
            layer.deleteLater()

            if was_selected:
                if self.layers:
                    self._select_layer(self.layers[-1])
                else:
                    self.selected_layer = None

            self._update_info_label()
            self._emit_layers()

    def _move_layer(self, layer, direction):
        if layer not in self.layers:
            return
        idx = self.layers.index(layer)
        new_idx = idx + direction
        if 0 <= new_idx < len(self.layers):
            self.layers[idx], self.layers[new_idx] = self.layers[new_idx], self.layers[idx]
            self._rebuild_layout()
            self._emit_layers()

    def _rebuild_layout(self):
        while self.layers_layout.count() > 1:  # Keep stretch
            self.layers_layout.takeAt(0)
        for l in self.layers:
            self.layers_layout.insertWidget(self.layers_layout.count() - 1, l)

    def _on_layer_changed(self):
        self._emit_layers()

    def _emit_layers(self):
        layer_data = [layer.get_layer_data() for layer in self.layers]
        self.layersChanged.emit(layer_data)

    def _update_info_label(self):
        count = len(self.layers)
        if count == 0:
            self.info_label.setText("Add layers to stack multiple shader effects")
        elif count == 1:
            self.info_label.setText("1 layer active")
        else:
            self.info_label.setText(f"{count} layers active (bottom to top)")

    def get_all_layers(self):
        return [layer.get_layer_data() for layer in self.layers]

    def clear_layers(self):
        for layer in self.layers[:]:
            self.layers_layout.removeWidget(layer)
            layer.deleteLater()
        self.layers.clear()
        self.base_layer = None
        self.selected_layer = None
        self.next_layer_id = 0
        self._update_info_label()

    def update_selected_layer_shader(self, shader_name):
        if self.selected_layer:
            self.selected_layer.set_shader(shader_name)

    def update_selected_layer_param(self, param_name, value):
        if self.selected_layer:
            self.selected_layer.set_param(param_name, value)

    def get_selected_layer(self):
        return self.selected_layer

    def get_layer_stack_data(self):
        """Return serializable layer stack data for preset persistence."""
        stack = []
        for layer in self.layers:
            data = layer.get_layer_data()
            # Exclude runtime-only fields
            stack.append({
                'name': data['name'],
                'shader': data['shader'],
                'params': data['params'],
                'opacity': data['opacity'],
                'blend_mode': data['blend_mode'],
                'enabled': data['enabled'],
                'is_base': data['is_base'],
            })
        return stack

    def restore_layer_stack(self, stack_data):
        """Restore layers from serialized stack data (e.g., from a preset)."""
        self.clear_layers()
        for i, layer_data in enumerate(stack_data):
            is_base = layer_data.get('is_base', i == 0)
            layer = ShaderLayerWidget(self.next_layer_id, self, is_base_layer=is_base)
            self.next_layer_id += 1
            layer._initializing = True
            layer.set_shader(layer_data.get('shader', 'Original'))
            layer.opacity = layer_data.get('opacity', 1.0)
            layer.blend_mode = layer_data.get('blend_mode', 'normal')
            layer.blend_label.setText(layer._BLEND_ABBREV.get(layer.blend_mode, "N"))
            layer.enabled = layer_data.get('enabled', True)
            layer.visible_cb.setChecked(layer.enabled)
            layer.layer_name = layer_data.get('name', f"Layer_{layer.layer_id}")
            layer.name_edit.setText(layer.layer_name)
            for name, value in layer_data.get('params', {}).items():
                layer.params[name] = value
            layer._initializing = False

            if is_base:
                self.base_layer = layer

            self._connect_layer_signals(layer)
            self.layers_layout.insertWidget(self.layers_layout.count() - 1, layer)
            self.layers.append(layer)

        if self.layers:
            self._select_layer(self.layers[0])
        self._update_info_label()
        self._emit_layers()


class ShaderNodeWidget(QtWidgets.QGraphicsItem):
    """A node in the shader graph."""

    def __init__(self, node_type, name, x=0, y=0):
        super().__init__()
        self.node_type = node_type
        self.name = name
        self.inputs = []
        self.outputs = []
        self.width = 150
        self.height = 80
        self.setPos(x, y)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

        # Define node ports based on type
        if node_type == "texture":
            self.outputs = [("color", "vec4")]
        elif node_type == "math":
            self.inputs = [("a", "float"), ("b", "float")]
            self.outputs = [("result", "float")]
        elif node_type == "mix":
            self.inputs = [("a", "vec3"), ("b", "vec3"), ("factor", "float")]
            self.outputs = [("result", "vec3")]
        elif node_type == "output":
            self.inputs = [("color", "vec4")]
        elif node_type == "brightness":
            self.inputs = [("color", "vec3"), ("value", "float")]
            self.outputs = [("result", "vec3")]
        elif node_type == "contrast":
            self.inputs = [("color", "vec3"), ("value", "float")]
            self.outputs = [("result", "vec3")]
        elif node_type == "saturation":
            self.inputs = [("color", "vec3"), ("value", "float")]
            self.outputs = [("result", "vec3")]

        self.height = max(80, 40 + max(len(self.inputs), len(self.outputs)) * 20)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        # Node background
        if self.isSelected():
            painter.setBrush(QtGui.QBrush(QtGui.QColor(80, 80, 120)))
        else:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(60, 60, 60)))
        painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 2))
        painter.drawRoundedRect(0, 0, self.width, self.height, 5, 5)

        # Title bar
        painter.setBrush(QtGui.QBrush(QtGui.QColor(80, 100, 120)))
        painter.drawRoundedRect(0, 0, self.width, 25, 5, 5)
        painter.drawRect(0, 20, self.width, 5)

        # Title text
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.drawText(5, 17, self.name)

        # Input ports
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200)))
        for i, (name, _) in enumerate(self.inputs):
            y = 35 + i * 20
            painter.setBrush(QtGui.QBrush(QtGui.QColor(100, 200, 100)))
            painter.drawEllipse(-5, y - 5, 10, 10)
            painter.drawText(10, y + 4, name)

        # Output ports
        for i, (name, _) in enumerate(self.outputs):
            y = 35 + i * 20
            painter.setBrush(QtGui.QBrush(QtGui.QColor(200, 100, 100)))
            painter.drawEllipse(self.width - 5, y - 5, 10, 10)
            painter.drawText(self.width - 50, y + 4, name)


class ShaderNodeGraphDialog(QtWidgets.QDialog):
    """Visual node-based shader editor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shader Node Graph")
        self.resize(1000, 700)
        self.nodes = []
        self.connections = []

        layout = QtWidgets.QVBoxLayout(self)

        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()

        add_texture_btn = QtWidgets.QPushButton("+ Texture")
        add_texture_btn.clicked.connect(lambda: self._add_node("texture", "Texture"))
        toolbar.addWidget(add_texture_btn)

        add_brightness_btn = QtWidgets.QPushButton("+ Brightness")
        add_brightness_btn.clicked.connect(lambda: self._add_node("brightness", "Brightness"))
        toolbar.addWidget(add_brightness_btn)

        add_contrast_btn = QtWidgets.QPushButton("+ Contrast")
        add_contrast_btn.clicked.connect(lambda: self._add_node("contrast", "Contrast"))
        toolbar.addWidget(add_contrast_btn)

        add_saturation_btn = QtWidgets.QPushButton("+ Saturation")
        add_saturation_btn.clicked.connect(lambda: self._add_node("saturation", "Saturation"))
        toolbar.addWidget(add_saturation_btn)

        add_mix_btn = QtWidgets.QPushButton("+ Mix")
        add_mix_btn.clicked.connect(lambda: self._add_node("mix", "Mix"))
        toolbar.addWidget(add_mix_btn)

        add_math_btn = QtWidgets.QPushButton("+ Math")
        add_math_btn.clicked.connect(lambda: self._add_node("math", "Math"))
        toolbar.addWidget(add_math_btn)

        toolbar.addStretch()

        generate_btn = QtWidgets.QPushButton("Generate Shader")
        generate_btn.clicked.connect(self._generate_shader)
        generate_btn.setStyleSheet("background-color: #2d5a2d;")
        toolbar.addWidget(generate_btn)

        layout.addLayout(toolbar)

        # Graphics scene and view
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.setSceneRect(-500, -500, 1000, 1000)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet("background-color: #2a2a2a;")
        layout.addWidget(self.view)

        # Add default nodes
        self._add_node("texture", "Input Texture", -200, 0)
        self._add_node("output", "Output", 200, 0)

        # Generated shader display
        self.shader_output = QtWidgets.QPlainTextEdit()
        self.shader_output.setMaximumHeight(150)
        self.shader_output.setReadOnly(True)
        self.shader_output.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(self.shader_output)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        apply_btn = QtWidgets.QPushButton("Apply Shader")
        apply_btn.clicked.connect(self.accept)
        apply_btn.setStyleSheet("background-color: #2d5a2d;")
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

    def _add_node(self, node_type, name, x=None, y=None):
        """Add a new node to the graph."""
        if x is None:
            x = random.randint(-200, 200)
        if y is None:
            y = random.randint(-200, 200)

        node = ShaderNodeWidget(node_type, name, x, y)
        self.scene.addItem(node)
        self.nodes.append(node)

    def _generate_shader(self):
        """Generate GLSL code from the node graph."""
        # Simple generation based on nodes present
        code_lines = [
            "#version 330 core",
            "uniform sampler2D u_texture;",
            "uniform float brightness;",
            "uniform float contrast;",
            "uniform float saturation;",
            "in vec2 v_uv;",
            "out vec4 f_color;",
            "",
            "void main() {",
            "    vec4 color = texture(u_texture, v_uv);",
        ]

        # Check which nodes are present and add their effects
        for node in self.nodes:
            if node.node_type == "brightness":
                code_lines.append("    color.rgb += brightness;")
            elif node.node_type == "contrast":
                code_lines.append("    color.rgb = (color.rgb - 0.5) * contrast + 0.5;")
            elif node.node_type == "saturation":
                code_lines.append("    float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));")
                code_lines.append("    color.rgb = mix(vec3(gray), color.rgb, saturation);")

        code_lines.extend([
            "    f_color = vec4(clamp(color.rgb, 0.0, 1.0), color.a);",
            "}"
        ])

        shader_code = "\n".join(code_lines)
        self.shader_output.setPlainText(shader_code)
        return shader_code

    def get_shader(self):
        """Get the generated shader code."""
        return self._generate_shader()


class ShaderEditorDialog(QtWidgets.QDialog):
    """Custom shader editor dialog with GLSL code editor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Shader Editor")
        self.resize(800, 600)
        self.uniforms = {}

        layout = QtWidgets.QVBoxLayout(self)

        # Shader name
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Shader Name:"))
        self.name_edit = QtWidgets.QLineEdit("My Custom Shader")
        name_row.addWidget(self.name_edit)
        layout.addLayout(name_row)

        # Split view: code editor and uniform editor
        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Code editor
        code_widget = QtWidgets.QWidget()
        code_layout = QtWidgets.QVBoxLayout(code_widget)
        code_layout.addWidget(QtWidgets.QLabel("Fragment Shader Code (GLSL):"))

        self.code_edit = QtWidgets.QPlainTextEdit()
        self.code_edit.setFont(QtGui.QFont("Consolas", 10))
        self.code_edit.setPlainText(self._get_template())
        code_layout.addWidget(self.code_edit)

        splitter.addWidget(code_widget)

        # Uniform editor
        uniform_widget = QtWidgets.QWidget()
        uniform_layout = QtWidgets.QVBoxLayout(uniform_widget)
        uniform_layout.addWidget(QtWidgets.QLabel("Custom Uniforms:"))

        self.uniform_list = QtWidgets.QListWidget()
        uniform_layout.addWidget(self.uniform_list)

        add_uniform_btn = QtWidgets.QPushButton("Add Uniform")
        add_uniform_btn.clicked.connect(self._add_uniform)
        uniform_layout.addWidget(add_uniform_btn)

        splitter.addWidget(uniform_widget)
        splitter.setSizes([500, 300])

        layout.addWidget(splitter)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()

        validate_btn = QtWidgets.QPushButton("Validate")
        validate_btn.clicked.connect(self._validate)
        btn_row.addWidget(validate_btn)

        btn_row.addStretch()

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QtWidgets.QPushButton("Save Shader")
        save_btn.clicked.connect(self.accept)
        save_btn.setStyleSheet("background-color: #2d5a2d;")
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

    def _get_template(self):
        """Get a template shader code."""
        return '''#version 330 core
uniform sampler2D u_texture;
uniform float brightness;
uniform float contrast;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec4 color = texture(u_texture, v_uv);

    // Apply brightness
    color.rgb += brightness;

    // Apply contrast
    color.rgb = (color.rgb - 0.5) * contrast + 0.5;

    f_color = vec4(clamp(color.rgb, 0.0, 1.0), color.a);
}
'''

    def _add_uniform(self):
        """Add a new uniform."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add Uniform")
        layout = QtWidgets.QFormLayout(dialog)

        name_edit = QtWidgets.QLineEdit()
        layout.addRow("Name:", name_edit)

        min_spin = QtWidgets.QDoubleSpinBox()
        min_spin.setRange(-1000, 1000)
        min_spin.setValue(0.0)
        layout.addRow("Min:", min_spin)

        max_spin = QtWidgets.QDoubleSpinBox()
        max_spin.setRange(-1000, 1000)
        max_spin.setValue(1.0)
        layout.addRow("Max:", max_spin)

        default_spin = QtWidgets.QDoubleSpinBox()
        default_spin.setRange(-1000, 1000)
        default_spin.setValue(0.5)
        layout.addRow("Default:", default_spin)

        btn_row = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(add_btn)
        layout.addRow(btn_row)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            name = name_edit.text().strip()
            if name:
                self.uniforms[name] = {
                    "min": min_spin.value(),
                    "max": max_spin.value(),
                    "default": default_spin.value(),
                    "step": 0.01
                }
                self.uniform_list.addItem(f"{name}: [{min_spin.value()}, {max_spin.value()}] = {default_spin.value()}")

    def _validate(self):
        """Validate the shader code."""
        # Basic validation - just check for required elements
        code = self.code_edit.toPlainText()
        errors = []

        if "#version" not in code:
            errors.append("Missing #version directive")
        if "void main()" not in code:
            errors.append("Missing main() function")
        if "f_color" not in code:
            errors.append("Missing f_color output")

        if errors:
            QtWidgets.QMessageBox.warning(self, "Validation Errors", "\n".join(errors))
        else:
            QtWidgets.QMessageBox.information(self, "Valid", "Shader code appears valid!")

    def get_shader(self):
        """Get the shader name and code."""
        return self.name_edit.text().strip(), self.code_edit.toPlainText()

    def get_uniforms(self):
        """Get the custom uniforms."""
        # Also extract uniforms from code
        code = self.code_edit.toPlainText()
        for line in code.split('\n'):
            if line.strip().startswith('uniform float') and 'u_texture' not in line:
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[2].rstrip(';')
                    if name not in self.uniforms:
                        self.uniforms[name] = {
                            "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01
                        }
        return self.uniforms


class ShaderStudio(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shader Studio")
        self.resize(1400, 900)
        self.param_widgets = {}
        self.current_params_cache = {}  # Cache params when switching shaders
        self.is_fullscreen = False
        self.custom_shaders = {}  # User-defined custom shaders

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ddd; }
            QPushButton {
                background-color: #3a3a3a; color: #fff;
                border: 1px solid #555; padding: 6px 12px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:disabled { background-color: #2a2a2a; color: #666; }
            QComboBox {
                background-color: #2a2a2a; border: 1px solid #555;
                padding: 5px; border-radius: 4px;
            }
            QComboBox QAbstractItemView { background-color: #2a2a2a; }
            QSlider::groove:horizontal {
                border: 1px solid #555; height: 6px;
                background: #2a2a2a; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5a9fff; width: 14px;
                margin: -4px 0; border-radius: 7px;
            }
            QGroupBox {
                background-color: #252525; border: 1px solid #3a3a3a;
                border-radius: 6px; margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; padding: 0 5px; color: #888;
            }
            QLabel { color: #ccc; }
            QCheckBox { color: #ccc; }
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: #2a2a2a; border: 1px solid #555;
                padding: 5px; border-radius: 4px; color: #fff;
            }
            QSplitter::handle { background-color: #3a3a3a; }
            QSplitter::handle:horizontal { width: 3px; }
            QMenuBar { background-color: #2a2a2a; color: #ddd; }
            QMenuBar::item:selected { background-color: #4a4a4a; }
            QMenu { background-color: #2a2a2a; color: #ddd; border: 1px solid #555; }
            QMenu::item:selected { background-color: #4a4a4a; }
            QToolTip {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 4px;
            }
        """)

        # Create menu bar
        self._create_menu_bar()

        # Create main splitter for resizable panels
        self.main_splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left panel (canvas area)
        left_widget = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()

        # Load buttons
        load_img_btn = QtWidgets.QPushButton("Load Image")
        load_img_btn.clicked.connect(self.load_image)
        load_img_btn.setStyleSheet("background-color: #3a6a9a;")
        toolbar.addWidget(load_img_btn)

        load_3d_btn = QtWidgets.QPushButton("Load 3D Model")
        load_3d_btn.clicked.connect(self.load_3d_model)
        load_3d_btn.setStyleSheet("background-color: #6a3a9a;")
        toolbar.addWidget(load_3d_btn)

        # Primitive shapes dropdown
        self.primitive_combo = QtWidgets.QComboBox()
        self.primitive_combo.addItems(["Load Primitive...", "Sphere", "Cube"])
        self.primitive_combo.currentTextChanged.connect(self._load_primitive)
        self.primitive_combo.setFixedWidth(120)
        toolbar.addWidget(self.primitive_combo)

        toolbar.addStretch()

        # 2D mode button
        self.mode_2d_btn = QtWidgets.QPushButton("2D Mode")
        self.mode_2d_btn.clicked.connect(self._switch_to_2d)
        self.mode_2d_btn.setEnabled(False)
        toolbar.addWidget(self.mode_2d_btn)

        left.addLayout(toolbar)

        # Canvas
        self.canvas = ShaderCanvas()
        left.addWidget(self.canvas, 1)

        # 3D Controls (hidden by default)
        self.controls_3d = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(self.controls_3d)
        controls_layout.setContentsMargins(0, 5, 0, 0)

        # Auto-rotate checkbox
        self.auto_rotate_cb = QtWidgets.QCheckBox("Auto Rotate")
        self.auto_rotate_cb.toggled.connect(self._toggle_auto_rotate)
        controls_layout.addWidget(self.auto_rotate_cb)

        # Rotation speed
        controls_layout.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 50)
        self.speed_slider.setValue(10)
        self.speed_slider.setFixedWidth(80)
        self.speed_slider.valueChanged.connect(self._update_rotation_speed)
        controls_layout.addWidget(self.speed_slider)

        # Zoom
        controls_layout.addWidget(QtWidgets.QLabel("Zoom:"))
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 100)
        self.zoom_slider.setValue(25)
        self.zoom_slider.setFixedWidth(80)
        self.zoom_slider.valueChanged.connect(self._update_zoom)
        controls_layout.addWidget(self.zoom_slider)

        controls_layout.addStretch()
        self.controls_3d.hide()
        left.addWidget(self.controls_3d)

        # Add left widget to splitter
        self.main_splitter.addWidget(left_widget)

        # Right: Inspector
        inspector = QtWidgets.QWidget()
        inspector.setMinimumWidth(280)
        inspector.setMaximumWidth(500)
        inspector.setStyleSheet("background-color: #252525; border-left: 1px solid #333;")
        insp_layout = QtWidgets.QVBoxLayout(inspector)
        insp_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QtWidgets.QLabel("Shader Inspector")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fff;")
        insp_layout.addWidget(title)

        # Image info
        self.image_info = QtWidgets.QLabel("No image loaded")
        self.image_info.setStyleSheet("color: #888; font-size: 11px;")
        insp_layout.addWidget(self.image_info)

        # GIF Controls (hidden by default)
        self.gif_controls = QtWidgets.QWidget()
        gif_layout = QtWidgets.QHBoxLayout(self.gif_controls)
        gif_layout.setContentsMargins(0, 5, 0, 5)

        self.gif_play_btn = QtWidgets.QPushButton("⏸")
        self.gif_play_btn.setFixedWidth(30)
        self.gif_play_btn.setToolTip("Play/Pause GIF")
        self.gif_play_btn.clicked.connect(self._toggle_gif_playback)
        gif_layout.addWidget(self.gif_play_btn)

        self.gif_frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gif_frame_slider.setMinimum(0)
        self.gif_frame_slider.setMaximum(0)
        self.gif_frame_slider.valueChanged.connect(self._on_gif_frame_changed)
        self.gif_frame_slider.setToolTip("Scrub through GIF frames")
        gif_layout.addWidget(self.gif_frame_slider)

        self.gif_frame_label = QtWidgets.QLabel("0/0")
        self.gif_frame_label.setFixedWidth(50)
        self.gif_frame_label.setStyleSheet("color: #888; font-size: 10px;")
        gif_layout.addWidget(self.gif_frame_label)

        self.gif_controls.hide()
        insp_layout.addWidget(self.gif_controls)

        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #444;")
        insp_layout.addWidget(line)

        # Category filter
        cat_row = QtWidgets.QHBoxLayout()
        cat_row.addWidget(QtWidgets.QLabel("Category:"))
        self.category_combo = QtWidgets.QComboBox()
        categories = ["All", "Custom Presets"] + sorted(set(s.get("category", "Other") for s in SHADERS.values()))
        self.category_combo.addItems(categories)
        self.category_combo.currentTextChanged.connect(self._filter_shaders)
        cat_row.addWidget(self.category_combo)
        insp_layout.addLayout(cat_row)

        # Shader selection
        shader_row = QtWidgets.QHBoxLayout()
        shader_row.addWidget(QtWidgets.QLabel("Shader:"))
        self.shader_combo = QtWidgets.QComboBox()
        self.shader_combo.addItems(SHADERS.keys())
        self.shader_combo.currentTextChanged.connect(self._on_shader_changed)
        shader_row.addWidget(self.shader_combo)
        insp_layout.addLayout(shader_row)

        # AI Model selector
        ai_row = QtWidgets.QHBoxLayout()
        ai_row.addWidget(QtWidgets.QLabel("Model:"))
        self.ai_model_combo = QtWidgets.QComboBox()
        self.ai_model_combo.addItem("Select AI Model")  # Default placeholder
        self.ai_model_combo.addItems(AI_MODELS.keys())
        self.ai_model_combo.currentTextChanged.connect(self._on_ai_model_changed)
        ai_row.addWidget(self.ai_model_combo)
        insp_layout.addLayout(ai_row)

        # AI Prompt container (hidden by default until model is selected)
        self.ai_prompt_container = QtWidgets.QWidget()
        ai_prompt_layout = QtWidgets.QVBoxLayout(self.ai_prompt_container)
        ai_prompt_layout.setContentsMargins(0, 0, 0, 0)

        # AI Prompt input
        prompt_label = QtWidgets.QLabel("Describe the effect you want:")
        prompt_label.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 4px;")
        ai_prompt_layout.addWidget(prompt_label)

        self.ai_prompt_input = QtWidgets.QTextEdit()
        self.ai_prompt_input.setPlaceholderText("Example: Apply an outline shader that detects edges and creates a sketch-like effect with thick black lines...")
        self.ai_prompt_input.setMaximumHeight(80)
        self.ai_prompt_input.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
                color: #ddd;
                font-size: 11px;
            }
            QTextEdit:focus {
                border-color: #4a6fa5;
            }
        """)
        ai_prompt_layout.addWidget(self.ai_prompt_input)

        # AI Action buttons row
        ai_btn_row = QtWidgets.QHBoxLayout()

        generate_ai_btn = QtWidgets.QPushButton("Generate Effect")
        generate_ai_btn.clicked.connect(self._generate_ai_effect)
        generate_ai_btn.setStyleSheet("background-color: #4a6fa5; font-weight: bold;")
        generate_ai_btn.setToolTip("Use AI to interpret your prompt and apply shader effects")
        ai_btn_row.addWidget(generate_ai_btn)

        suggest_ai_btn = QtWidgets.QPushButton("Suggest")
        suggest_ai_btn.setFixedWidth(70)
        suggest_ai_btn.clicked.connect(self._suggest_ai_prompts)
        suggest_ai_btn.setStyleSheet("background-color: #3d5a3d;")
        suggest_ai_btn.setToolTip("Get AI-powered suggestions for effects")
        ai_btn_row.addWidget(suggest_ai_btn)

        ai_prompt_layout.addLayout(ai_btn_row)

        # AI Response/Status
        self.ai_response_label = QtWidgets.QLabel("")
        self.ai_response_label.setWordWrap(True)
        self.ai_response_label.setStyleSheet("color: #7aa2d6; font-style: italic; padding: 2px 0; font-size: 10px;")
        self.ai_response_label.setMaximumHeight(60)
        ai_prompt_layout.addWidget(self.ai_response_label)

        # Hide the prompt container by default
        self.ai_prompt_container.hide()
        insp_layout.addWidget(self.ai_prompt_container)

        # Custom preset selection (for user presets)
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        self._refresh_preset_combo()
        self.preset_combo.currentTextChanged.connect(self._load_user_preset)
        preset_row.addWidget(self.preset_combo)
        insp_layout.addLayout(preset_row)

        # Description
        self.description = QtWidgets.QLabel("")
        self.description.setWordWrap(True)
        self.description.setStyleSheet("color: #888; font-style: italic; padding: 5px 0;")
        insp_layout.addWidget(self.description)

        # Layer Opacity slider (controls the selected layer's opacity)
        opacity_row = QtWidgets.QHBoxLayout()
        opacity_label = QtWidgets.QLabel("Layer Opacity:")
        opacity_label.setStyleSheet("color: #ccc; font-size: 11px;")
        opacity_row.addWidget(opacity_label)

        self.layer_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.layer_opacity_slider.setMinimum(0)
        self.layer_opacity_slider.setMaximum(100)
        self.layer_opacity_slider.setValue(100)
        self.layer_opacity_slider.setToolTip("Opacity of the selected layer")
        self.layer_opacity_slider.valueChanged.connect(self._on_layer_opacity_changed)
        opacity_row.addWidget(self.layer_opacity_slider)

        self.layer_opacity_value = QtWidgets.QLabel("100%")
        self.layer_opacity_value.setFixedWidth(40)
        self.layer_opacity_value.setStyleSheet("color: #aaa; font-size: 11px;")
        opacity_row.addWidget(self.layer_opacity_value)

        insp_layout.addLayout(opacity_row)

        # Blend mode dropdown
        blend_row = QtWidgets.QHBoxLayout()
        blend_label = QtWidgets.QLabel("Blend Mode:")
        blend_label.setStyleSheet("color: #ccc; font-size: 11px;")
        blend_row.addWidget(blend_label)

        self.blend_mode_combo = QtWidgets.QComboBox()
        for mode_name in BLEND_MODE_NAMES:
            self.blend_mode_combo.addItem(mode_name.replace('_', ' ').title())
        self.blend_mode_combo.currentTextChanged.connect(self._on_blend_mode_changed)
        blend_row.addWidget(self.blend_mode_combo)

        insp_layout.addLayout(blend_row)

        # Parameters section
        params_label = QtWidgets.QLabel("Parameters")
        params_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #fff;")
        insp_layout.addWidget(params_label)

        # Scroll area for parameters
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self.params_widget = QtWidgets.QWidget()
        self.params_layout = QtWidgets.QVBoxLayout(self.params_widget)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.addStretch()
        scroll.setWidget(self.params_widget)
        insp_layout.addWidget(scroll, 1)

        # Separator before layers
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line2.setStyleSheet("background-color: #444;")
        insp_layout.addWidget(line2)

        # Shader Layer Panel
        self.layer_panel = ShaderLayerPanel()
        self.layer_panel.layersChanged.connect(self._on_layers_changed)
        self.layer_panel.layerSelected.connect(self._on_layer_selected)
        insp_layout.addWidget(self.layer_panel)

        # Separator after layers
        line3 = QtWidgets.QFrame()
        line3.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line3.setStyleSheet("background-color: #444;")
        insp_layout.addWidget(line3)

        # Preset management buttons
        preset_btns = QtWidgets.QHBoxLayout()

        save_preset_btn = QtWidgets.QPushButton("Save Preset")
        save_preset_btn.clicked.connect(self._save_preset)
        save_preset_btn.setStyleSheet("background-color: #5a5a2d;")
        preset_btns.addWidget(save_preset_btn)

        delete_preset_btn = QtWidgets.QPushButton("Delete Preset")
        delete_preset_btn.clicked.connect(self._delete_preset)
        delete_preset_btn.setStyleSheet("background-color: #5a2d2d;")
        preset_btns.addWidget(delete_preset_btn)

        insp_layout.addLayout(preset_btns)

        # Reset button
        reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_params)
        insp_layout.addWidget(reset_btn)

        # Export buttons row
        export_row = QtWidgets.QHBoxLayout()

        export_btn = QtWidgets.QPushButton("Export")
        export_btn.clicked.connect(self.export_image)
        export_btn.setStyleSheet("background-color: #2d5a2d;")
        export_row.addWidget(export_btn)

        batch_btn = QtWidgets.QPushButton("Batch")
        batch_btn.clicked.connect(self.batch_process)
        batch_btn.setStyleSheet("background-color: #2d5a5a;")
        export_row.addWidget(batch_btn)

        upscale_btn = QtWidgets.QPushButton("Upscale")
        upscale_btn.clicked.connect(self.upscale_image)
        upscale_btn.setStyleSheet("background-color: #5a2d5a;")
        export_row.addWidget(upscale_btn)

        insp_layout.addLayout(export_row)

        # Bake Pass section
        bake_sep = QtWidgets.QFrame()
        bake_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        bake_sep.setStyleSheet("background-color: #444;")
        insp_layout.addWidget(bake_sep)

        bake_label = QtWidgets.QLabel("Shader Chain")
        bake_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #fff;")
        insp_layout.addWidget(bake_label)

        bake_row = QtWidgets.QHBoxLayout()

        bake_pass_btn = QtWidgets.QPushButton("Bake Pass")
        bake_pass_btn.clicked.connect(self._bake_pass)
        bake_pass_btn.setStyleSheet("background-color: #5a4a2d;")
        bake_pass_btn.setToolTip("Commit the current shader effect and start a new pass on top")
        bake_row.addWidget(bake_pass_btn)

        reset_chain_btn = QtWidgets.QPushButton("Reset Chain")
        reset_chain_btn.clicked.connect(self._reset_chain)
        reset_chain_btn.setStyleSheet("background-color: #5a2d2d;")
        reset_chain_btn.setToolTip("Reload the original image and clear the shader chain")
        bake_row.addWidget(reset_chain_btn)

        insp_layout.addLayout(bake_row)

        self.chain_display_label = QtWidgets.QLabel("No chain")
        self.chain_display_label.setWordWrap(True)
        self.chain_display_label.setStyleSheet("color: #aaa; font-size: 11px; padding: 2px;")
        insp_layout.addWidget(self.chain_display_label)

        # Undo/Redo buttons row
        undo_row = QtWidgets.QHBoxLayout()

        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.clicked.connect(self._undo)
        self.undo_btn.setEnabled(False)
        undo_row.addWidget(self.undo_btn)

        self.redo_btn = QtWidgets.QPushButton("Redo")
        self.redo_btn.clicked.connect(self._redo)
        self.redo_btn.setEnabled(False)
        undo_row.addWidget(self.redo_btn)

        insp_layout.addLayout(undo_row)

        # Tools row (color picker, randomize, copy/paste)
        tools_row = QtWidgets.QHBoxLayout()

        color_picker_btn = QtWidgets.QPushButton("🎨")
        color_picker_btn.setToolTip("Color Picker (click on image to sample color)")
        color_picker_btn.setFixedWidth(35)
        color_picker_btn.clicked.connect(self._enable_color_picker)
        tools_row.addWidget(color_picker_btn)

        randomize_btn = QtWidgets.QPushButton("🎲")
        randomize_btn.setToolTip("Randomize Parameters (Ctrl+R)")
        randomize_btn.setFixedWidth(35)
        randomize_btn.clicked.connect(self._randomize_params)
        tools_row.addWidget(randomize_btn)

        copy_btn = QtWidgets.QPushButton("📋")
        copy_btn.setToolTip("Copy Parameters (Ctrl+C)")
        copy_btn.setFixedWidth(35)
        copy_btn.clicked.connect(self._copy_params)
        tools_row.addWidget(copy_btn)

        paste_btn = QtWidgets.QPushButton("📥")
        paste_btn.setToolTip("Paste Parameters (Ctrl+V)")
        paste_btn.setFixedWidth(35)
        paste_btn.clicked.connect(self._paste_params)
        tools_row.addWidget(paste_btn)

        tools_row.addStretch()
        insp_layout.addLayout(tools_row)

        # Color display (for color picker)
        self.color_display = QtWidgets.QLabel("")
        self.color_display.setFixedHeight(30)
        self.color_display.setStyleSheet("background-color: #333; border: 1px solid #555; border-radius: 4px;")
        self.color_display.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.color_display.hide()
        insp_layout.addWidget(self.color_display)

        # Add inspector to splitter
        self.main_splitter.addWidget(inspector)
        self.main_splitter.setSizes([900, 350])

        # Set splitter as central widget
        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(5, 5, 5, 5)
        central_layout.addWidget(self.main_splitter)
        self.setCentralWidget(central)

        # Connect color picker signal
        self.canvas.colorPicked.connect(self._on_color_picked)

        # Connect texture loaded signal - ensures base layer is always created
        self.canvas.textureLoaded.connect(self._on_texture_loaded)

        # Setup keyboard shortcuts
        self._setup_shortcuts()

        # Initialize UI
        self._update_params_ui("Original")

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QtGui.QAction("&Open Image...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)

        open_3d_action = QtGui.QAction("Open 3D Model...", self)
        open_3d_action.triggered.connect(self.load_3d_model)
        file_menu.addAction(open_3d_action)

        file_menu.addSeparator()

        # Recent files submenu
        self.recent_menu = file_menu.addMenu("Recent Files")
        self._update_recent_menu()

        file_menu.addSeparator()

        export_action = QtGui.QAction("&Export Image...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)

        batch_action = QtGui.QAction("Batch Process...", self)
        batch_action.triggered.connect(self.batch_process)
        file_menu.addAction(batch_action)

        file_menu.addSeparator()

        # Video processing
        video_action = QtGui.QAction("Process Video...", self)
        video_action.triggered.connect(self.process_video)
        video_action.setEnabled(VIDEO_AVAILABLE)
        file_menu.addAction(video_action)

        gif_action = QtGui.QAction("Export GIF Animation...", self)
        gif_action.triggered.connect(self.export_gif)
        file_menu.addAction(gif_action)

        file_menu.addSeparator()

        exit_action = QtGui.QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QtGui.QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._undo)
        edit_menu.addAction(undo_action)

        redo_action = QtGui.QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self._redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        copy_action = QtGui.QAction("&Copy Parameters", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self._copy_params)
        edit_menu.addAction(copy_action)

        paste_action = QtGui.QAction("&Paste Parameters", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self._paste_params)
        edit_menu.addAction(paste_action)

        edit_menu.addSeparator()

        randomize_action = QtGui.QAction("&Randomize Parameters", self)
        randomize_action.setShortcut("Ctrl+R")
        randomize_action.triggered.connect(self._randomize_params)
        edit_menu.addAction(randomize_action)

        reset_action = QtGui.QAction("Reset to &Defaults", self)
        reset_action.setShortcut("Ctrl+D")
        reset_action.triggered.connect(self._reset_params)
        edit_menu.addAction(reset_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        fullscreen_action = QtGui.QAction("&Fullscreen Preview", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        view_menu.addSeparator()

        wireframe_action = QtGui.QAction("&Wireframe Mode", self)
        wireframe_action.setCheckable(True)
        wireframe_action.triggered.connect(lambda checked: self.canvas.set_wireframe_mode(checked))
        view_menu.addAction(wireframe_action)

        normals_action = QtGui.QAction("Show &Normals", self)
        normals_action.setCheckable(True)
        normals_action.triggered.connect(lambda checked: self.canvas.set_show_normals(checked))
        view_menu.addAction(normals_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        color_picker_action = QtGui.QAction("&Color Picker", self)
        color_picker_action.triggered.connect(self._enable_color_picker)
        tools_menu.addAction(color_picker_action)

        upscale_action = QtGui.QAction("&Upscale Image...", self)
        upscale_action.triggered.connect(self.upscale_image)
        tools_menu.addAction(upscale_action)

        tools_menu.addSeparator()

        lut_action = QtGui.QAction("Load &LUT (.cube)...", self)
        lut_action.triggered.connect(self._load_lut)
        tools_menu.addAction(lut_action)

        overlay_action = QtGui.QAction("Add Texture &Overlay...", self)
        overlay_action.triggered.connect(self._add_overlay)
        tools_menu.addAction(overlay_action)

        tools_menu.addSeparator()

        shader_editor_action = QtGui.QAction("&Custom Shader Editor...", self)
        shader_editor_action.triggered.connect(self._open_shader_editor)
        tools_menu.addAction(shader_editor_action)

        node_graph_action = QtGui.QAction("Shader &Node Graph...", self)
        node_graph_action.triggered.connect(self._open_node_graph)
        tools_menu.addAction(node_graph_action)

        tools_menu.addSeparator()

        # AI Quick Effects submenu
        ai_effects_menu = tools_menu.addMenu("AI &Quick Effects")
        for preset_name, prompt in QUICK_AI_PROMPTS.items():
            action = QtGui.QAction(preset_name, self)
            action.setToolTip(prompt)
            action.triggered.connect(lambda checked, p=preset_name: self._apply_quick_ai_preset(p))
            ai_effects_menu.addAction(action)

        ai_effects_menu.addSeparator()
        ai_suggest_action = QtGui.QAction("More Suggestions...", self)
        ai_suggest_action.triggered.connect(self._suggest_ai_prompts)
        ai_effects_menu.addAction(ai_suggest_action)

        # Presets menu
        presets_menu = menubar.addMenu("&Presets")

        import_preset_action = QtGui.QAction("&Import Presets...", self)
        import_preset_action.triggered.connect(self._import_presets)
        presets_menu.addAction(import_preset_action)

        export_preset_action = QtGui.QAction("&Export Presets...", self)
        export_preset_action.triggered.connect(self._export_presets)
        presets_menu.addAction(export_preset_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        shortcuts_action = QtGui.QAction("&Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

        about_action = QtGui.QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _update_recent_menu(self):
        """Update the recent files menu."""
        self.recent_menu.clear()
        if not RECENT_FILES:
            no_recent = QtGui.QAction("No recent files", self)
            no_recent.setEnabled(False)
            self.recent_menu.addAction(no_recent)
        else:
            for path in RECENT_FILES:
                action = QtGui.QAction(os.path.basename(path), self)
                action.setToolTip(path)
                action.triggered.connect(lambda checked, p=path: self._open_recent(p))
                self.recent_menu.addAction(action)

            self.recent_menu.addSeparator()
            clear_action = QtGui.QAction("Clear Recent Files", self)
            clear_action.triggered.connect(self._clear_recent)
            self.recent_menu.addAction(clear_action)

    def _open_recent(self, path):
        """Open a recent file."""
        if os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.obj', '.gltf', '.glb']:
                if self.canvas.load_3d_model(path):
                    self.image_info.setText(f"3D Model: {os.path.basename(path)}")
                    self.controls_3d.show()
                    self.mode_2d_btn.setEnabled(True)
                    self._hide_gif_controls()
                    add_recent_file(path)
            else:
                self.canvas.set_2d_mode()
                self.controls_3d.hide()
                self.mode_2d_btn.setEnabled(False)
                if self.canvas.load_texture(path):
                    self._on_image_loaded(path)
                    add_recent_file(path)
            self._update_recent_menu()

    def _on_texture_loaded(self, path):
        """Signal handler: called whenever canvas loads a texture, regardless of source.
        Ensures _on_image_loaded is always called."""
        # Skip during batch processing to avoid disrupting the batch loop
        if getattr(self, '_batch_processing', False):
            return
        # Only call _on_image_loaded if base layer doesn't exist yet
        if not self.layer_panel.base_layer:
            self._on_image_loaded(path)

    def _on_image_loaded(self, path):
        """Called after an image is successfully loaded. Sets up layers and UI."""
        size = self.canvas.image_size
        if self.canvas.is_gif and len(self.canvas.gif_frames) > 1:
            frame_count = len(self.canvas.gif_frames)
            self.image_info.setText(f"{os.path.basename(path)} ({size[0]}x{size[1]}) - {frame_count} frames")
            self._show_gif_controls()
        else:
            self.image_info.setText(f"{os.path.basename(path)} ({size[0]}x{size[1]})")
            self._hide_gif_controls()

        # Create the base layer with the current shader
        current_shader = self.shader_combo.currentText()
        self.layer_panel.create_base_layer(current_shader)

    def _clear_recent(self):
        """Clear recent files list."""
        global RECENT_FILES
        RECENT_FILES = []
        save_settings()
        self._update_recent_menu()

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # F11 for fullscreen
        fullscreen_shortcut = QtGui.QShortcut(QtGui.QKeySequence("F11"), self)
        fullscreen_shortcut.activated.connect(self._toggle_fullscreen)

        # Escape to exit fullscreen
        escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        escape_shortcut.activated.connect(self._exit_fullscreen)

    # --- COLOR PICKER ---
    def _enable_color_picker(self):
        """Enable color picker mode on the canvas."""
        self.canvas.enable_color_picker()

    def _on_color_picked(self, r, g, b, a):
        """Handle color picked from canvas."""
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.color_display.setText(f"RGB({r}, {g}, {b}) | {hex_color}")
        self.color_display.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"color: {'#000' if (r + g + b) > 384 else '#fff'}; "
            f"border: 1px solid #555; border-radius: 4px; font-weight: bold;"
        )
        self.color_display.show()

        # Copy to clipboard
        QtWidgets.QApplication.clipboard().setText(hex_color)

    # --- PARAMETER TOOLS ---
    def _randomize_params(self):
        """Randomize all parameters."""
        shader_def = SHADERS.get(self.canvas.current_preset, {})
        uniforms = shader_def.get("uniforms", {})

        for name, props in uniforms.items():
            min_v, max_v = props["min"], props["max"]
            value = random.uniform(min_v, max_v)
            self.canvas.set_param(name, value)
            if name in self.param_widgets:
                self.param_widgets[name].set_value(value)

    def _copy_params(self):
        """Copy current parameters to clipboard."""
        global CLIPBOARD_PARAMS
        CLIPBOARD_PARAMS = {
            "shader": self.canvas.current_preset,
            "params": copy.deepcopy(self.canvas.params)
        }
        QtWidgets.QApplication.clipboard().setText(json.dumps(CLIPBOARD_PARAMS, indent=2))

    def _paste_params(self):
        """Paste parameters from clipboard."""
        global CLIPBOARD_PARAMS
        if CLIPBOARD_PARAMS and CLIPBOARD_PARAMS["shader"] == self.canvas.current_preset:
            for name, value in CLIPBOARD_PARAMS["params"].items():
                self.canvas.set_param(name, value)
                if name in self.param_widgets:
                    self.param_widgets[name].set_value(value)

    # --- FULLSCREEN ---
    def _toggle_fullscreen(self):
        """Toggle fullscreen preview mode."""
        if self.is_fullscreen:
            self._exit_fullscreen()
        else:
            self.is_fullscreen = True
            self.menuBar().hide()
            self.showFullScreen()

    def _exit_fullscreen(self):
        """Exit fullscreen mode."""
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.menuBar().show()
            self.showNormal()

    # --- DRAG & DROP ---
    def dragEnterEvent(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle drop event."""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            ext = os.path.splitext(path)[1].lower()

            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
                self.canvas.set_2d_mode()
                self.controls_3d.hide()
                self.mode_2d_btn.setEnabled(False)
                if self.canvas.load_texture(path):
                    self._on_image_loaded(path)
                    add_recent_file(path)
                    self._update_recent_menu()
            elif ext in ['.obj', '.gltf', '.glb']:
                if self.canvas.load_3d_model(path):
                    self.image_info.setText(f"3D Model: {os.path.basename(path)}")
                    self.controls_3d.show()
                    self.mode_2d_btn.setEnabled(True)
                    self._hide_gif_controls()
                    add_recent_file(path)
                    self._update_recent_menu()
            elif ext == '.cube':
                self._apply_lut(path)

    # --- LUT SUPPORT ---
    def _load_lut(self):
        """Load a .cube LUT file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load LUT", "", "Cube LUT (*.cube);;All Files (*)"
        )
        if path:
            self._apply_lut(path)

    def _apply_lut(self, path):
        """Apply a LUT to the current image."""
        lut = load_cube_lut(path)
        if lut is not None:
            self.canvas.current_lut = lut
            QtWidgets.QMessageBox.information(
                self, "LUT Loaded",
                f"LUT '{os.path.basename(path)}' loaded successfully!"
            )

    # --- TEXTURE OVERLAYS ---
    def _add_overlay(self):
        """Add a texture overlay."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Overlay Texture", "",
            "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if not path:
            return

        # Show blend mode dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Overlay Settings")
        layout = QtWidgets.QVBoxLayout(dialog)

        # Blend mode
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Blend Mode:"))
        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItems(["multiply", "screen", "overlay", "soft_light", "add"])
        mode_row.addWidget(mode_combo)
        layout.addLayout(mode_row)

        # Opacity
        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.addWidget(QtWidgets.QLabel("Opacity:"))
        opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        opacity_slider.setRange(0, 100)
        opacity_slider.setValue(50)
        opacity_row.addWidget(opacity_slider)
        layout.addLayout(opacity_row)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.canvas.overlay_blend_mode = mode_combo.currentText()
            self.canvas.overlay_opacity = opacity_slider.value() / 100.0
            # Store overlay path for later application
            self.canvas.overlay_texture = path

    # --- PRESET IMPORT/EXPORT ---
    def _import_presets(self):
        """Import presets from a JSON file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Presets", "", "JSON Files (*.json)"
        )
        if path:
            try:
                with open(path, 'r') as f:
                    imported = json.load(f)
                count = 0
                for name, data in imported.items():
                    if name not in USER_PRESETS:
                        USER_PRESETS[name] = data
                        count += 1
                save_user_presets()
                self._refresh_preset_combo()
                QtWidgets.QMessageBox.information(
                    self, "Import Complete",
                    f"Imported {count} presets!"
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Import Failed", f"Error: {e}")

    def _export_presets(self):
        """Export presets to a JSON file."""
        if not USER_PRESETS:
            QtWidgets.QMessageBox.warning(self, "No Presets", "No custom presets to export.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Presets", "shader_presets.json", "JSON Files (*.json)"
        )
        if path:
            try:
                with open(path, 'w') as f:
                    json.dump(USER_PRESETS, f, indent=2)
                QtWidgets.QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {len(USER_PRESETS)} presets to {path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export Failed", f"Error: {e}")

    # --- CUSTOM SHADER EDITOR ---
    def _open_shader_editor(self):
        """Open the custom shader editor dialog."""
        dialog = ShaderEditorDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            shader_name, shader_code = dialog.get_shader()
            if shader_name and shader_code:
                # Add to custom shaders
                self.custom_shaders[shader_name] = {
                    "category": "Custom",
                    "description": "User-defined custom shader",
                    "uniforms": dialog.get_uniforms(),
                    "frag": shader_code
                }
                # Add to SHADERS
                SHADERS[shader_name] = self.custom_shaders[shader_name]
                # Refresh UI
                self._filter_shaders(self.category_combo.currentText())

    def _open_node_graph(self):
        """Open the visual node graph shader editor."""
        dialog = ShaderNodeGraphDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            shader_code = dialog.get_shader()
            if shader_code:
                # Add as custom shader
                shader_name = f"Node Graph {len(self.custom_shaders) + 1}"
                self.custom_shaders[shader_name] = {
                    "category": "Custom",
                    "description": "Generated from node graph",
                    "uniforms": {
                        "brightness": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
                        "contrast": {"min": 0.0, "max": 3.0, "default": 1.0, "step": 0.01},
                        "saturation": {"min": 0.0, "max": 3.0, "default": 1.0, "step": 0.01},
                    },
                    "frag": shader_code
                }
                SHADERS[shader_name] = self.custom_shaders[shader_name]
                self._filter_shaders(self.category_combo.currentText())

                # Switch to the new shader
                idx = self.shader_combo.findText(shader_name)
                if idx >= 0:
                    self.shader_combo.setCurrentIndex(idx)

    # --- VIDEO PROCESSING ---
    def process_video(self):
        """Process a video file with the current shader."""
        if not VIDEO_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, "Not Available",
                "Video processing requires OpenCV. Install with: pip install opencv-python"
            )
            return

        # Select input video
        input_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not input_path:
            return

        # Select output path
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Processed Video", "",
            "MP4 Video (*.mp4);;AVI Video (*.avi)"
        )
        if not output_path:
            return

        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress = QtWidgets.QProgressDialog(
                "Processing video...", "Cancel", 0, total_frames, self
            )
            progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if progress.wasCanceled():
                    break

                # Convert BGR to RGBA
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

                # Load as texture
                self.canvas.makeCurrent()
                self.canvas._upload_texture(frame_rgba)
                self.canvas.image_size = (width, height)

                # Render
                processed = self.canvas.export_to_image(width, height)
                if processed is not None:
                    # Convert back to BGR
                    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGBA2BGR)
                    out.write(processed_bgr)

                frame_count += 1
                progress.setValue(frame_count)
                QtWidgets.QApplication.processEvents()

            cap.release()
            out.release()
            progress.close()

            QtWidgets.QMessageBox.information(
                self, "Complete",
                f"Video processed successfully!\nOutput: {output_path}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Video processing failed: {e}")

    # --- GIF EXPORT ---
    def export_gif(self):
        """Export an animated GIF of the 3D model rotating."""
        if not self.canvas.mode_3d:
            QtWidgets.QMessageBox.warning(
                self, "3D Mode Required",
                "GIF export is only available in 3D mode."
            )
            return

        # Settings dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("GIF Export Settings")
        layout = QtWidgets.QVBoxLayout(dialog)

        # Frames
        frames_row = QtWidgets.QHBoxLayout()
        frames_row.addWidget(QtWidgets.QLabel("Frames:"))
        frames_spin = QtWidgets.QSpinBox()
        frames_spin.setRange(10, 120)
        frames_spin.setValue(36)
        frames_row.addWidget(frames_spin)
        layout.addLayout(frames_row)

        # FPS
        fps_row = QtWidgets.QHBoxLayout()
        fps_row.addWidget(QtWidgets.QLabel("FPS:"))
        fps_spin = QtWidgets.QSpinBox()
        fps_spin.setRange(5, 30)
        fps_spin.setValue(15)
        fps_row.addWidget(fps_spin)
        layout.addLayout(fps_row)

        # Size
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Size:"))
        size_spin = QtWidgets.QSpinBox()
        size_spin.setRange(128, 1024)
        size_spin.setValue(400)
        size_spin.setSingleStep(64)
        size_row.addWidget(size_spin)
        layout.addLayout(size_row)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)
        export_btn = QtWidgets.QPushButton("Export")
        export_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(export_btn)
        layout.addLayout(btn_row)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        # Get settings
        num_frames = frames_spin.value()
        fps = fps_spin.value()
        size = size_spin.value()

        # Get output path
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save GIF", "animation.gif", "GIF (*.gif)"
        )
        if not path:
            return

        # Generate frames
        frames = []
        original_rotation = self.canvas.rotation_y

        progress = QtWidgets.QProgressDialog(
            "Generating GIF...", "Cancel", 0, num_frames, self
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        for i in range(num_frames):
            if progress.wasCanceled():
                break

            self.canvas.rotation_y = (360.0 / num_frames) * i
            self.canvas.update()
            QtWidgets.QApplication.processEvents()

            frame = self.canvas.export_to_image(size, size)
            if frame is not None:
                frames.append(Image.fromarray(frame))

            progress.setValue(i + 1)

        self.canvas.rotation_y = original_rotation
        self.canvas.update()
        progress.close()

        if frames:
            # Save GIF
            duration = int(1000 / fps)
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
            QtWidgets.QMessageBox.information(
                self, "Export Complete",
                f"GIF exported successfully!\n{len(frames)} frames at {fps} FPS\n{path}"
            )

    # --- HELP DIALOGS ---
    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        shortcuts = """
<h3>Keyboard Shortcuts</h3>
<table>
<tr><td><b>Ctrl+O</b></td><td>Open Image</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export Image</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>
<tr><td><b>Ctrl+Y</b></td><td>Redo</td></tr>
<tr><td><b>Ctrl+C</b></td><td>Copy Parameters</td></tr>
<tr><td><b>Ctrl+V</b></td><td>Paste Parameters</td></tr>
<tr><td><b>Ctrl+R</b></td><td>Randomize Parameters</td></tr>
<tr><td><b>Ctrl+D</b></td><td>Reset to Defaults</td></tr>
<tr><td><b>F11</b></td><td>Toggle Fullscreen</td></tr>
<tr><td><b>Escape</b></td><td>Exit Fullscreen</td></tr>
</table>
<h4>3D Mode (Mouse)</h4>
<table>
<tr><td><b>Left Drag</b></td><td>Orbit/Rotate</td></tr>
<tr><td><b>Middle Drag</b></td><td>Pan</td></tr>
<tr><td><b>Right Drag</b></td><td>Zoom</td></tr>
<tr><td><b>Scroll Wheel</b></td><td>Zoom</td></tr>
</table>
"""
        QtWidgets.QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)

    def _show_about(self):
        """Show about dialog."""
        about = f"""
<h2>Shader Studio</h2>
<p>Version 2.0</p>
<p>A professional shader tool for 2D and 3D rendering.</p>
<h4>Features:</h4>
<ul>
<li>{len(SHADERS)} shader presets</li>
<li>3D model support (OBJ, GLTF, GLB)</li>
<li>Video processing</li>
<li>AI upscaling</li>
<li>Custom shader editor</li>
<li>LUT support</li>
<li>And much more!</li>
</ul>
"""
        QtWidgets.QMessageBox.about(self, "About Shader Studio", about)

    def _filter_shaders(self, category):
        self.shader_combo.blockSignals(True)
        self.shader_combo.clear()

        if category == "Custom Presets":
            # Show only user presets
            for name in USER_PRESETS.keys():
                base_shader = USER_PRESETS[name].get("base_shader", "Original")
                self.shader_combo.addItem(f"{name} ({base_shader})")
        else:
            for name, shader in SHADERS.items():
                if category == "All" or shader.get("category") == category:
                    self.shader_combo.addItem(name)

        self.shader_combo.blockSignals(False)
        if self.shader_combo.count() > 0:
            self._on_shader_changed(self.shader_combo.currentText())

    def _on_shader_changed(self, name):
        if not name:
            return

        # Cache current params before switching
        if self.canvas.current_preset:
            cache_key = self.canvas.current_preset
            self.current_params_cache[cache_key] = dict(self.canvas.params)

        # Check if this is a user preset (format: "PresetName (BaseShader)")
        if " (" in name and name.endswith(")"):
            preset_name = name.split(" (")[0]
            if preset_name in USER_PRESETS:
                self._apply_user_preset(preset_name)
                return

        self.canvas.set_preset(name)
        self._update_params_ui(name)

        # Update the selected layer's shader if there is one
        if self.layer_panel.get_selected_layer():
            self.layer_panel.update_selected_layer_shader(name)

        # Restore cached params if available
        if name in self.current_params_cache:
            for param_name, value in self.current_params_cache[name].items():
                self.canvas.set_param(param_name, value)
                if param_name in self.param_widgets:
                    self.param_widgets[param_name].set_value(value)

    def _on_ai_model_changed(self, model_name):
        """Update when AI model selection changes."""
        if model_name == "Select AI Model":
            # Hide the prompt container when no model is selected
            self.ai_prompt_container.hide()
            return

        # Show the prompt container when a model is selected
        self.ai_prompt_container.show()

        if model_name in AI_MODELS:
            model = AI_MODELS[model_name]
            self.ai_response_label.setText(model.get("description", ""))

    def _on_layers_changed(self, layers_data):
        """Handle changes to the shader layer stack."""
        # Store the layer data and trigger a repaint
        self.canvas.shader_layers = layers_data if layers_data else []
        self.canvas.update()

    def _on_layer_selected(self, layer_data):
        """Handle layer selection - update the shader inspector to show this layer's settings."""
        if not layer_data:
            return

        shader_name = layer_data.get('shader', 'Original')
        params = layer_data.get('params', {})
        opacity = layer_data.get('opacity', 1.0)
        blend_mode = layer_data.get('blend_mode', 'normal')

        # Update the shader dropdown without triggering canvas update
        self.shader_combo.blockSignals(True)
        self.shader_combo.setCurrentText(shader_name)
        self.shader_combo.blockSignals(False)

        # Update the opacity slider without triggering change
        self.layer_opacity_slider.blockSignals(True)
        self.layer_opacity_slider.setValue(int(opacity * 100))
        self.layer_opacity_value.setText(f"{int(opacity * 100)}%")
        self.layer_opacity_slider.blockSignals(False)

        # Update the blend mode dropdown without triggering change
        self.blend_mode_combo.blockSignals(True)
        display_mode = blend_mode.replace('_', ' ').title()
        self.blend_mode_combo.setCurrentText(display_mode)
        self.blend_mode_combo.blockSignals(False)

        # Update the params UI for this shader
        self._update_params_ui(shader_name)

        # Set the parameter values from the layer
        for param_name, value in params.items():
            if param_name in self.param_widgets:
                self.param_widgets[param_name].set_value(value)

    def _on_layer_opacity_changed(self, value):
        """Handle opacity slider change - update selected layer."""
        opacity = value / 100.0
        self.layer_opacity_value.setText(f"{value}%")
        selected = self.layer_panel.get_selected_layer()
        if selected:
            selected.set_opacity(opacity)

    def _on_blend_mode_changed(self, text):
        """Handle blend mode dropdown change - update selected layer."""
        mode = text.lower().replace(' ', '_')
        selected = self.layer_panel.get_selected_layer()
        if selected:
            selected.set_blend_mode(mode)

    def _generate_ai_effect(self):
        """Generate and apply shader effect based on user prompt."""
        prompt = self.ai_prompt_input.toPlainText().strip()
        if not prompt:
            self.ai_response_label.setText("Please enter a description of the effect you want.")
            return

        model_name = self.ai_model_combo.currentText()
        model_config = AI_MODELS.get(model_name, {})
        model_type = model_config.get("type", "local")

        self.ai_response_label.setText("Analyzing prompt...")
        QtWidgets.QApplication.processEvents()

        if model_type == "local":
            # Use built-in pattern matching
            result = self._interpret_prompt_locally(prompt)
        else:
            # Try to use external API
            result = self._interpret_prompt_with_api(prompt, model_config)

        if result:
            self._apply_interpreted_effect(result)
        else:
            self.ai_response_label.setText("Could not interpret the prompt. Try using keywords like 'outline', 'glow', 'vintage', etc.")

    def _interpret_prompt_locally(self, prompt):
        """Interpret the prompt using local keyword matching."""
        prompt_lower = prompt.lower()
        words = prompt_lower.replace(',', ' ').replace('.', ' ').split()

        # Find matching effects
        matched_effects = []
        intensity_multiplier = 1.0

        for word in words:
            # Check for intensity modifiers
            if word in EFFECT_KEYWORDS:
                effect = EFFECT_KEYWORDS[word]
                if "multiplier" in effect:
                    intensity_multiplier = effect["multiplier"]
                elif "shader" in effect:
                    matched_effects.append(effect)

        # Also check for multi-word phrases
        for phrase, effect in EFFECT_KEYWORDS.items():
            if ' ' in phrase and phrase in prompt_lower:
                if "shader" in effect:
                    matched_effects.append(effect)

        if not matched_effects:
            # Try fuzzy matching with common variations
            for keyword, effect in EFFECT_KEYWORDS.items():
                if "shader" in effect:
                    # Check if any word starts with the keyword
                    for word in words:
                        if word.startswith(keyword[:4]) or keyword.startswith(word[:4]):
                            matched_effects.append(effect)
                            break

        if matched_effects:
            # Use the first matched effect as primary
            primary_effect = matched_effects[0].copy()

            # Apply intensity multiplier to numeric params
            if intensity_multiplier != 1.0 and "params" in primary_effect:
                modified_params = {}
                for param, value in primary_effect["params"].items():
                    if isinstance(value, (int, float)):
                        # Apply multiplier but keep within reasonable bounds
                        modified_params[param] = value * intensity_multiplier
                    else:
                        modified_params[param] = value
                primary_effect["params"] = modified_params

            return primary_effect

        return None

    def _interpret_prompt_with_api(self, prompt, model_config):
        """Interpret the prompt using an external AI API."""
        import os

        model_type = model_config.get("type")
        api_key_env = model_config.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "")

        if not api_key:
            self.ai_response_label.setText(f"API key not found. Set {api_key_env} environment variable or use Local mode.")
            # Fall back to local interpretation
            return self._interpret_prompt_locally(prompt)

        # Build the system prompt for the AI
        shader_list = list(SHADERS.keys())
        system_prompt = f"""You are a shader effect interpreter. Given a user's description of a visual effect,
respond with a JSON object containing:
- "shader": the name of the shader to use (must be one of: {', '.join(shader_list[:20])}...)
- "params": a dictionary of parameter names and values to apply

Only respond with valid JSON, no other text. Example:
{{"shader": "Sobel Edge", "params": {{"edge_strength": 1.5, "threshold": 0.1}}}}
"""

        try:
            if model_type == "openai":
                return self._call_openai_api(api_key, model_config.get("model", "gpt-3.5-turbo"), system_prompt, prompt)
            elif model_type == "anthropic":
                return self._call_anthropic_api(api_key, model_config.get("model", "claude-3-sonnet-20240229"), system_prompt, prompt)
            elif model_type == "google":
                return self._call_google_api(api_key, model_config.get("model", "gemini-pro"), system_prompt, prompt)
        except Exception as e:
            self.ai_response_label.setText(f"API error: {str(e)[:50]}. Using local interpretation.")
            return self._interpret_prompt_locally(prompt)

        return self._interpret_prompt_locally(prompt)

    def _call_openai_api(self, api_key, model, system_prompt, user_prompt):
        """Call OpenAI API for prompt interpretation."""
        import json
        try:
            import urllib.request
            import urllib.error

            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.3
            }

            req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    def _call_anthropic_api(self, api_key, model, system_prompt, user_prompt):
        """Call Anthropic API for prompt interpretation."""
        import json
        try:
            import urllib.request

            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            data = {
                "model": model,
                "max_tokens": 200,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            }

            req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                content = result["content"][0]["text"]
                return json.loads(content)
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return None

    def _call_google_api(self, api_key, model, system_prompt, user_prompt):
        """Call Google Gemini API for prompt interpretation."""
        import json
        try:
            import urllib.request

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser request: {user_prompt}"}]}],
                "generationConfig": {"maxOutputTokens": 200, "temperature": 0.3}
            }

            req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                # Extract JSON from response
                import re
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            print(f"Google API error: {e}")
            return None

    def _apply_interpreted_effect(self, effect):
        """Apply the interpreted effect to the canvas."""
        shader_name = effect.get("shader")
        params = effect.get("params", {})

        if not shader_name:
            self.ai_response_label.setText("No shader effect could be determined.")
            return

        if shader_name not in SHADERS:
            # Try to find a close match
            for name in SHADERS.keys():
                if shader_name.lower() in name.lower() or name.lower() in shader_name.lower():
                    shader_name = name
                    break
            else:
                self.ai_response_label.setText(f"Shader '{shader_name}' not found. Try a different description.")
                return

        # Apply the shader
        idx = self.shader_combo.findText(shader_name)
        if idx >= 0:
            self.shader_combo.setCurrentIndex(idx)

        # Apply parameters
        for param_name, value in params.items():
            self.canvas.set_param(param_name, value)
            if param_name in self.param_widgets:
                self.param_widgets[param_name].set_value(value)

        # Update canvas
        self.canvas.update()

        # Show what was applied
        param_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()][:3])
        self.ai_response_label.setText(f"Applied: {shader_name}\n{param_str}{'...' if len(params) > 3 else ''}")
        self.statusBar().showMessage(f"AI Effect Applied: {shader_name}", 3000)

    def _suggest_ai_prompts(self):
        """Show suggestions for effect prompts."""
        suggestions = [
            "outline with thick black lines",
            "dreamy soft glow effect",
            "vintage film with grain",
            "cyberpunk neon glow",
            "oil painting artistic",
            "cartoon cel-shaded look",
            "black and white noir",
            "pixelated retro game",
            "underwater caustics",
            "dramatic cinematic grade"
        ]

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Effect Suggestions")
        dialog.setMinimumWidth(350)
        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel("Click a suggestion to use it:")
        layout.addWidget(label)

        list_widget = QtWidgets.QListWidget()
        list_widget.addItems(suggestions)
        list_widget.itemDoubleClicked.connect(lambda item: self._use_suggestion(item.text(), dialog))
        layout.addWidget(list_widget)

        btn_row = QtWidgets.QHBoxLayout()
        use_btn = QtWidgets.QPushButton("Use Selected")
        use_btn.clicked.connect(lambda: self._use_suggestion(list_widget.currentItem().text() if list_widget.currentItem() else "", dialog))
        btn_row.addWidget(use_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)
        dialog.exec()

    def _use_suggestion(self, suggestion, dialog):
        """Use the selected suggestion."""
        if suggestion:
            self.ai_prompt_input.setPlainText(suggestion)
            dialog.accept()
            self._generate_ai_effect()

    def _apply_quick_ai_preset(self, preset_name):
        """Apply a quick AI preset."""
        if preset_name in QUICK_AI_PROMPTS:
            self.ai_prompt_input.setPlainText(QUICK_AI_PROMPTS[preset_name])
            self._generate_ai_effect()

    def _update_params_ui(self, shader_name):
        # Clear existing
        while self.params_layout.count() > 1:
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.param_widgets.clear()

        shader_def = SHADERS.get(shader_name, {})
        self.description.setText(shader_def.get("description", ""))

        uniforms = shader_def.get("uniforms", {})
        if not uniforms:
            label = QtWidgets.QLabel("No adjustable parameters")
            label.setStyleSheet("color: #666; font-style: italic;")
            self.params_layout.insertWidget(0, label)
            return

        for name, props in uniforms.items():
            widget = ParameterSlider(name, props)
            widget.valueChanged.connect(self._on_param_changed)
            self.param_widgets[name] = widget
            self.params_layout.insertWidget(self.params_layout.count() - 1, widget)

    def _on_param_changed(self, name, value):
        # Save current state for undo before making change
        old_value = self.canvas.params.get(name)
        if old_value is not None and abs(old_value - value) > 0.001:
            self._push_undo_state(name, old_value, value)

        self.canvas.set_param(name, value)

        # Also update the selected layer's params
        if self.layer_panel.get_selected_layer():
            self.layer_panel.update_selected_layer_param(name, value)

    def _push_undo_state(self, name, old_value, new_value):
        """Push a parameter change to the undo stack."""
        state = {
            "shader": self.canvas.current_preset,
            "param": name,
            "old_value": old_value,
            "new_value": new_value
        }
        self.canvas.undo_stack.append(state)

        # Limit undo stack size
        if len(self.canvas.undo_stack) > self.canvas.max_undo:
            self.canvas.undo_stack.pop(0)

        # Clear redo stack on new action
        self.canvas.redo_stack.clear()

        self._update_undo_buttons()

    def _undo(self):
        """Undo the last parameter change."""
        if not self.canvas.undo_stack:
            return

        state = self.canvas.undo_stack.pop()

        # Only undo if we're on the same shader
        if state["shader"] == self.canvas.current_preset:
            param_name = state["param"]
            old_value = state["old_value"]

            # Apply the old value
            self.canvas.set_param(param_name, old_value)
            if param_name in self.param_widgets:
                self.param_widgets[param_name].set_value(old_value)

            # Push to redo stack
            self.canvas.redo_stack.append(state)

        self._update_undo_buttons()

    def _redo(self):
        """Redo the last undone parameter change."""
        if not self.canvas.redo_stack:
            return

        state = self.canvas.redo_stack.pop()

        # Only redo if we're on the same shader
        if state["shader"] == self.canvas.current_preset:
            param_name = state["param"]
            new_value = state["new_value"]

            # Apply the new value
            self.canvas.set_param(param_name, new_value)
            if param_name in self.param_widgets:
                self.param_widgets[param_name].set_value(new_value)

            # Push back to undo stack
            self.canvas.undo_stack.append(state)

        self._update_undo_buttons()

    def _update_undo_buttons(self):
        """Update the enabled state of undo/redo buttons."""
        self.undo_btn.setEnabled(len(self.canvas.undo_stack) > 0)
        self.redo_btn.setEnabled(len(self.canvas.redo_stack) > 0)

    def _reset_params(self):
        shader_def = SHADERS.get(self.canvas.current_preset, {})
        uniforms = shader_def.get("uniforms", {})
        for name, props in uniforms.items():
            default = props['default']
            self.canvas.set_param(name, default)
            if name in self.param_widgets:
                self.param_widgets[name].set_value(default)

        # Clear cache for this shader
        if self.canvas.current_preset in self.current_params_cache:
            del self.current_params_cache[self.canvas.current_preset]

    # --- Bake Pass ---

    def _bake_pass(self):
        """Bake the current shader effect and commit as new source texture."""
        if self.canvas.current_preset == "Original":
            QtWidgets.QMessageBox.information(self, "Bake Pass", "Select a shader other than 'Original' before baking.")
            return
        entry = self.canvas.bake_current_pass()
        if entry is None:
            QtWidgets.QMessageBox.warning(self, "Bake Pass", "Failed to bake pass. Make sure an image is loaded.")
            return
        # Update chain display
        chain_names = [e['shader'] for e in self.canvas._bake_chain]
        self.chain_display_label.setText(" \u2192 ".join(chain_names))
        # Reset shader combo to Original
        idx = self.shader_combo.findText("Original")
        if idx >= 0:
            self.shader_combo.setCurrentIndex(idx)
        self._update_params_ui("Original")

    def _reset_chain(self):
        """Reset to the original image and clear the bake chain."""
        if not self.canvas._bake_chain:
            return
        self.canvas.reset_chain()
        self.chain_display_label.setText("No chain")
        # Reset shader combo to Original
        idx = self.shader_combo.findText("Original")
        if idx >= 0:
            self.shader_combo.setCurrentIndex(idx)
        self._update_params_ui("Original")

    # --- Preset Management ---

    def _refresh_preset_combo(self):
        """Refresh the user preset dropdown."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Select Preset...")
        for name in sorted(USER_PRESETS.keys()):
            base = USER_PRESETS[name].get("base_shader", "?")
            self.preset_combo.addItem(f"{name} ({base})")
        self.preset_combo.blockSignals(False)

    def _save_preset(self):
        """Save current shader parameters as a custom preset."""
        current_shader = self.canvas.current_preset
        current_params = dict(self.canvas.params)

        # Ask for preset name
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Preset",
            f"Enter a name for this {current_shader} preset:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"My {current_shader}"
        )

        if ok and name:
            # Check if exists
            if name in USER_PRESETS:
                reply = QtWidgets.QMessageBox.question(
                    self, "Overwrite?",
                    f"Preset '{name}' already exists. Overwrite?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
                )
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    return

            # Save preset (including layer stack)
            preset_data = {
                "base_shader": current_shader,
                "params": current_params
            }
            layer_stack = self.layer_panel.get_layer_stack_data()
            if layer_stack:
                preset_data["layer_stack"] = layer_stack
            USER_PRESETS[name] = preset_data
            save_user_presets()
            self._refresh_preset_combo()

            QtWidgets.QMessageBox.information(
                self, "Saved",
                f"Preset '{name}' saved successfully!"
            )

    def _delete_preset(self):
        """Delete a user preset."""
        if self.preset_combo.currentIndex() <= 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a preset to delete.")
            return

        preset_text = self.preset_combo.currentText()
        preset_name = preset_text.split(" (")[0]

        reply = QtWidgets.QMessageBox.question(
            self, "Delete Preset?",
            f"Are you sure you want to delete '{preset_name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            if preset_name in USER_PRESETS:
                del USER_PRESETS[preset_name]
                save_user_presets()
                self._refresh_preset_combo()
                QtWidgets.QMessageBox.information(self, "Deleted", f"Preset '{preset_name}' deleted.")

    def _load_user_preset(self, text):
        """Load a user preset from the dropdown."""
        if not text or text == "Select Preset...":
            return

        preset_name = text.split(" (")[0]
        self._apply_user_preset(preset_name)

    def _apply_user_preset(self, preset_name):
        """Apply a user preset."""
        if preset_name not in USER_PRESETS:
            return

        preset = USER_PRESETS[preset_name]
        base_shader = preset.get("base_shader", "Original")
        params = preset.get("params", {})

        # Switch to base shader
        self.shader_combo.blockSignals(True)
        idx = self.shader_combo.findText(base_shader)
        if idx >= 0:
            self.shader_combo.setCurrentIndex(idx)
        self.shader_combo.blockSignals(False)

        self.canvas.set_preset(base_shader)
        self._update_params_ui(base_shader)

        # Apply saved params
        for name, value in params.items():
            self.canvas.set_param(name, value)
            if name in self.param_widgets:
                self.param_widgets[name].set_value(value)

        # Restore layer stack if present
        layer_stack = preset.get("layer_stack")
        if layer_stack:
            self.layer_panel.restore_layer_stack(layer_stack)

        self.description.setText(f"Custom preset based on {base_shader}")

    # --- 3D Mode Methods ---

    def _load_primitive(self, text):
        """Load a primitive 3D shape."""
        if text == "Load Primitive...":
            return

        self.primitive_combo.blockSignals(True)
        self.primitive_combo.setCurrentIndex(0)
        self.primitive_combo.blockSignals(False)

        if self.canvas.load_primitive(text.lower()):
            self.image_info.setText(f"3D Primitive: {text}")
            self.controls_3d.show()
            self.mode_2d_btn.setEnabled(True)

    def load_3d_model(self):
        """Load a 3D model (OBJ, GLTF, GLB)."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open 3D Model", "",
            "3D Models (*.obj *.gltf *.glb);;OBJ Files (*.obj);;GLTF Files (*.gltf *.glb);;All Files (*)"
        )
        if path:
            if self.canvas.load_3d_model(path):
                self.image_info.setText(f"3D Model: {os.path.basename(path)}")
                self.controls_3d.show()
                self.mode_2d_btn.setEnabled(True)

    def _switch_to_2d(self):
        """Switch back to 2D mode."""
        self.canvas.set_2d_mode()
        self.controls_3d.hide()
        self.mode_2d_btn.setEnabled(False)
        self.auto_rotate_cb.setChecked(False)

    def _toggle_auto_rotate(self, enabled):
        """Toggle auto-rotation."""
        self.canvas.set_auto_rotate(enabled)

    def _update_rotation_speed(self, value):
        """Update rotation speed."""
        self.canvas.rotation_speed = value / 10.0

    def _update_zoom(self, value):
        """Update zoom level."""
        self.canvas.zoom = value / 10.0
        self.canvas.update()

    # --- GIF Controls ---
    def _show_gif_controls(self):
        """Show GIF playback controls."""
        if self.canvas.is_gif and self.canvas.gif_frames:
            frame_count = len(self.canvas.gif_frames)
            self.gif_frame_slider.setMaximum(frame_count - 1)
            self.gif_frame_slider.setValue(0)
            self.gif_frame_label.setText(f"1/{frame_count}")
            self.gif_play_btn.setText("⏸" if self.canvas.gif_playing else "▶")
            self.gif_controls.show()

            # Connect to canvas frame updates
            self.canvas.gif_timer.timeout.connect(self._update_gif_slider)

    def _hide_gif_controls(self):
        """Hide GIF playback controls."""
        self.gif_controls.hide()
        try:
            self.canvas.gif_timer.timeout.disconnect(self._update_gif_slider)
        except:
            pass

    def _toggle_gif_playback(self):
        """Toggle GIF animation playback."""
        self.canvas.toggle_gif_playback()
        self.gif_play_btn.setText("⏸" if self.canvas.gif_playing else "▶")

    def _on_gif_frame_changed(self, value):
        """Handle GIF frame slider change."""
        if self.canvas.is_gif:
            # Pause playback when manually scrubbing
            if self.canvas.gif_playing:
                self.canvas.gif_playing = False
                self.canvas.gif_timer.stop()
                self.gif_play_btn.setText("▶")

            self.canvas.set_gif_frame(value)
            frame_count = len(self.canvas.gif_frames)
            self.gif_frame_label.setText(f"{value + 1}/{frame_count}")

    def _update_gif_slider(self):
        """Update the GIF slider position during playback."""
        if self.canvas.is_gif and self.canvas.gif_playing:
            # Block signals to prevent feedback loop
            self.gif_frame_slider.blockSignals(True)
            self.gif_frame_slider.setValue(self.canvas.gif_current_frame)
            self.gif_frame_slider.blockSignals(False)

            frame_count = len(self.canvas.gif_frames)
            self.gif_frame_label.setText(f"{self.canvas.gif_current_frame + 1}/{frame_count}")

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.gif);;GIF Animation (*.gif);;All Files (*)"
        )
        if path:
            self.canvas.set_2d_mode()
            self.controls_3d.hide()
            self.mode_2d_btn.setEnabled(False)
            if self.canvas.load_texture(path):
                self._on_image_loaded(path)

    def export_image(self):
        if not self.canvas.image_path and not self.canvas.mode_3d:
            QtWidgets.QMessageBox.warning(self, "Warning", "No image or 3D model loaded.")
            return

        # Get base name
        if self.canvas.image_path:
            base = os.path.splitext(os.path.basename(self.canvas.image_path))[0]
        else:
            base = "3d_render"
        shader = self.canvas.current_preset.lower().replace(' ', '_')
        default = f"{base}_{shader}.png"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Image", default,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tiff)"
        )
        if path:
            try:
                # Export at original resolution
                image_data = self.canvas.export_to_image()
                if image_data is not None:
                    img = Image.fromarray(image_data)
                    img.save(path)
                    QtWidgets.QMessageBox.information(
                        self, "Export Complete",
                        f"Image exported to:\n{path}"
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Export Failed",
                        "Failed to render image for export."
                    )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Export Failed",
                    f"Error exporting image: {e}"
                )

    def batch_process(self):
        """Batch process multiple images with the current shader settings."""
        # Show batch options dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Batch Processing Options")
        dialog.setMinimumWidth(500)
        layout = QtWidgets.QVBoxLayout(dialog)

        # Output format selection
        format_group = QtWidgets.QGroupBox("Output Format")
        format_layout = QtWidgets.QVBoxLayout(format_group)

        self.batch_individual_radio = QtWidgets.QRadioButton("Individual images (PNG)")
        self.batch_individual_radio.setChecked(True)
        format_layout.addWidget(self.batch_individual_radio)

        self.batch_gif_radio = QtWidgets.QRadioButton("Animated GIF (combine all images)")
        format_layout.addWidget(self.batch_gif_radio)

        layout.addWidget(format_group)

        # GIF options (shown when GIF is selected)
        self.gif_options_group = QtWidgets.QGroupBox("GIF Options")
        gif_options_layout = QtWidgets.QFormLayout(self.gif_options_group)

        self.gif_fps_spin = QtWidgets.QSpinBox()
        self.gif_fps_spin.setRange(1, 60)
        self.gif_fps_spin.setValue(10)
        gif_options_layout.addRow("Frame Rate (FPS):", self.gif_fps_spin)

        self.gif_loop_check = QtWidgets.QCheckBox("Loop forever")
        self.gif_loop_check.setChecked(True)
        gif_options_layout.addRow("", self.gif_loop_check)

        self.gif_options_group.hide()
        layout.addWidget(self.gif_options_group)

        # Connect radio buttons to show/hide GIF options
        self.batch_gif_radio.toggled.connect(lambda checked: self.gif_options_group.setVisible(checked))

        # --- Shader Chain ---
        chain_group = QtWidgets.QGroupBox("Shader Chain (multi-pass)")
        chain_layout = QtWidgets.QVBoxLayout(chain_group)

        chain_info = QtWidgets.QLabel("Add shaders to chain multiple effects. Leave empty to use the current shader.")
        chain_info.setWordWrap(True)
        chain_info.setStyleSheet("color: #aaa; font-size: 11px;")
        chain_layout.addWidget(chain_info)

        # Shader chain list — pre-populate from bake chain if available
        if self.canvas._bake_chain:
            import copy
            self._batch_shader_chain = copy.deepcopy(self.canvas._bake_chain)
        else:
            self._batch_shader_chain = []
        self._batch_chain_list = QtWidgets.QListWidget()
        self._batch_chain_list.setMaximumHeight(120)
        self._batch_chain_list.setStyleSheet("background-color: #2a2a2a; color: #ddd;")
        chain_layout.addWidget(self._batch_chain_list)

        # Add shader row: combo + add button
        add_row = QtWidgets.QHBoxLayout()
        self._batch_chain_combo = QtWidgets.QComboBox()
        shader_names = sorted(SHADERS.keys())
        self._batch_chain_combo.addItems(shader_names)
        self._batch_chain_combo.setCurrentText(self.canvas.current_preset)
        add_row.addWidget(self._batch_chain_combo)

        add_btn = QtWidgets.QPushButton("Add")
        add_btn.setFixedWidth(60)
        add_btn.setStyleSheet("background-color: #2d5a2d;")
        add_btn.clicked.connect(lambda: self._batch_chain_add_shader(dialog))
        add_row.addWidget(add_btn)
        chain_layout.addLayout(add_row)

        # Action buttons row
        action_row = QtWidgets.QHBoxLayout()

        configure_btn = QtWidgets.QPushButton("Configure")
        configure_btn.clicked.connect(lambda: self._batch_chain_configure(dialog))
        action_row.addWidget(configure_btn)

        move_up_btn = QtWidgets.QPushButton("\u2191 Up")
        move_up_btn.clicked.connect(lambda: self._batch_chain_move(-1))
        action_row.addWidget(move_up_btn)

        move_down_btn = QtWidgets.QPushButton("\u2193 Down")
        move_down_btn.clicked.connect(lambda: self._batch_chain_move(1))
        action_row.addWidget(move_down_btn)

        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.setStyleSheet("background-color: #5a2d2d;")
        remove_btn.clicked.connect(self._batch_chain_remove)
        action_row.addWidget(remove_btn)

        chain_layout.addLayout(action_row)
        layout.addWidget(chain_group)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)

        proceed_btn = QtWidgets.QPushButton("Select Images...")
        proceed_btn.clicked.connect(dialog.accept)
        proceed_btn.setStyleSheet("background-color: #4a6fa5;")
        btn_layout.addWidget(proceed_btn)

        layout.addLayout(btn_layout)

        # Refresh list to show any pre-populated entries from bake chain
        if self._batch_shader_chain:
            self._batch_chain_refresh_list()

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        export_as_gif = self.batch_gif_radio.isChecked()
        gif_fps = self.gif_fps_spin.value()
        gif_loop = self.gif_loop_check.isChecked()
        shader_chain = list(self._batch_shader_chain)  # Copy chain before dialog is destroyed

        # Select input files
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Images for Batch Processing", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
        )
        if not files:
            return

        if export_as_gif:
            self._batch_export_gif(files, gif_fps, gif_loop)
        else:
            self._batch_export_individual(files, shader_chain)

    def _batch_chain_add_shader(self, dialog):
        """Add a shader to the batch chain list."""
        shader_name = self._batch_chain_combo.currentText()
        if shader_name not in SHADERS:
            return

        # Initialize with default params
        defaults = {}
        for param_name, param_def in SHADERS[shader_name].get("uniforms", {}).items():
            defaults[param_name] = param_def.get("default", 0.0)

        entry = {'shader': shader_name, 'params': defaults}
        self._batch_shader_chain.append(entry)
        self._batch_chain_refresh_list()

    def _batch_chain_remove(self):
        """Remove the selected shader from the chain."""
        row = self._batch_chain_list.currentRow()
        if 0 <= row < len(self._batch_shader_chain):
            self._batch_shader_chain.pop(row)
            self._batch_chain_refresh_list()

    def _batch_chain_move(self, direction):
        """Move the selected shader up or down in the chain."""
        row = self._batch_chain_list.currentRow()
        new_row = row + direction
        if 0 <= row < len(self._batch_shader_chain) and 0 <= new_row < len(self._batch_shader_chain):
            self._batch_shader_chain[row], self._batch_shader_chain[new_row] = \
                self._batch_shader_chain[new_row], self._batch_shader_chain[row]
            self._batch_chain_refresh_list()
            self._batch_chain_list.setCurrentRow(new_row)

    def _batch_chain_refresh_list(self):
        """Refresh the chain list widget to match the chain data."""
        self._batch_chain_list.clear()
        for i, entry in enumerate(self._batch_shader_chain):
            shader_name = entry['shader']
            # Show a brief summary of non-default params
            param_summary = []
            defaults = {k: v.get("default", 0.0) for k, v in SHADERS.get(shader_name, {}).get("uniforms", {}).items()}
            for k, v in entry['params'].items():
                if abs(v - defaults.get(k, 0.0)) > 0.001:
                    param_summary.append(f"{k}={v:.2f}")
            suffix = f"  ({', '.join(param_summary)})" if param_summary else ""
            self._batch_chain_list.addItem(f"{i+1}. {shader_name}{suffix}")

    def _batch_chain_configure(self, parent_dialog):
        """Open a parameter configuration dialog for the selected chain shader."""
        row = self._batch_chain_list.currentRow()
        if row < 0 or row >= len(self._batch_shader_chain):
            QtWidgets.QMessageBox.information(parent_dialog, "Configure", "Select a shader in the chain first.")
            return

        entry = self._batch_shader_chain[row]
        shader_name = entry['shader']
        shader_def = SHADERS.get(shader_name)
        if not shader_def:
            return

        uniforms = shader_def.get("uniforms", {})
        if not uniforms:
            QtWidgets.QMessageBox.information(parent_dialog, "Configure", f"{shader_name} has no configurable parameters.")
            return

        # Create config dialog
        config_dialog = QtWidgets.QDialog(parent_dialog)
        config_dialog.setWindowTitle(f"Configure: {shader_name}")
        config_dialog.setMinimumWidth(400)
        config_layout = QtWidgets.QVBoxLayout(config_dialog)

        sliders = {}
        for param_name, param_def in uniforms.items():
            param_min = param_def.get("min", 0.0)
            param_max = param_def.get("max", 1.0)
            param_default = param_def.get("default", 0.0)
            param_step = param_def.get("step", 0.01)
            current_val = entry['params'].get(param_name, param_default)

            row_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(param_name.replace('_', ' ').title())
            label.setFixedWidth(140)
            row_layout.addWidget(label)

            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            steps = max(1, int((param_max - param_min) / param_step))
            slider.setRange(0, steps)
            slider_pos = int((current_val - param_min) / (param_max - param_min) * steps)
            slider.setValue(max(0, min(steps, slider_pos)))
            row_layout.addWidget(slider)

            value_label = QtWidgets.QLabel(f"{current_val:.3f}")
            value_label.setFixedWidth(60)
            row_layout.addWidget(value_label)

            # Update value label on slider change
            def make_update(sl, vl, pmin, pmax, st):
                def update(pos):
                    val = pmin + (pos / st) * (pmax - pmin)
                    vl.setText(f"{val:.3f}")
                return update
            slider.valueChanged.connect(make_update(slider, value_label, param_min, param_max, steps))

            sliders[param_name] = (slider, param_min, param_max, steps)
            config_layout.addLayout(row_layout)

        # Reset to defaults button
        def reset_defaults():
            for pname, (sl, pmin, pmax, st) in sliders.items():
                default_val = uniforms[pname].get("default", 0.0)
                slider_pos = int((default_val - pmin) / (pmax - pmin) * st)
                sl.setValue(max(0, min(st, slider_pos)))
        reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(reset_defaults)
        config_layout.addWidget(reset_btn)

        # OK / Cancel
        btn_layout = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(config_dialog.reject)
        btn_layout.addWidget(cancel_btn)
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(config_dialog.accept)
        ok_btn.setStyleSheet("background-color: #4a6fa5;")
        btn_layout.addWidget(ok_btn)
        config_layout.addLayout(btn_layout)

        if config_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Read slider values back into entry params
            for param_name, (slider, param_min, param_max, steps) in sliders.items():
                val = param_min + (slider.value() / steps) * (param_max - param_min)
                entry['params'][param_name] = val
            self._batch_chain_refresh_list()

    def _batch_export_individual(self, files, shader_chain=None):
        """Export batch processed images as individual files.
        If shader_chain is provided and non-empty, applies shaders sequentially
        (multi-pass), feeding each pass's output as the next pass's input.
        """
        import numpy as np

        # Select output directory
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if not output_dir:
            return

        use_chain = shader_chain and len(shader_chain) > 0
        total_passes = len(shader_chain) if use_chain else 1
        total_steps = len(files) * total_passes

        # Create progress dialog
        progress = QtWidgets.QProgressDialog(
            "Processing images...", "Cancel", 0, total_steps, self
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Batch Processing")

        # Build output name suffix
        if use_chain:
            chain_suffix = "_".join(e['shader'].lower().replace(' ', '_') for e in shader_chain)
        else:
            chain_suffix = self.canvas.current_preset.lower().replace(' ', '_')

        processed = 0
        failed = 0
        failed_files = []
        step = 0

        self._batch_processing = True
        try:
            for i, file_path in enumerate(files):
                if progress.wasCanceled():
                    break

                file_base = os.path.basename(file_path)

                try:
                    # Load original image
                    if not self.canvas.load_texture(file_path):
                        print(f"[BATCH] Failed to load: {file_path}")
                        failed += 1
                        failed_files.append(file_base)
                        step += total_passes
                        continue

                    self.canvas.makeCurrent()

                    if use_chain:
                        # Multi-pass shader chain
                        image_data = None
                        w, h = self.canvas.image_size

                        for pass_idx, chain_entry in enumerate(shader_chain):
                            if progress.wasCanceled():
                                break

                            step += 1
                            progress.setValue(step)
                            progress.setLabelText(
                                f"Image {i+1}/{len(files)}: Pass {pass_idx+1}/{total_passes} "
                                f"({chain_entry['shader']})"
                            )
                            QtWidgets.QApplication.processEvents()

                            # Compile shader if needed
                            shader_name = chain_entry['shader']
                            program = self.canvas.layer_programs.get(shader_name)
                            if program is None:
                                program = self.canvas._compile_layer_shader(shader_name)
                            if program is None:
                                program = self.canvas.program
                            if program is None:
                                print(f"[BATCH] No program for shader: {shader_name}")
                                break

                            # Ensure VAO vertex attributes are set up for this program
                            GL.glUseProgram(program)
                            GL.glBindVertexArray(self.canvas.vao)
                            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.canvas.vbo)
                            pos_loc = GL.glGetAttribLocation(program, "in_vert")
                            if pos_loc >= 0:
                                GL.glEnableVertexAttribArray(pos_loc)
                                GL.glVertexAttribPointer(pos_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
                            uv_loc = GL.glGetAttribLocation(program, "in_uv")
                            if uv_loc >= 0:
                                GL.glEnableVertexAttribArray(uv_loc)
                                GL.glVertexAttribPointer(uv_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 16,
                                                         GL.ctypes.c_void_p(8))

                            # Render this pass
                            image_data = self.canvas._export_2d_single(
                                program, chain_entry['params'], w, h
                            )
                            if image_data is None:
                                print(f"[BATCH] Pass {pass_idx+1} returned None for: {file_path}")
                                break

                            # Feed output back as input for next pass
                            if pass_idx < len(shader_chain) - 1:
                                self.canvas._upload_texture(np.ascontiguousarray(np.flipud(image_data)))

                        if image_data is None:
                            failed += 1
                            failed_files.append(file_base)
                            continue
                    else:
                        # Legacy single-shader path
                        step += 1
                        progress.setValue(step)
                        progress.setLabelText(f"Processing: {file_base}")
                        QtWidgets.QApplication.processEvents()

                        self.canvas.update()
                        QtWidgets.QApplication.processEvents()

                        layers = self.layer_panel.get_all_layers()
                        image_data = self.canvas.export_to_image(layers=layers if layers else None)
                        if image_data is None:
                            print(f"[BATCH] export_to_image returned None for: {file_path}")
                            failed += 1
                            failed_files.append(file_base)
                            continue

                    # Save output
                    base = os.path.splitext(file_base)[0]
                    output_path = os.path.join(output_dir, f"{base}_{chain_suffix}.png")
                    img = Image.fromarray(image_data)
                    img.save(output_path)
                    processed += 1
                    print(f"[BATCH] Saved: {output_path}")

                except Exception as e:
                    print(f"[BATCH] Error processing {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
                    failed_files.append(file_base)
        finally:
            self._batch_processing = False

        progress.setValue(total_steps)

        # After batch, reload the last successfully processed file and create base layer
        if processed > 0:
            last_file = files[min(processed, len(files)) - 1]
            self.canvas.load_texture(last_file)
            self._on_image_loaded(last_file)

        detail = ""
        if failed_files:
            detail = f"\n\nFailed files:\n" + "\n".join(failed_files[:10])
            if len(failed_files) > 10:
                detail += f"\n... and {len(failed_files) - 10} more"

        chain_info = ""
        if use_chain:
            chain_info = f"\nShader chain: {' -> '.join(e['shader'] for e in shader_chain)}\n"

        QtWidgets.QMessageBox.information(
            self, "Batch Processing Complete",
            f"Processed: {processed} images\nFailed: {failed} images\n{chain_info}\nOutput: {output_dir}{detail}"
        )

    def _batch_export_gif(self, files, fps, loop_forever):
        """Export batch processed images as an animated GIF."""
        # Select output file
        shader_name = self.canvas.current_preset.lower().replace(' ', '_')
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Animated GIF", f"batch_{shader_name}.gif",
            "GIF Animation (*.gif)"
        )
        if not output_path:
            return

        # Create progress dialog
        progress = QtWidgets.QProgressDialog(
            "Processing images...", "Cancel", 0, len(files), self
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Creating GIF")

        frames = []
        failed = 0

        for i, file_path in enumerate(files):
            if progress.wasCanceled():
                return

            progress.setValue(i)
            progress.setLabelText(f"Processing frame {i+1}/{len(files)}: {os.path.basename(file_path)}")
            QtWidgets.QApplication.processEvents()

            try:
                # Load image
                if not self.canvas.load_texture(file_path):
                    failed += 1
                    continue

                # Render and get frame
                image_data = self.canvas.export_to_image()
                if image_data is None:
                    failed += 1
                    continue

                # Convert to PIL Image and add to frames
                frame = Image.fromarray(image_data)
                frames.append(frame)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                failed += 1

        if not frames:
            QtWidgets.QMessageBox.warning(
                self, "Error",
                "No frames could be processed. GIF was not created."
            )
            return

        progress.setLabelText("Saving GIF...")
        progress.setValue(len(files))
        QtWidgets.QApplication.processEvents()

        try:
            # Calculate frame duration in milliseconds
            frame_duration = int(1000 / fps)

            # Save as GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0 if loop_forever else 1,
                optimize=True
            )

            QtWidgets.QMessageBox.information(
                self, "GIF Export Complete",
                f"Created animated GIF with {len(frames)} frames\n"
                f"Frame rate: {fps} FPS\n"
                f"Duration: {len(frames) / fps:.1f} seconds\n"
                f"Failed: {failed} images\n\n"
                f"Saved to: {output_path}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export Error",
                f"Failed to save GIF: {e}"
            )

    def upscale_image(self):
        """Upscale the current image with AI-enhanced or traditional methods."""
        if not self.canvas.image_path or self.canvas.mode_3d:
            QtWidgets.QMessageBox.warning(
                self, "Warning",
                "Please load an image first (upscaling only works on 2D images)."
            )
            return

        # Show upscale options dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Upscale Image")
        dialog.setMinimumWidth(300)
        layout = QtWidgets.QVBoxLayout(dialog)

        # Scale factor
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(QtWidgets.QLabel("Scale:"))
        scale_combo = QtWidgets.QComboBox()
        scale_combo.addItems(["2x", "3x", "4x"])
        scale_row.addWidget(scale_combo)
        layout.addLayout(scale_row)

        # Method
        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(QtWidgets.QLabel("Method:"))
        method_combo = QtWidgets.QComboBox()
        methods = ["Lanczos (High Quality)", "Bicubic", "Bilinear"]
        if UPSCALING_AVAILABLE:
            methods.insert(0, "AI Enhanced (Recommended)")
        method_combo.addItems(methods)
        method_row.addWidget(method_combo)
        layout.addLayout(method_row)

        # Apply shader checkbox
        apply_shader_cb = QtWidgets.QCheckBox("Apply current shader effect")
        apply_shader_cb.setChecked(True)
        layout.addWidget(apply_shader_cb)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)

        upscale_btn = QtWidgets.QPushButton("Upscale")
        upscale_btn.clicked.connect(dialog.accept)
        upscale_btn.setStyleSheet("background-color: #2d5a2d;")
        btn_row.addWidget(upscale_btn)
        layout.addLayout(btn_row)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        # Get settings
        scale = int(scale_combo.currentText()[0])
        method_text = method_combo.currentText()
        if "AI" in method_text:
            method = "ai"
        elif "Lanczos" in method_text:
            method = "lanczos"
        elif "Bicubic" in method_text:
            method = "bicubic"
        else:
            method = "bilinear"

        apply_shader = apply_shader_cb.isChecked()

        # Get save path
        base = os.path.splitext(os.path.basename(self.canvas.image_path))[0]
        default = f"{base}_{scale}x_upscaled.png"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Upscaled Image", default,
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return

        try:
            # Show progress
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

            if apply_shader:
                # Export with shader applied, then upscale
                image_data = self.canvas.export_to_image()
            else:
                # Load original image
                img = Image.open(self.canvas.image_path).convert("RGBA")
                image_data = np.array(img)

            # Upscale
            upscaled = UPSCALER.upscale(image_data, scale, method)

            # Save
            img = Image.fromarray(upscaled)
            img.save(path)

            QtWidgets.QApplication.restoreOverrideCursor()

            QtWidgets.QMessageBox.information(
                self, "Upscale Complete",
                f"Image upscaled to {upscaled.shape[1]}x{upscaled.shape[0]} pixels.\n\nSaved to: {path}"
            )

        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(
                self, "Upscale Failed",
                f"Error: {e}"
            )


def main():
    print("Starting Shader Studio v2...")

    app = QtWidgets.QApplication(sys.argv)

    # Set default OpenGL format
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)

    window = ShaderStudio()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
