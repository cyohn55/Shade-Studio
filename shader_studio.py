"""
Shader Studio - A real-time shader visualization tool with multiple artistic styles.

Features:
- Multiple shader presets: Pixelation, Grease Pencil, Toon, Comic Book, and more
- Real-time parameter adjustment with grouped inspector controls
- Support for PNG images and 3D model rendering
- Export functionality for processed images

Author: Shader Studio
"""

import sys
import numpy as np
import moderngl
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PIL import Image
import os

# --- SHADER LIBRARY ---
# Each shader has: name, category, uniforms (with min, max, default, step), and fragment shader code

SHADERS = {
    # ==================== BASIC ====================
    "Original": {
        "category": "Basic",
        "description": "Original image without effects",
        "uniforms": {},
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                f_color = texture(u_texture, v_uv);
            }
        """
    },

    # ==================== PIXELATION ====================
    "Pixelation": {
        "category": "Stylized",
        "description": "Retro pixel art effect with adjustable resolution",
        "uniforms": {
            "pixel_size": {"min": 1.0, "max": 64.0, "default": 8.0, "step": 1.0},
            "color_depth": {"min": 2.0, "max": 256.0, "default": 32.0, "step": 1.0},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float pixel_size;
            uniform float color_depth;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel_uv = floor(v_uv * tex_size / pixel_size) * pixel_size / tex_size;
                vec4 color = texture(u_texture, pixel_uv);

                // Reduce color depth
                float levels = color_depth;
                color.rgb = floor(color.rgb * levels + 0.5) / levels;

                f_color = color;
            }
        """
    },

    "Pixel Art Dither": {
        "category": "Stylized",
        "description": "Pixelation with ordered dithering for retro game look",
        "uniforms": {
            "pixel_size": {"min": 2.0, "max": 32.0, "default": 6.0, "step": 1.0},
            "color_levels": {"min": 2.0, "max": 16.0, "default": 4.0, "step": 1.0},
            "dither_strength": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float pixel_size;
            uniform float color_levels;
            uniform float dither_strength;
            in vec2 v_uv;
            out vec4 f_color;

            // Bayer 4x4 dither matrix
            float bayer4x4(vec2 pos) {
                int x = int(mod(pos.x, 4.0));
                int y = int(mod(pos.y, 4.0));
                int index = x + y * 4;
                float matrix[16] = float[16](
                    0.0/16.0, 8.0/16.0, 2.0/16.0, 10.0/16.0,
                    12.0/16.0, 4.0/16.0, 14.0/16.0, 6.0/16.0,
                    3.0/16.0, 11.0/16.0, 1.0/16.0, 9.0/16.0,
                    15.0/16.0, 7.0/16.0, 13.0/16.0, 5.0/16.0
                );
                return matrix[index] - 0.5;
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel_coord = floor(v_uv * tex_size / pixel_size);
                vec2 pixel_uv = pixel_coord * pixel_size / tex_size;

                vec4 color = texture(u_texture, pixel_uv);

                // Apply dithering
                float dither = bayer4x4(pixel_coord) * dither_strength;
                color.rgb += dither / color_levels;

                // Quantize colors
                color.rgb = floor(color.rgb * color_levels + 0.5) / color_levels;

                f_color = color;
            }
        """
    },

    # ==================== GREASE PENCIL ====================
    "Grease Pencil": {
        "category": "Sketch",
        "description": "Hand-drawn pencil sketch effect like Blender's Grease Pencil",
        "uniforms": {
            "line_thickness": {"min": 0.5, "max": 5.0, "default": 1.5, "step": 0.1},
            "line_darkness": {"min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1},
            "fill_opacity": {"min": 0.0, "max": 1.0, "default": 0.8, "step": 0.05},
            "edge_threshold": {"min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
            "noise_amount": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float line_thickness;
            uniform float line_darkness;
            uniform float fill_opacity;
            uniform float edge_threshold;
            uniform float noise_amount;
            in vec2 v_uv;
            out vec4 f_color;

            // Simple hash for noise
            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = line_thickness / tex_size;

                vec4 base_color = texture(u_texture, v_uv);

                // Sobel edge detection
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

                // Add noise to simulate pencil texture
                float noise = hash(v_uv * tex_size) * noise_amount;
                edge += noise * 0.1;

                // Create pencil stroke effect
                float stroke = smoothstep(edge_threshold, edge_threshold + 0.1, edge);
                stroke *= line_darkness;

                // Combine fill color with strokes
                vec3 fill = base_color.rgb * fill_opacity + vec3(1.0 - fill_opacity);
                vec3 final_color = mix(fill, vec3(0.1), stroke);

                // Add paper texture
                float paper = 0.95 + hash(v_uv * tex_size * 0.5) * 0.05;
                final_color *= paper;

                f_color = vec4(final_color, base_color.a);
            }
        """
    },

    "Pencil Sketch": {
        "category": "Sketch",
        "description": "Classic pencil drawing with cross-hatching",
        "uniforms": {
            "darkness": {"min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1},
            "hatch_density": {"min": 10.0, "max": 100.0, "default": 40.0, "step": 5.0},
            "hatch_angle": {"min": 0.0, "max": 3.14159, "default": 0.785, "step": 0.1},
            "contrast": {"min": 0.5, "max": 3.0, "default": 1.5, "step": 0.1},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float darkness;
            uniform float hatch_density;
            uniform float hatch_angle;
            uniform float contrast;
            in vec2 v_uv;
            out vec4 f_color;

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            float hatch(vec2 uv, float angle, float density) {
                float c = cos(angle);
                float s = sin(angle);
                vec2 rotated = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
                return abs(sin(rotated.x * density));
            }

            void main() {
                vec4 color = texture(u_texture, v_uv);
                float lum = luminance(color.rgb);

                // Apply contrast
                lum = pow(lum, 1.0 / contrast);

                // Create hatching layers based on darkness
                float h1 = hatch(v_uv, hatch_angle, hatch_density);
                float h2 = hatch(v_uv, hatch_angle + 1.57, hatch_density * 0.8);
                float h3 = hatch(v_uv, hatch_angle + 0.785, hatch_density * 1.2);

                float dark = 1.0 - lum;
                float stroke = 0.0;

                if(dark > 0.2) stroke = max(stroke, h1 * smoothstep(0.2, 0.4, dark));
                if(dark > 0.4) stroke = max(stroke, h2 * smoothstep(0.4, 0.6, dark));
                if(dark > 0.6) stroke = max(stroke, h3 * smoothstep(0.6, 0.8, dark));

                stroke *= darkness;

                vec3 paper = vec3(0.95, 0.93, 0.88);
                vec3 pencil = vec3(0.15, 0.12, 0.1);
                vec3 final_color = mix(paper, pencil, stroke);

                f_color = vec4(final_color, color.a);
            }
        """
    },

    # ==================== TOON / CEL SHADING ====================
    "Toon Shader": {
        "category": "Toon",
        "description": "Classic cel-shaded cartoon look with color banding",
        "uniforms": {
            "color_bands": {"min": 2.0, "max": 10.0, "default": 4.0, "step": 1.0},
            "edge_thickness": {"min": 0.5, "max": 5.0, "default": 1.5, "step": 0.1},
            "edge_threshold": {"min": 0.05, "max": 0.5, "default": 0.15, "step": 0.01},
            "saturation_boost": {"min": 0.5, "max": 2.0, "default": 1.2, "step": 0.1},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float color_bands;
            uniform float edge_thickness;
            uniform float edge_threshold;
            uniform float saturation_boost;
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

                // Quantize colors
                vec3 hsv = rgb2hsv(base.rgb);
                hsv.y *= saturation_boost;
                hsv.z = floor(hsv.z * color_bands + 0.5) / color_bands;
                vec3 quantized = hsv2rgb(hsv);

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

                vec3 final_color = (edge > edge_threshold) ? vec3(0.0) : quantized;

                f_color = vec4(final_color, base.a);
            }
        """
    },

    "Anime Shader": {
        "category": "Toon",
        "description": "Anime-style rendering with smooth gradients and highlights",
        "uniforms": {
            "shadow_threshold": {"min": 0.1, "max": 0.8, "default": 0.4, "step": 0.05},
            "shadow_softness": {"min": 0.01, "max": 0.3, "default": 0.1, "step": 0.01},
            "highlight_threshold": {"min": 0.5, "max": 0.95, "default": 0.8, "step": 0.05},
            "rim_light": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
            "outline_thickness": {"min": 0.0, "max": 3.0, "default": 1.0, "step": 0.1},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float shadow_threshold;
            uniform float shadow_softness;
            uniform float highlight_threshold;
            uniform float rim_light;
            uniform float outline_thickness;
            in vec2 v_uv;
            out vec4 f_color;

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = outline_thickness / tex_size;

                vec4 base = texture(u_texture, v_uv);
                float lum = luminance(base.rgb);

                // Create anime-style lighting bands
                float shadow = smoothstep(shadow_threshold - shadow_softness, shadow_threshold + shadow_softness, lum);
                float highlight = smoothstep(highlight_threshold - 0.05, highlight_threshold + 0.05, lum);

                // Apply shadow tint
                vec3 shadow_color = base.rgb * 0.6;
                vec3 mid_color = base.rgb;
                vec3 highlight_color = base.rgb + vec3(0.15);

                vec3 color = mix(shadow_color, mid_color, shadow);
                color = mix(color, highlight_color, highlight);

                // Add rim light effect
                float edge = 0.0;
                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        if(x == 0 && y == 0) continue;
                        float a = texture(u_texture, v_uv + vec2(x, y) * pixel * 2.0).a;
                        edge += (base.a - a);
                    }
                }
                edge = clamp(edge, 0.0, 1.0) * rim_light;
                color += vec3(edge);

                // Outline detection
                float outline = 0.0;
                if(outline_thickness > 0.0) {
                    float samples[9];
                    int idx = 0;
                    for(int y = -1; y <= 1; y++) {
                        for(int x = -1; x <= 1; x++) {
                            samples[idx++] = luminance(texture(u_texture, v_uv + vec2(x, y) * pixel).rgb);
                        }
                    }
                    float gx = samples[2] + 2.0*samples[5] + samples[8] - samples[0] - 2.0*samples[3] - samples[6];
                    float gy = samples[0] + 2.0*samples[1] + samples[2] - samples[6] - 2.0*samples[7] - samples[8];
                    outline = sqrt(gx*gx + gy*gy);
                }

                color = (outline > 0.2) ? vec3(0.05) : color;

                f_color = vec4(color, base.a);
            }
        """
    },

    # ==================== COMIC BOOK ====================
    "Comic Book": {
        "category": "Comic",
        "description": "Classic comic book style with halftone dots and bold outlines",
        "uniforms": {
            "dot_size": {"min": 2.0, "max": 20.0, "default": 8.0, "step": 1.0},
            "dot_angle": {"min": 0.0, "max": 1.57, "default": 0.785, "step": 0.1},
            "outline_thickness": {"min": 1.0, "max": 5.0, "default": 2.0, "step": 0.5},
            "color_strength": {"min": 0.0, "max": 1.0, "default": 0.7, "step": 0.05},
            "ink_darkness": {"min": 0.0, "max": 1.0, "default": 0.9, "step": 0.05},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float dot_size;
            uniform float dot_angle;
            uniform float outline_thickness;
            uniform float color_strength;
            uniform float ink_darkness;
            in vec2 v_uv;
            out vec4 f_color;

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            float halftone(vec2 uv, float size, float angle, float darkness) {
                float c = cos(angle);
                float s = sin(angle);
                vec2 rotated = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
                vec2 cell = fract(rotated * size) - 0.5;
                float dist = length(cell);
                float radius = (1.0 - darkness) * 0.5;
                return smoothstep(radius - 0.05, radius + 0.05, dist);
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = outline_thickness / tex_size;

                vec4 base = texture(u_texture, v_uv);
                float lum = luminance(base.rgb);

                // Create halftone pattern
                float halftone_val = halftone(v_uv * tex_size, 1.0 / dot_size, dot_angle, lum);

                // Quantize base color for comic look
                vec3 comic_color = floor(base.rgb * 4.0 + 0.5) / 4.0;
                comic_color = mix(vec3(1.0), comic_color, color_strength);

                // Apply halftone
                vec3 color = mix(vec3(0.0), comic_color, halftone_val);

                // Edge detection for outlines
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

                // Apply bold ink outlines
                color = mix(color, vec3(0.0), smoothstep(0.1, 0.3, edge) * ink_darkness);

                f_color = vec4(color, base.a);
            }
        """
    },

    "Pop Art": {
        "category": "Comic",
        "description": "Bold pop art style with limited color palette",
        "uniforms": {
            "color_count": {"min": 2.0, "max": 8.0, "default": 4.0, "step": 1.0},
            "saturation": {"min": 0.5, "max": 3.0, "default": 2.0, "step": 0.1},
            "contrast": {"min": 0.5, "max": 3.0, "default": 1.5, "step": 0.1},
            "dot_scale": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float color_count;
            uniform float saturation;
            uniform float contrast;
            uniform float dot_scale;
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
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec4 base = texture(u_texture, v_uv);

                // Boost saturation and contrast
                vec3 hsv = rgb2hsv(base.rgb);
                hsv.y = min(hsv.y * saturation, 1.0);
                hsv.z = pow(hsv.z, 1.0 / contrast);

                // Quantize hue and value
                hsv.x = floor(hsv.x * 6.0 + 0.5) / 6.0;
                hsv.z = floor(hsv.z * color_count + 0.5) / color_count;

                vec3 color = hsv2rgb(hsv);

                // Add halftone dots in shadow areas
                if(dot_scale > 0.0) {
                    float lum = hsv.z;
                    vec2 cell = fract(v_uv * tex_size / 6.0) - 0.5;
                    float dot = length(cell);
                    float threshold = (1.0 - lum) * 0.4 * dot_scale;
                    if(dot < threshold) {
                        color *= 0.7;
                    }
                }

                f_color = vec4(color, base.a);
            }
        """
    },

    # ==================== ARTISTIC ====================
    "Oil Painting": {
        "category": "Artistic",
        "description": "Simulated oil painting with brush stroke texture",
        "uniforms": {
            "brush_size": {"min": 1.0, "max": 10.0, "default": 4.0, "step": 0.5},
            "color_intensity": {"min": 0.5, "max": 2.0, "default": 1.2, "step": 0.1},
            "stroke_strength": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float brush_size;
            uniform float color_intensity;
            uniform float stroke_strength;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = brush_size / tex_size;

                // Kuwahara-like filter for oil paint effect
                vec3 mean[4];
                float var[4];

                for(int k = 0; k < 4; k++) {
                    mean[k] = vec3(0.0);
                    var[k] = 0.0;
                }

                // Sample 4 quadrants
                vec2 offsets[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));

                for(int k = 0; k < 4; k++) {
                    vec3 sum = vec3(0.0);
                    vec3 sum2 = vec3(0.0);
                    float count = 0.0;

                    for(int j = 0; j < 3; j++) {
                        for(int i = 0; i < 3; i++) {
                            vec2 off = (offsets[k] * vec2(i, j)) * pixel;
                            vec3 c = texture(u_texture, v_uv + off).rgb;
                            sum += c;
                            sum2 += c * c;
                            count += 1.0;
                        }
                    }

                    mean[k] = sum / count;
                    vec3 variance = (sum2 / count) - (mean[k] * mean[k]);
                    var[k] = variance.r + variance.g + variance.b;
                }

                // Choose region with minimum variance
                float minVar = var[0];
                vec3 color = mean[0];
                for(int k = 1; k < 4; k++) {
                    if(var[k] < minVar) {
                        minVar = var[k];
                        color = mean[k];
                    }
                }

                // Boost color intensity
                color = pow(color, vec3(1.0 / color_intensity));

                // Add canvas texture
                float canvas = 0.95 + hash(v_uv * tex_size * 0.3) * 0.1 * stroke_strength;
                color *= canvas;

                vec4 base = texture(u_texture, v_uv);
                f_color = vec4(color, base.a);
            }
        """
    },

    "Watercolor": {
        "category": "Artistic",
        "description": "Soft watercolor painting effect with bleeding edges",
        "uniforms": {
            "blur_amount": {"min": 1.0, "max": 10.0, "default": 3.0, "step": 0.5},
            "edge_darkening": {"min": 0.0, "max": 1.0, "default": 0.4, "step": 0.05},
            "color_bleed": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05},
            "paper_texture": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float blur_amount;
            uniform float edge_darkening;
            uniform float color_bleed;
            uniform float paper_texture;
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

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = blur_amount / tex_size;

                // Wobbly sampling for watercolor bleeding
                vec2 wobble = vec2(
                    noise(v_uv * tex_size * 0.1) - 0.5,
                    noise(v_uv * tex_size * 0.1 + 100.0) - 0.5
                ) * color_bleed * pixel * 5.0;

                // Gaussian-like blur
                vec3 color = vec3(0.0);
                float total = 0.0;

                for(int y = -2; y <= 2; y++) {
                    for(int x = -2; x <= 2; x++) {
                        float weight = exp(-(x*x + y*y) / 4.0);
                        vec2 offset = vec2(x, y) * pixel + wobble;
                        color += texture(u_texture, v_uv + offset).rgb * weight;
                        total += weight;
                    }
                }
                color /= total;

                // Edge detection for darkening
                vec4 base = texture(u_texture, v_uv);
                float edge = 0.0;
                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        if(x == 0 && y == 0) continue;
                        vec3 c = texture(u_texture, v_uv + vec2(x, y) * pixel).rgb;
                        edge += length(base.rgb - c);
                    }
                }
                edge = clamp(edge * 0.5, 0.0, 1.0);

                // Darken edges like watercolor pooling
                color = mix(color, color * 0.7, edge * edge_darkening);

                // Paper texture
                float paper = 0.9 + noise(v_uv * tex_size * 0.5) * 0.2 * paper_texture;
                color *= paper;

                // Lighten overall for watercolor feel
                color = mix(color, vec3(1.0), 0.1);

                f_color = vec4(color, base.a);
            }
        """
    },

    # ==================== EFFECTS ====================
    "Sobel Edge": {
        "category": "Effects",
        "description": "Edge detection using Sobel operator",
        "uniforms": {
            "edge_intensity": {"min": 0.5, "max": 5.0, "default": 2.0, "step": 0.1},
            "threshold": {"min": 0.0, "max": 0.5, "default": 0.1, "step": 0.01},
            "invert": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
            "color_edges": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float edge_intensity;
            uniform float threshold;
            uniform float invert;
            uniform float color_edges;
            in vec2 v_uv;
            out vec4 f_color;

            float luminance(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));
                vec2 pixel = 1.0 / tex_size;

                float samples[9];
                vec3 colors[9];
                int idx = 0;

                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        vec4 c = texture(u_texture, v_uv + vec2(x, y) * pixel);
                        colors[idx] = c.rgb;
                        samples[idx++] = luminance(c.rgb);
                    }
                }

                float gx = samples[2] + 2.0*samples[5] + samples[8] - samples[0] - 2.0*samples[3] - samples[6];
                float gy = samples[0] + 2.0*samples[1] + samples[2] - samples[6] - 2.0*samples[7] - samples[8];
                float edge = sqrt(gx*gx + gy*gy) * edge_intensity;

                edge = smoothstep(threshold, threshold + 0.1, edge);

                if(invert > 0.5) edge = 1.0 - edge;

                vec3 color;
                if(color_edges > 0.5) {
                    color = colors[4] * edge;
                } else {
                    color = vec3(edge);
                }

                vec4 base = texture(u_texture, v_uv);
                f_color = vec4(color, base.a);
            }
        """
    },

    "Chromatic Aberration": {
        "category": "Effects",
        "description": "RGB channel separation for lens distortion effect",
        "uniforms": {
            "strength": {"min": 0.0, "max": 0.02, "default": 0.005, "step": 0.001},
            "radial": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 1.0},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float strength;
            uniform float radial;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec2 center = vec2(0.5);
                vec2 dir;

                if(radial > 0.5) {
                    dir = normalize(v_uv - center) * length(v_uv - center);
                } else {
                    dir = vec2(1.0, 0.0);
                }

                vec2 offset = dir * strength;

                float r = texture(u_texture, v_uv + offset).r;
                float g = texture(u_texture, v_uv).g;
                float b = texture(u_texture, v_uv - offset).b;
                float a = texture(u_texture, v_uv).a;

                f_color = vec4(r, g, b, a);
            }
        """
    },

    "Vignette": {
        "category": "Effects",
        "description": "Darkened edges with customizable falloff",
        "uniforms": {
            "intensity": {"min": 0.0, "max": 2.0, "default": 0.8, "step": 0.05},
            "radius": {"min": 0.1, "max": 1.5, "default": 0.7, "step": 0.05},
            "softness": {"min": 0.1, "max": 1.0, "default": 0.4, "step": 0.05},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float intensity;
            uniform float radius;
            uniform float softness;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec4 color = texture(u_texture, v_uv);

                vec2 center = vec2(0.5);
                float dist = length(v_uv - center);
                float vignette = smoothstep(radius, radius - softness, dist);

                color.rgb *= mix(1.0 - intensity, 1.0, vignette);

                f_color = color;
            }
        """
    },

    "Glitch": {
        "category": "Effects",
        "description": "Digital glitch distortion effect",
        "uniforms": {
            "block_size": {"min": 5.0, "max": 50.0, "default": 20.0, "step": 1.0},
            "intensity": {"min": 0.0, "max": 0.1, "default": 0.03, "step": 0.005},
            "color_shift": {"min": 0.0, "max": 0.05, "default": 0.01, "step": 0.002},
            "seed": {"min": 0.0, "max": 100.0, "default": 42.0, "step": 1.0},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float block_size;
            uniform float intensity;
            uniform float color_shift;
            uniform float seed;
            in vec2 v_uv;
            out vec4 f_color;

            float hash(vec2 p) {
                return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec2 tex_size = vec2(textureSize(u_texture, 0));

                // Block-based distortion
                vec2 block = floor(v_uv * tex_size / block_size);
                float glitch_strength = step(0.95, hash(block));

                vec2 uv = v_uv;
                if(glitch_strength > 0.0) {
                    float offset = (hash(block + 0.1) - 0.5) * intensity;
                    uv.x += offset;
                }

                // Color channel separation
                float r = texture(u_texture, uv + vec2(color_shift * glitch_strength, 0.0)).r;
                float g = texture(u_texture, uv).g;
                float b = texture(u_texture, uv - vec2(color_shift * glitch_strength, 0.0)).b;
                float a = texture(u_texture, v_uv).a;

                f_color = vec4(r, g, b, a);
            }
        """
    },

    # ==================== COLOR GRADING ====================
    "Color Grade": {
        "category": "Color",
        "description": "Professional color grading controls",
        "uniforms": {
            "brightness": {"min": -0.5, "max": 0.5, "default": 0.0, "step": 0.02},
            "contrast": {"min": 0.5, "max": 2.0, "default": 1.0, "step": 0.05},
            "saturation": {"min": 0.0, "max": 2.0, "default": 1.0, "step": 0.05},
            "temperature": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "tint": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.05},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float brightness;
            uniform float contrast;
            uniform float saturation;
            uniform float temperature;
            uniform float tint;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec4 color = texture(u_texture, v_uv);

                // Brightness
                color.rgb += brightness;

                // Contrast
                color.rgb = (color.rgb - 0.5) * contrast + 0.5;

                // Saturation
                float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                color.rgb = mix(vec3(gray), color.rgb, saturation);

                // Temperature (warm/cool)
                color.r += temperature * 0.1;
                color.b -= temperature * 0.1;

                // Tint (green/magenta)
                color.g += tint * 0.1;

                f_color = vec4(clamp(color.rgb, 0.0, 1.0), color.a);
            }
        """
    },

    "Sepia": {
        "category": "Color",
        "description": "Vintage sepia tone effect",
        "uniforms": {
            "intensity": {"min": 0.0, "max": 1.0, "default": 0.8, "step": 0.05},
            "brightness": {"min": 0.8, "max": 1.2, "default": 1.0, "step": 0.02},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float intensity;
            uniform float brightness;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec4 color = texture(u_texture, v_uv);

                vec3 sepia;
                sepia.r = dot(color.rgb, vec3(0.393, 0.769, 0.189));
                sepia.g = dot(color.rgb, vec3(0.349, 0.686, 0.168));
                sepia.b = dot(color.rgb, vec3(0.272, 0.534, 0.131));

                vec3 result = mix(color.rgb, sepia, intensity) * brightness;

                f_color = vec4(clamp(result, 0.0, 1.0), color.a);
            }
        """
    },

    "Duotone": {
        "category": "Color",
        "description": "Two-color tone mapping effect",
        "uniforms": {
            "shadow_r": {"min": 0.0, "max": 1.0, "default": 0.1, "step": 0.05},
            "shadow_g": {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
            "shadow_b": {"min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05},
            "highlight_r": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.05},
            "highlight_g": {"min": 0.0, "max": 1.0, "default": 0.8, "step": 0.05},
            "highlight_b": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.05},
        },
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float shadow_r, shadow_g, shadow_b;
            uniform float highlight_r, highlight_g, highlight_b;
            in vec2 v_uv;
            out vec4 f_color;

            void main() {
                vec4 color = texture(u_texture, v_uv);
                float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));

                vec3 shadow = vec3(shadow_r, shadow_g, shadow_b);
                vec3 highlight = vec3(highlight_r, highlight_g, highlight_b);

                vec3 result = mix(shadow, highlight, lum);

                f_color = vec4(result, color.a);
            }
        """
    },
}

# Vertex shader (shared by all effects)
VERTEX_SHADER = """
    #version 330
    in vec2 in_vert;
    in vec2 in_uv;
    out vec2 v_uv;
    void main() {
        gl_Position = vec4(in_vert, 0.0, 1.0);
        v_uv = in_uv;
    }
"""


class ShaderCanvas(QOpenGLWidget):
    """OpenGL widget for rendering shaders in real-time."""

    def __init__(self):
        super().__init__()
        self.ctx = None
        self.texture = None
        self.vbo = None
        self.vao = None
        self.prog = None
        self.current_preset = "Original"
        self.params = {}
        self.image_path = None
        self.image_size = (0, 0)
        self.setMinimumSize(400, 400)

        # Request an OpenGL format with proper settings
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
        self.setFormat(fmt)

    def initializeGL(self):
        try:
            self.ctx = moderngl.create_context()
            print(f"OpenGL Version: {self.ctx.info['GL_VERSION']}")
        except Exception as e:
            print(f"Failed to create ModernGL context: {e}")
            return

        # Quad vertices: [x, y, u, v]
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)

        # Create a placeholder texture with a visible pattern
        placeholder = np.zeros((256, 256, 4), dtype='uint8')
        # Create a checkerboard pattern so we can see if rendering works
        for y in range(256):
            for x in range(256):
                if (x // 32 + y // 32) % 2 == 0:
                    placeholder[y, x] = [60, 60, 60, 255]
                else:
                    placeholder[y, x] = [40, 40, 40, 255]
        self.texture = self.ctx.texture((256, 256), 4, placeholder.tobytes())
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self._init_params()
        self.update_shader()

    def _init_params(self):
        """Initialize parameters from shader definition."""
        shader_def = SHADERS.get(self.current_preset, {})
        uniforms = shader_def.get("uniforms", {})
        self.params = {}
        for name, props in uniforms.items():
            self.params[name] = props.get("default", 0.0)

    def update_shader(self):
        """Compile and update the current shader program."""
        if not self.ctx:
            print("Cannot update shader: context not initialized")
            return

        self.makeCurrent()

        try:
            if self.prog:
                self.prog.release()

            frag_shader = SHADERS[self.current_preset]["frag"]
            self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=frag_shader)
            print(f"Compiled shader: {self.current_preset}")

            if self.vao:
                self.vao.release()
            self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', 'in_uv')
        except Exception as e:
            print(f"Error compiling shader '{self.current_preset}': {e}")
            import traceback
            traceback.print_exc()

    def load_texture(self, path):
        """Load an image file as texture."""
        if not self.ctx:
            print("Context not initialized")
            return False

        try:
            self.makeCurrent()
            img = Image.open(path)
            img = img.convert("RGBA")
            self.image_size = img.size
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

            if self.texture:
                self.texture.release()

            # Create texture from image data
            self.texture = self.ctx.texture(img.size, 4, img.tobytes())
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.image_path = path
            print(f"Loaded texture: {path} ({img.size[0]}x{img.size[1]})")
            self.update()
            return True
        except Exception as e:
            print(f"Error loading texture: {e}")
            import traceback
            traceback.print_exc()
            return False

    def set_preset(self, name):
        """Switch to a different shader preset."""
        if name not in SHADERS:
            return
        self.current_preset = name
        self._init_params()
        self.update_shader()
        self.update()

    def set_param(self, name, value):
        """Update a shader parameter."""
        if name in self.params:
            self.params[name] = value
            self.update()

    def paintGL(self):
        if not self.ctx:
            return

        # Get the actual widget size for viewport
        width = self.width()
        height = self.height()

        # Handle high DPI displays
        pixel_ratio = self.devicePixelRatio()
        width = int(width * pixel_ratio)
        height = int(height * pixel_ratio)

        # Set viewport to match widget size
        self.ctx.viewport = (0, 0, width, height)

        # Clear with dark background
        self.ctx.clear(0.12, 0.12, 0.12, 1.0)

        if not self.texture or not self.prog or not self.vao:
            return

        # Bind texture
        self.texture.use(location=0)

        # Set uniforms
        try:
            if 'u_texture' in self.prog:
                self.prog['u_texture'].value = 0
            for name, val in self.params.items():
                if name in self.prog:
                    self.prog[name].value = val
        except Exception as e:
            print(f"Error setting uniforms: {e}")

        # Render the quad
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def resizeGL(self, width, height):
        """Handle widget resize."""
        if self.ctx:
            pixel_ratio = self.devicePixelRatio()
            self.ctx.viewport = (0, 0, int(width * pixel_ratio), int(height * pixel_ratio))

    def export_image(self, path):
        """Export the current rendered image."""
        if not self.ctx or not self.texture:
            return False

        try:
            self.makeCurrent()

            # Create framebuffer at texture size
            width, height = self.texture.size
            fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 4)]
            )

            fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self.texture.use(location=0)

            if self.prog:
                if 'u_texture' in self.prog:
                    self.prog['u_texture'].value = 0
                for name, val in self.params.items():
                    if name in self.prog:
                        self.prog[name].value = val

            if self.vao:
                self.vao.render(moderngl.TRIANGLE_STRIP)

            # Read pixels and save
            data = fbo.read(components=4)
            img = Image.frombytes('RGBA', (width, height), data)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(path)

            fbo.release()
            self.ctx.screen.use()

            return True
        except Exception as e:
            print(f"Error exporting image: {e}")
            return False


class ParameterGroup(QtWidgets.QGroupBox):
    """A collapsible group of parameter sliders."""

    paramChanged = QtCore.pyqtSignal(str, float)

    def __init__(self, title, params_def, current_values):
        super().__init__(title)
        self.setCheckable(False)
        self.params_def = params_def
        self.sliders = {}
        self.labels = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        for name, props in params_def.items():
            # Parameter label with value
            label = QtWidgets.QLabel(f"{self._format_name(name)}: {current_values.get(name, props['default']):.3f}")
            label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
            self.labels[name] = label
            layout.addWidget(label)

            # Slider
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)

            # Convert value to slider position
            val = current_values.get(name, props['default'])
            pos = self._value_to_slider(val, props)
            slider.setValue(pos)

            slider.valueChanged.connect(lambda v, n=name, p=props: self._on_slider_changed(n, v, p))
            self.sliders[name] = slider
            layout.addWidget(slider)

    def _format_name(self, name):
        """Convert parameter name to display format."""
        return name.replace('_', ' ').title()

    def _value_to_slider(self, value, props):
        """Convert actual value to slider position (0-1000)."""
        min_val = props['min']
        max_val = props['max']
        return int((value - min_val) / (max_val - min_val) * 1000)

    def _slider_to_value(self, pos, props):
        """Convert slider position to actual value."""
        min_val = props['min']
        max_val = props['max']
        step = props.get('step', 0.01)
        value = min_val + (pos / 1000) * (max_val - min_val)
        # Round to step
        value = round(value / step) * step
        return value

    def _on_slider_changed(self, name, pos, props):
        value = self._slider_to_value(pos, props)
        self.labels[name].setText(f"{self._format_name(name)}: {value:.3f}")
        self.paramChanged.emit(name, value)

    def set_values(self, values):
        """Update all slider values."""
        for name, value in values.items():
            if name in self.sliders and name in self.params_def:
                props = self.params_def[name]
                pos = self._value_to_slider(value, props)
                self.sliders[name].blockSignals(True)
                self.sliders[name].setValue(pos)
                self.sliders[name].blockSignals(False)
                self.labels[name].setText(f"{self._format_name(name)}: {value:.3f}")


class ShaderInspector(QtWidgets.QWidget):
    """Inspector panel with shader selection and parameter controls."""

    shaderChanged = QtCore.pyqtSignal(str)
    paramChanged = QtCore.pyqtSignal(str, float)
    exportRequested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QtWidgets.QLabel("Shader Inspector")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff; padding: 5px 0;")
        layout.addWidget(title)

        # Image info
        self.image_info = QtWidgets.QLabel("No image loaded")
        self.image_info.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.image_info)

        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #444444;")
        layout.addWidget(line)

        # Category filter
        cat_layout = QtWidgets.QHBoxLayout()
        cat_label = QtWidgets.QLabel("Category:")
        cat_label.setStyleSheet("color: #cccccc;")
        cat_layout.addWidget(cat_label)

        self.category_combo = QtWidgets.QComboBox()
        categories = ["All"] + sorted(set(s.get("category", "Other") for s in SHADERS.values()))
        self.category_combo.addItems(categories)
        self.category_combo.currentTextChanged.connect(self._filter_shaders)
        cat_layout.addWidget(self.category_combo)
        layout.addLayout(cat_layout)

        # Shader selection
        shader_layout = QtWidgets.QHBoxLayout()
        shader_label = QtWidgets.QLabel("Shader:")
        shader_label.setStyleSheet("color: #cccccc;")
        shader_layout.addWidget(shader_label)

        self.shader_combo = QtWidgets.QComboBox()
        self.shader_combo.addItems(SHADERS.keys())
        self.shader_combo.currentTextChanged.connect(self._on_shader_changed)
        shader_layout.addWidget(self.shader_combo)
        layout.addLayout(shader_layout)

        # Description
        self.description = QtWidgets.QLabel("")
        self.description.setStyleSheet("color: #888888; font-size: 11px; font-style: italic; padding: 5px 0;")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)

        # Separator
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line2.setStyleSheet("background-color: #444444;")
        layout.addWidget(line2)

        # Parameters section header
        params_header = QtWidgets.QLabel("Parameters")
        params_header.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffffff; padding: 5px 0;")
        layout.addWidget(params_header)

        # Scroll area for parameters
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        self.params_widget = QtWidgets.QWidget()
        self.params_layout = QtWidgets.QVBoxLayout(self.params_widget)
        self.params_layout.setSpacing(10)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.addStretch()

        scroll.setWidget(self.params_widget)
        layout.addWidget(scroll, 1)

        # Reset button
        self.reset_btn = QtWidgets.QPushButton("Reset Parameters")
        self.reset_btn.clicked.connect(self._reset_params)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        layout.addWidget(self.reset_btn)

        # Export button
        self.export_btn = QtWidgets.QPushButton("Export Image")
        self.export_btn.clicked.connect(self.exportRequested.emit)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d5a2d;
                color: #ffffff;
                border: 1px solid #3d7a3d;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d7a3d;
            }
        """)
        layout.addWidget(self.export_btn)

        self.current_group = None
        self._update_params("Original")

    def _filter_shaders(self, category):
        """Filter shader list by category."""
        self.shader_combo.blockSignals(True)
        self.shader_combo.clear()

        for name, shader in SHADERS.items():
            if category == "All" or shader.get("category", "Other") == category:
                self.shader_combo.addItem(name)

        self.shader_combo.blockSignals(False)

        # Select first item
        if self.shader_combo.count() > 0:
            self.shader_combo.setCurrentIndex(0)
            self._on_shader_changed(self.shader_combo.currentText())

    def _on_shader_changed(self, name):
        if not name:
            return
        self._update_params(name)
        self.shaderChanged.emit(name)

    def _update_params(self, shader_name):
        """Update parameter controls for the selected shader."""
        # Clear existing params
        while self.params_layout.count() > 1:
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Update description
        shader_def = SHADERS.get(shader_name, {})
        self.description.setText(shader_def.get("description", ""))

        # Get uniforms
        uniforms = shader_def.get("uniforms", {})

        if not uniforms:
            no_params = QtWidgets.QLabel("No adjustable parameters")
            no_params.setStyleSheet("color: #666666; font-style: italic;")
            self.params_layout.insertWidget(0, no_params)
            return

        # Create parameter group
        default_values = {name: props['default'] for name, props in uniforms.items()}
        self.current_group = ParameterGroup("Shader Parameters", uniforms, default_values)
        self.current_group.paramChanged.connect(self.paramChanged.emit)
        self.params_layout.insertWidget(0, self.current_group)

    def _reset_params(self):
        """Reset all parameters to defaults."""
        shader_name = self.shader_combo.currentText()
        shader_def = SHADERS.get(shader_name, {})
        uniforms = shader_def.get("uniforms", {})

        for name, props in uniforms.items():
            default = props['default']
            self.paramChanged.emit(name, default)

        if self.current_group:
            default_values = {name: props['default'] for name, props in uniforms.items()}
            self.current_group.set_values(default_values)

    def set_image_info(self, path, size):
        """Update image info display."""
        if path:
            filename = os.path.basename(path)
            self.image_info.setText(f"{filename} ({size[0]}x{size[1]})")
        else:
            self.image_info.setText("No image loaded")


class ShaderStudio(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shader Studio")
        self.resize(1200, 800)

        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 5px;
                min-width: 100px;
            }
            QComboBox:hover {
                border-color: #5a9fff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                selection-background-color: #3d5a80;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444444;
                height: 6px;
                background: #2d2d2d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5a9fff;
                border: none;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #7ab5ff;
            }
            QGroupBox {
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #aaaaaa;
            }
            QScrollBar:vertical {
                background: #1e1e1e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #444444;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555555;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        # Central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left panel (canvas + toolbar)
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()

        load_btn = QtWidgets.QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d5a80;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4d6a90;
            }
        """)
        toolbar.addWidget(load_btn)

        toolbar.addStretch()

        # Zoom controls
        zoom_label = QtWidgets.QLabel("View:")
        zoom_label.setStyleSheet("color: #888888;")
        toolbar.addWidget(zoom_label)

        fit_btn = QtWidgets.QPushButton("Fit")
        fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #444444;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
        """)
        toolbar.addWidget(fit_btn)

        left_layout.addLayout(toolbar)

        # Canvas
        self.canvas = ShaderCanvas()
        left_layout.addWidget(self.canvas, 1)

        main_layout.addWidget(left_panel, 1)

        # Right panel (inspector)
        self.inspector = ShaderInspector()
        self.inspector.shaderChanged.connect(self._on_shader_changed)
        self.inspector.paramChanged.connect(self._on_param_changed)
        self.inspector.exportRequested.connect(self.export_image)

        # Wrap inspector in a styled container
        inspector_container = QtWidgets.QWidget()
        inspector_container.setStyleSheet("background-color: #252525; border-left: 1px solid #333333;")
        inspector_layout = QtWidgets.QVBoxLayout(inspector_container)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.addWidget(self.inspector)

        main_layout.addWidget(inspector_container)

        # Load a sample image if available
        self._try_load_sample()

    def _try_load_sample(self):
        """Try to load a sample image from the images_input folder."""
        sample_dir = os.path.join(os.path.dirname(__file__), "images_input")
        if os.path.exists(sample_dir):
            for f in os.listdir(sample_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.canvas.load_texture(os.path.join(sample_dir, f))
                    self._update_image_info(os.path.join(sample_dir, f))
                    break

    def load_image(self):
        """Open file dialog to load an image."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        if path:
            if self.canvas.load_texture(path):
                self._update_image_info(path)

    def _update_image_info(self, path):
        """Update inspector with image info."""
        try:
            img = Image.open(path)
            self.inspector.set_image_info(path, img.size)
        except Exception:
            self.inspector.set_image_info(path, (0, 0))

    def _on_shader_changed(self, name):
        """Handle shader preset change."""
        self.canvas.set_preset(name)

    def _on_param_changed(self, name, value):
        """Handle parameter value change."""
        self.canvas.set_param(name, value)

    def export_image(self):
        """Export the rendered image."""
        if not self.canvas.image_path:
            QtWidgets.QMessageBox.warning(self, "Warning", "No image loaded to export.")
            return

        # Generate default filename
        base_name = os.path.splitext(os.path.basename(self.canvas.image_path))[0]
        shader_name = self.canvas.current_preset.lower().replace(' ', '_')
        default_name = f"{base_name}_{shader_name}.png"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Image",
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )

        if path:
            if self.canvas.export_image(path):
                QtWidgets.QMessageBox.information(self, "Success", f"Image exported to:\n{path}")
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Failed to export image.")


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Set application-wide font
    font = QtGui.QFont("Segoe UI", 10)
    app.setFont(font)

    window = ShaderStudio()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
