# Shader Studio v3

## Project Overview

Single-file PyQt6 + PyOpenGL desktop application for applying real-time GLSL shader effects to images and 3D models. The main file `shader_studio_v3.py` is ~11,000+ lines containing all application code, shader definitions, and UI.

## How to Run

```bash
python shader_studio_v3.py
```

To capture crash output:
```bash
python shader_studio_v3.py 2>&1 | tee crash_log.txt
```

## Version Control

Git repo initialized. `.gitignore` excludes `__pycache__`, `build/`, `dist/`, logs, PNGs, `.bak` files, and 3D model binaries.

## Claude Code Setup

### Hooks (`.claude/settings.local.json`)
- **PreToolUse**: Blocks Edit/Write to `.bak`, `shader_studio_v2.py`, `shader_studio.py` (protected files)
- **PostToolUse**: Runs `py_compile` syntax check on every `.py` file edit

### Skills
- `/run-shader-studio` — Kill existing instance, launch app with crash logging
- `/add-shader` — Add new GLSL shader effect following project conventions

### Environment
- Windows with Git Bash: use `/c/Users/...` paths in Bash, `C:\Users\...` in Read/Edit/Write tools
- `_update_params_ui(shader_name)` requires the shader name argument

## Dependencies

- **Required**: PyQt6, PyOpenGL, NumPy, Pillow
- **Optional**: OpenCV (cv2) for video, imageio for GIF export, PyTorch for AI upscaling, pygltflib for GLTF models
- **Python**: 3.12

No requirements.txt exists. Install manually: `pip install PyQt6 PyOpenGL numpy Pillow`

## Build

PyInstaller build uses `shade_studio.spec`. Entry point is currently set to `shader_studio_v2.py` (needs updating to v3).

```bash
pyinstaller --clean shade_studio.spec
```

## Architecture

### File Structure

| File | Purpose |
|------|---------|
| `shader_studio_v3.py` | **Active main app** (all code in one file) |
| `shader_studio_v2.py` | Previous version (do NOT edit) |
| `shader_studio.py` | Original version (do NOT edit) |
| `shader_presets.json` | User-saved shader presets |
| `shader_test.py` | Minimal OpenGL test widget |
| `shade_studio.spec` | PyInstaller build config |

### Key Classes (in shader_studio_v3.py)

| Class | Line | Purpose |
|-------|------|---------|
| `SimpleUpscaler` | ~154 | AI/Lanczos image upscaling |
| `ShaderCanvas` | ~7168 | Main OpenGL rendering widget (QOpenGLWidget) |
| `ParameterSlider` | ~8547 | Custom slider widget for shader params |
| `ShaderEditorDialog` | ~8600 | GLSL code editor dialog |
| `ShaderStudio` | ~8740 | Main application window (QMainWindow) |

### Shader System

All shaders are defined in the `SHADERS` dict at the top of the file (~80+ shaders). Each entry follows this pattern:

```python
"Shader Name": {
    "category": "Category",
    "description": "What it does",
    "uniforms": {
        "param_name": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01},
    },
    "frag": """#version 330 core
        uniform sampler2D u_texture;
        uniform vec2 u_resolution;
        // ... uniform declarations matching uniforms dict ...
        in vec2 v_uv;
        out vec4 fragColor;
        void main() { ... }
    """
}
```

All shaders use OpenGL 3.3 core profile (`#version 330 core`).

### Rendering Pipeline

**Working pattern for shader rendering**:
1. `_compile_shader()` compiles vertex + fragment, links program, sets up VAO vertex attributes
2. `paintGL()` renders to screen for viewport display
3. `_export_2d_single(program, params, w, h)` renders to a temporary FBO, reads pixels via `glReadPixels` - this is the proven workhorse for all off-screen rendering

**Multi-pass shader chaining** (batch mode):
1. Render pass N with `_export_2d_single()` -> get numpy array
2. Upload result as new texture with `_upload_texture(np.ascontiguousarray(np.flipud(image_data)))`
3. Render pass N+1 with the new texture
4. Repeat

**Bake Pass** (interactive chaining):
1. User applies shader, adjusts params, clicks "Bake Pass"
2. `bake_current_pass()` renders via `_export_2d_single`, uploads result as new source texture
3. Chain is recorded in `_bake_chain` list for later batch replay
4. User can then apply another shader on top
5. "Reset Chain" reloads the original image

### Key Methods on ShaderCanvas

| Method | Purpose |
|--------|---------|
| `load_texture(path)` | Load image file as OpenGL texture |
| `_upload_texture(data)` | Upload numpy array as GL texture |
| `_export_2d_single(program, params, w, h)` | Render shader to FBO, return numpy array |
| `_compile_shader()` | Compile current shader program with VAO setup |
| `bake_current_pass()` | Commit current shader as new source texture |
| `reset_chain()` | Reload original image, clear bake chain |
| `paintGL()` | Main render entry point |
| `batch_process()` | Batch dialog with shader chain UI |
| `_batch_export_individual()` | Multi-pass batch export (accepts `shader_chain`) |
| `_batch_chain_refresh_list()` | Refresh chain list widget (call after modifying `_batch_shader_chain`) |

## Critical Bugs & Gotchas

### np.flipud Non-Contiguous Arrays

**Bug**: `np.flipud()` returns a non-contiguous array view. OpenGL's `glTexImage2D` may not correctly read non-contiguous memory, causing a diagonal artifact (top-left half black, bottom-right half shows image).

**Fix**: Always wrap with `np.ascontiguousarray()`:
```python
self._upload_texture(np.ascontiguousarray(np.flipud(image_data)))
```

### Silent Crashes

The app sometimes crashes silently (no traceback). Always redirect stderr when testing:
```bash
python shader_studio_v3.py 2>&1 | tee crash_log.txt
```

## Conventions

### Adding a New Shader

1. Add entry to the `SHADERS` dict following the existing pattern
2. Include `category`, `description`, `uniforms` dict, and `frag` GLSL source
3. All fragment shaders must declare `#version 330 core`
4. Required inputs: `uniform sampler2D u_texture;`, `uniform vec2 u_resolution;`, `in vec2 v_uv;`, `out vec4 fragColor;`
5. Uniform names in GLSL must match the keys in the `uniforms` dict exactly

### Shader Categories

Basic, Stylized, Edge Detection, Color, Post-Processing, Blur, Distortion, Lighting, Artistic, Procedural Texture, Special Effects

### UI Patterns

- Inspector panel on the right side with shader combo, parameters (auto-generated sliders), preset management, export/batch/upscale buttons, bake pass controls
- Parameters auto-generate `ParameterSlider` widgets from the shader's `uniforms` dict
- Dark theme throughout (background ~#2a2a2a, text ~#ddd)
- Button styling tiers: `compact_btn` (11px font, 3px 6px padding) for primary buttons (Export, Batch, Upscale), `tiny_btn` (11px font, 2px 4px padding) for secondary buttons (Bake, Reset, Undo, Redo)
- No category filter dropdown — `_filter_shaders("All")` is called directly when refreshing the shader list

## State Variables (ShaderCanvas.__init__)

Key state groups:
- **Core**: `program`, `texture_id`, `image_size`, `image_path`, `params`, `current_preset`
- **Bake Chain**: `_bake_chain`, `_original_image_data`, `_original_image_path`
- **FBO**: `fbo`, `fbo_texture` (for post-processing)
- **3D Mode**: `mode_3d`, vertices/normals/UVs, multiple light sources, camera
- **Animation**: `gif_frames`, `gif_timer`, `gif_playing`
- **Undo/Redo**: history stack for parameter changes
