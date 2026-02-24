---
name: add-shader
description: Add a new GLSL shader effect to Shader Studio v3
---

# Add Shader

Add a new shader effect to the `SHADERS` dict in `shader_studio_v3.py`.

## Arguments

- `name` (required): The display name for the shader (e.g. "Frosted Glass")
- `category` (required): One of: Basic, Stylized, Edge Detection, Color, Post-Processing, Blur, Distortion, Lighting, Artistic, Procedural Texture, Special Effects
- `description` (required): Brief description of the effect

## Template

Every shader entry must follow this exact pattern:

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
        // declare uniforms matching the keys above
        in vec2 v_uv;
        out vec4 fragColor;
        void main() {
            // shader logic here
        }
    """
}
```

## Rules

1. Use `#version 330 core` — no exceptions
2. Always declare `uniform sampler2D u_texture;` and `uniform vec2 u_resolution;`
3. Input varying is `in vec2 v_uv;`, output is `out vec4 fragColor;`
4. Uniform names in GLSL **must exactly match** the keys in the `uniforms` dict
5. Add the entry to the `SHADERS` dict in `shader_studio_v3.py`, grouped with shaders of the same category
6. Provide sensible min/max/default/step values for each parameter
7. Include at least one adjustable uniform parameter so users can tweak the effect
8. Test the shader compiles by running a syntax check after editing

## Example

```python
"Frosted Glass": {
    "category": "Distortion",
    "description": "Simulates a frosted glass surface with adjustable blur and distortion",
    "uniforms": {
        "frost_amount": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01},
        "blur_radius": {"min": 1.0, "max": 20.0, "default": 5.0, "step": 0.5},
        "distortion": {"min": 0.0, "max": 0.1, "default": 0.02, "step": 0.001},
    },
    "frag": """#version 330 core
        uniform sampler2D u_texture;
        uniform vec2 u_resolution;
        uniform float frost_amount;
        uniform float blur_radius;
        uniform float distortion;
        in vec2 v_uv;
        out vec4 fragColor;
        // ... shader implementation ...
    """
}
```
