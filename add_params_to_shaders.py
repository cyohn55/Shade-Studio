"""
Script to add 6 standard post-processing parameters to every shader in shader_studio_v3.py.

For each shader in the SHADERS dict:
1. Adds uniform entries to the "uniforms" dict (skips if already present)
2. Adds uniform float declarations to the GLSL frag source (skips if already declared)
3. Adds a CUSTOM post-processing GLSL code block that ONLY applies params newly
   added by this script (avoids double-applying params the shader already handles)

Parameters added:
- brightness (-1.0 to 1.0, default 0.0)
- contrast (0.0 to 3.0, default 1.0)
- saturation (0.0 to 3.0, default 1.0)
- gamma (0.1 to 3.0, default 1.0)
- hue_shift (0.0 to 1.0, default 0.0)
- vignette (0.0 to 1.0, default 0.0)
"""

import re
import sys

PARAMS_TO_ADD = {
    "brightness": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
    "contrast":   {"min": 0.0,  "max": 3.0, "default": 1.0, "step": 0.01},
    "saturation": {"min": 0.0,  "max": 3.0, "default": 1.0, "step": 0.01},
    "gamma":      {"min": 0.1,  "max": 3.0, "default": 1.0, "step": 0.01},
    "hue_shift":  {"min": 0.0,  "max": 1.0, "default": 0.0, "step": 0.01},
    "vignette":   {"min": 0.0,  "max": 1.0, "default": 0.0, "step": 0.01},
}

UNIFORM_DECLS = {
    "brightness": "uniform float brightness;",
    "contrast":   "uniform float contrast;",
    "saturation": "uniform float saturation;",
    "gamma":      "uniform float gamma;",
    "hue_shift":  "uniform float hue_shift;",
    "vignette":   "uniform float vignette;",
}

# Individual GLSL snippets for each param (only included if the param is NEW)
PP_SNIPPETS = {
    "brightness": "    _pp += brightness;\n",
    "contrast":   "    _pp = (_pp - 0.5) * contrast + 0.5;\n",
    "saturation": "    float _pp_gray = dot(_pp, vec3(0.299, 0.587, 0.114));\n    _pp = mix(vec3(_pp_gray), _pp, saturation);\n",
    "gamma":      "    _pp = pow(max(_pp, vec3(0.0)), vec3(1.0 / gamma));\n",
    "hue_shift":  """    if (hue_shift > 0.001) {
        vec4 _hK = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
        vec4 _hp = mix(vec4(_pp.bg, _hK.wz), vec4(_pp.gb, _hK.xy), step(_pp.b, _pp.g));
        vec4 _hq = mix(vec4(_hp.xyw, _pp.r), vec4(_pp.r, _hp.yzx), step(_hp.x, _pp.r));
        float _hd = _hq.x - min(_hq.w, _hq.y);
        float _he = 1.0e-10;
        vec3 _hsv = vec3(abs(_hq.z + (_hq.w - _hq.y) / (6.0 * _hd + _he)), _hd / (_hq.x + _he), _hq.x);
        _hsv.x = fract(_hsv.x + hue_shift);
        vec4 _hK2 = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
        vec3 _hp2 = abs(fract(_hsv.xxx + _hK2.xyz) * 6.0 - _hK2.www);
        _pp = _hsv.z * mix(_hK2.xxx, clamp(_hp2 - _hK2.xxx, 0.0, 1.0), _hsv.y);
    }
""",
    "vignette":   """    if (vignette > 0.001) {
        vec2 _vuv = v_uv - 0.5;
        float _vd = dot(_vuv, _vuv);
        _pp *= 1.0 - vignette * _vd * 2.0;
    }
""",
}


def build_pp_code(new_params):
    """Build a custom PP code block that only applies the given new params."""
    if not new_params:
        return ""
    lines = "\n    // --- Post-processing adjustments ---\n"
    lines += "    vec3 _pp = f_color.rgb;\n"
    for p in ["brightness", "contrast", "saturation", "gamma", "hue_shift", "vignette"]:
        if p in new_params:
            lines += PP_SNIPPETS[p]
    lines += "    f_color = vec4(clamp(_pp, 0.0, 1.0), f_color.a);\n"
    return lines


def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    shader_count = 0
    uniforms_added = 0
    glsl_modified = 0
    skipped_params = {}

    # Step 1: Add uniform entries to each shader's "uniforms" dict
    uniforms_pattern = re.compile(r'("uniforms"\s*:\s*\{)')
    matches = list(uniforms_pattern.finditer(content))

    for match in reversed(matches):
        start = match.end()
        depth = 1
        pos = start
        while depth > 0 and pos < len(content):
            if content[pos] == '{':
                depth += 1
            elif content[pos] == '}':
                depth -= 1
            pos += 1
        closing_brace_pos = pos - 1

        uniforms_text = content[start:closing_brace_pos]

        new_entries = []
        for param_name, param_props in PARAMS_TO_ADD.items():
            if re.search(rf'"{re.escape(param_name)}"\s*:', uniforms_text):
                skipped_params[param_name] = skipped_params.get(param_name, 0) + 1
                continue
            entry = f'            "{param_name}": {{"min": {param_props["min"]}, "max": {param_props["max"]}, "default": {param_props["default"]}, "step": {param_props["step"]}}},'
            new_entries.append(entry)
            uniforms_added += 1

        if new_entries:
            insert_text = "\n" + "\n".join(new_entries) + "\n        "
            content = content[:closing_brace_pos] + insert_text + content[closing_brace_pos:]

        shader_count += 1

    print(f"Processed {shader_count} uniform blocks")
    print(f"Added {uniforms_added} uniform entries")
    print(f"Skipped (already present): {skipped_params}")

    # Step 2: Add GLSL uniform declarations and per-shader PP code to each frag source
    frag_pattern = re.compile(r'"frag"\s*:\s*"""')
    matches = list(frag_pattern.finditer(content))

    for match in reversed(matches):
        frag_start = match.end()
        frag_end = content.index('"""', frag_start)
        frag_src = content[frag_start:frag_end]

        original_frag = frag_src

        # Track which params are NEWLY added (not already in the shader)
        new_params = set()
        decls_to_add = []
        for param_name, decl in UNIFORM_DECLS.items():
            if re.search(rf'\buniform\s+float\s+{re.escape(param_name)}\b', frag_src):
                continue
            decls_to_add.append(f"            {decl}")
            new_params.add(param_name)

        if decls_to_add:
            # Insert declarations before 'void main()'
            main_match = re.search(r'\n(\s*)void\s+main\s*\(\s*\)', frag_src)
            if main_match:
                insert_pos = main_match.start()
                decl_block = "\n" + "\n".join(decls_to_add)
                frag_src = frag_src[:insert_pos] + decl_block + frag_src[insert_pos:]

        # Build per-shader PP code (only for newly added params)
        if new_params:
            pp_code = build_pp_code(new_params)
            last_brace = frag_src.rfind('}')
            if last_brace >= 0:
                frag_src = frag_src[:last_brace] + pp_code + "    " + frag_src[last_brace:]
                glsl_modified += 1

        if frag_src != original_frag:
            content = content[:frag_start] + frag_src + content[frag_end:]

    print(f"Modified {glsl_modified} frag sources with post-processing code")

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done!")


if __name__ == "__main__":
    filepath = "shader_studio_v3.py"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    process_file(filepath)
