---
name: run-shader-studio
description: Launch Shader Studio v3 for testing with crash logging
disable-model-invocation: true
---

# Run Shader Studio

Launch the Shader Studio v3 application for testing. Handles killing any existing instance and capturing crash output.

## Steps

1. Kill any existing Python process running shader_studio_v3.py (use `taskkill` or check `tasklist` first)
2. Launch the app in the background with crash logging:
   ```bash
   cd '/c/Users/cyohn/Desktop/Portfolio/Blender Experiments' && python shader_studio_v3.py > crash_log.txt 2>&1 &
   ```
3. Wait 3 seconds, then check `crash_log.txt` for any startup errors
4. Report whether the app launched successfully or show any errors found

## Notes

- The app sometimes crashes silently. Always redirect stderr.
- If crash_log.txt has content after startup, something went wrong — show the user the output.
- If the log is empty after a few seconds, the app started successfully.
