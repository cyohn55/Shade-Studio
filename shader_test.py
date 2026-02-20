"""
Simple shader test to debug OpenGL rendering issues.
"""

import sys
import numpy as np
import moderngl
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PIL import Image
import os

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

FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_texture, v_uv);
}
"""


class TestCanvas(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.ctx = None
        self.texture = None
        self.vbo = None
        self.vao = None
        self.prog = None
        self.setMinimumSize(512, 512)

        # Set OpenGL format
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)

    def initializeGL(self):
        print("Initializing OpenGL...")

        try:
            self.ctx = moderngl.create_context()
            print(f"OpenGL Version: {self.ctx.info['GL_VERSION']}")
            print(f"OpenGL Vendor: {self.ctx.info['GL_VENDOR']}")
            print(f"OpenGL Renderer: {self.ctx.info['GL_RENDERER']}")
        except Exception as e:
            print(f"Failed to create context: {e}")
            return

        # Create quad
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices)
        print("VBO created")

        # Compile shaders
        try:
            self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
            print("Shaders compiled successfully")
        except Exception as e:
            print(f"Shader compilation error: {e}")
            return

        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', 'in_uv')
        print("VAO created")

        # Create test texture (red/green gradient)
        size = 256
        data = np.zeros((size, size, 4), dtype='uint8')
        for y in range(size):
            for x in range(size):
                data[y, x, 0] = int(x / size * 255)  # Red gradient horizontal
                data[y, x, 1] = int(y / size * 255)  # Green gradient vertical
                data[y, x, 2] = 128
                data[y, x, 3] = 255

        self.texture = self.ctx.texture((size, size), 4, data.tobytes())
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        print("Test texture created (256x256 gradient)")

    def load_image(self, path):
        if not self.ctx:
            print("No context!")
            return

        self.makeCurrent()
        try:
            img = Image.open(path).convert("RGBA")
            print(f"Loaded image: {img.size}")

            # Flip for OpenGL
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

            if self.texture:
                self.texture.release()

            self.texture = self.ctx.texture(img.size, 4, img.tobytes())
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            print("Texture uploaded to GPU")

            self.update()
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()

    def paintGL(self):
        if not self.ctx:
            print("paintGL: No context")
            return

        # Get widget dimensions with DPI scaling
        w = int(self.width() * self.devicePixelRatio())
        h = int(self.height() * self.devicePixelRatio())

        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.2, 0.2, 0.2, 1.0)

        if not self.texture or not self.prog or not self.vao:
            print("paintGL: Missing resources")
            return

        self.texture.use(location=0)
        self.prog['u_texture'].value = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def resizeGL(self, w, h):
        if self.ctx:
            ratio = self.devicePixelRatio()
            self.ctx.viewport = (0, 0, int(w * ratio), int(h * ratio))


class TestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shader Test")
        self.resize(800, 600)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.canvas = TestCanvas()
        layout.addWidget(self.canvas)

        btn_layout = QtWidgets.QHBoxLayout()

        load_btn = QtWidgets.QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(load_btn)

        layout.addLayout(btn_layout)

        self.status = QtWidgets.QLabel("Ready - should show red/green gradient")
        layout.addWidget(self.status)

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.canvas.load_image(path)
            self.status.setText(f"Loaded: {os.path.basename(path)}")


def main():
    print("Starting Shader Test...")
    print(f"Python: {sys.version}")

    app = QtWidgets.QApplication(sys.argv)

    # Set default surface format for all OpenGL widgets
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)

    window = TestWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
