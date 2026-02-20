import sys
import numpy as np
import moderngl
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PIL import Image

# --- SHADER LIBRARY ---
SHADERS = {
    "Original": {
        "uniforms": {"rotation": 0.0},
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
    "Sobel Edge": {
        "uniforms": {"edge_intensity": 2.0, "threshold": 0.2, "rotation": 0.0},
        "frag": """
            #version 330
            uniform sampler2D u_texture;
            uniform float edge_intensity;
            uniform float threshold;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                vec2 off = 1.0 / textureSize(u_texture, 0);
                float x[9], y[9];
                for(int i=0; i<3; i++) for(int j=0; j<3; j++) {
                    vec3 c = texture(u_texture, v_uv + vec2(i-1, j-1) * off).rgb;
                    x[i*3+j] = y[i*3+j] = dot(c, vec3(0.299, 0.587, 0.114));
                }
                float gx = (x[2]+2*x[5]+x[8]) - (x[0]+2*x[3]+x[6]);
                float gy = (y[0]+2*y[1]+y[2]) - (y[6]+2*y[7]+y[8]);
                float g = sqrt(gx*gx + gy*gy) * edge_intensity;
                vec4 base = texture(u_texture, v_uv);
                f_color = (g > threshold) ? vec4(0,0,0,1) : base;
            }
        """
    }
}

class ShaderCanvas(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.ctx = None
        self.texture = None
        self.vbo = None
        self.vao = None
        self.prog = None
        self.current_preset = "Original"
        self.params = SHADERS[self.current_preset]["uniforms"].copy()

    def initializeGL(self):
        # Create context and force it to be current
        self.ctx = moderngl.create_context()
        
        # Standard Quad: [x, y, u, v]
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0, 
             1.0, -1.0, 1.0, 0.0, 
            -1.0,  1.0, 0.0, 1.0, 
             1.0,  1.0, 1.0, 1.0
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        
        # Start with a simple 2x2 blue/red texture to prove it's working
        test_pixels = np.array([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255], dtype='u1')
        self.texture = self.ctx.texture((2, 2), 4, test_pixels.tobytes())
        
        self.update_shader()

    def update_shader(self):
        if not self.ctx: return
        
        self.makeCurrent() # Ensure we are editing the right GPU context
        vert_src = """
            #version 330
            uniform float rotation;
            in vec2 in_vert; 
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                float s = sin(rotation);
                float c = cos(rotation);
                mat2 rot = mat2(c, -s, s, c);
                gl_Position = vec4(rot * in_vert, 0.0, 1.0);
                v_uv = in_uv;
            }
        """
        if self.prog: self.prog.release()
        self.prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=SHADERS[self.current_preset]["frag"])
        
        if self.vao: self.vao.release()
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', 'in_uv')

    def load_texture(self, img):
        if not self.ctx: return
        self.makeCurrent()
        
        img = img.convert("RGBA")
        if self.texture: self.texture.release()
        
        self.texture = self.ctx.texture(img.size, 4, img.tobytes())
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.update()

    def paintGL(self):
        if not self.ctx or not self.texture: return
        
        self.ctx.clear(0.1, 0.1, 0.1)
        self.texture.use(location=0)
        
        if self.prog:
            if 'u_texture' in self.prog:
                self.prog['u_texture'].value = 0
            for name, val in self.params.items():
                if name in self.prog:
                    self.prog[name].value = val
        
        if self.vao:
            self.vao.render(moderngl.TRIANGLE_STRIP)

class ShaderTool(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defensive Shader Studio")
        self.resize(1000, 700)
        
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QHBoxLayout(main_widget)
        
        self.canvas = ShaderCanvas()
        layout.addWidget(self.canvas, 4)
        
        sidebar = QtWidgets.QVBoxLayout()
        layout.addLayout(sidebar, 1)
        
        btn_img = QtWidgets.QPushButton("Upload PNG")
        btn_img.clicked.connect(self.upload_image)
        sidebar.addWidget(btn_img)
        
        self.drop = QtWidgets.QComboBox()
        self.drop.addItems(SHADERS.keys())
        self.drop.currentTextChanged.connect(self.change_shader)
        sidebar.addWidget(self.drop)
        
        self.uniform_container = QtWidgets.QVBoxLayout()
        sidebar.addLayout(self.uniform_container)
        sidebar.addStretch()
        self.build_ui()

    def upload_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName()
        if path:
            img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
            self.canvas.load_texture(img)

    def change_shader(self, name):
        self.canvas.current_preset = name
        self.canvas.params = SHADERS[name]["uniforms"].copy()
        self.canvas.update_shader()
        self.build_ui()
        self.canvas.update()

    def build_ui(self):
        while self.uniform_container.count():
            item = self.uniform_container.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for k, v in self.canvas.params.items():
            label = QtWidgets.QLabel(f"{k}: {v:.2f}")
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(v * 10))
            slider.valueChanged.connect(lambda val, key=k, lbl=label: self.update_val(key, val/10, lbl))
            self.uniform_container.addWidget(label)
            self.uniform_container.addWidget(slider)

    def update_val(self, key, val, label):
        self.canvas.params[key] = val
        label.setText(f"{key}: {val:.2f}")
        self.canvas.update()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ShaderTool()
    window.show()
    sys.exit(app.exec())