import numpy as np
from vispy import gloo


class Camera3D:
    def __init__(self, color=[0, 1, 0], scaler=1.0):
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self._setup_camera(color=color, scaler=scaler)

    def _setup_camera(self, color=[0, 1, 0], scaler=1.0):
        self.camera_vertices = [
            [0, 0, -0.3 * scaler],
            [-0.2 * scaler, -0.2 * scaler, 0.3 * scaler],
            [-0.2 * scaler, 0.2 * scaler, 0.3 * scaler],
            [0.2 * scaler, -0.2 * scaler, 0.3 * scaler],
            [0.2 * scaler, 0.2 * scaler, 0.3 * scaler],
        ]
        colors = [color, color, color, color, color]
        indices = [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 2, 4, 3, 4]
        vertices_type = [
            ("a_position", np.float32, 3),
            ("a_color", np.float32, 3),
        ]
        self.collated = np.asarray(list(zip(self.camera_vertices, colors)), vertices_type)
        self.vertex_buffer = gloo.VertexBuffer(self.collated)
        self.index_buffer = gloo.IndexBuffer(indices)
