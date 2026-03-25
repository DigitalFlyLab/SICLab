import numpy as np
from PIL import Image, ImageFilter
import os
import math

class EdgeMotionSimulationNPZ:
    def __init__(
        self,
        width=82,
        height=41,
        frame_rate=1000,
        pre_duration=1.0,
        post_duration=1.0,
        speeds=(2, 4, 6, 8, 12),
        angles_deg=range(0, 360, 30),
        output_root='./stimulus/MovingEdgeNPZ/',
        edge_types=('dark_to_light', 'light_to_dark'),
        blur_radius=0.0  
    ):
        self.width = width
        self.height = height
        self.frame_rate = frame_rate

        self.pre_frames = int(pre_duration * frame_rate)
        self.post_frames = int(post_duration * frame_rate)

        self.speeds = speeds
        self.angles_deg = list(angles_deg)
        self.edge_types = edge_types
        self.blur_radius = blur_radius

        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

        y, x = np.mgrid[0:self.height, 0:self.width]
        self.X = x - self.width / 2
        self.Y = y - self.height / 2


    def compute_motion(self, speed, cos_t, sin_t):
        rotated_x = self.X * cos_t - self.Y * sin_t
        min_x = rotated_x.min()
        max_x = rotated_x.max()
        span = max_x - min_x

        margin = 0.1 * span
        total_span = span + 2 * margin
        motion_frames = math.ceil(total_span / speed * self.frame_rate)

        start_pos = min_x - margin
        return motion_frames, start_pos

    def generate_frame(self, frame_idx, speed, cos_t, sin_t, motion_frames, start_pos, edge_type):
        if edge_type == 'dark_to_light':      # OFF
            img = np.zeros((self.height, self.width), dtype=np.uint8)
            edge_color = 255
        else:                                   # ON
            img = np.ones((self.height, self.width), dtype=np.uint8) * 255
            edge_color = 0

        if frame_idx < self.pre_frames:
            offset = 0.0
        elif frame_idx < self.pre_frames + motion_frames:
            offset = speed * (frame_idx - self.pre_frames) / self.frame_rate
        else:
            offset = 1e9

        rotated_x = self.X * cos_t - self.Y * sin_t
        edge_position = rotated_x - (start_pos + offset)
        img[edge_position >= 0] = edge_color

        if self.blur_radius > 0:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
            img = np.array(pil_img, dtype=np.uint8)

        return img

    def generate_npz(self, angle_deg, speed, edge_type):
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        motion_frames, start_pos = self.compute_motion(speed, cos_t, sin_t)
        total_frames = self.pre_frames + motion_frames + self.post_frames


        frames = np.zeros((total_frames, self.height, self.width), dtype=np.uint8)
        for f in range(total_frames):
            frames[f] = self.generate_frame(f, speed, cos_t, sin_t, motion_frames, start_pos, edge_type)

        tag = 'OFF' if edge_type == 'dark_to_light' else 'ON'
        angle_save = (360 - angle_deg) % 360 
        save_name = f"{tag}_edge_angle{angle_save:03d}_speed{speed}.npz"
        save_path = os.path.join(self.output_root, save_name)
        np.savez_compressed(save_path, frames=frames)


    def run_all(self):
        for edge_type in self.edge_types:
            for speed in self.speeds:
                for angle in self.angles_deg:
                    self.generate_npz(angle, speed, edge_type)


def main():
    simulator = EdgeMotionSimulationNPZ(
        width=82,
        height=41,
        frame_rate=1000,
        pre_duration=1.0,
        post_duration=1.0,
        speeds=(2, 4, 8, 12, 16),
        angles_deg=range(0, 360, 30),
        output_root='./stimulus/MovingEdge',
        edge_types=('dark_to_light', 'light_to_dark'),
        blur_radius=0
    )
    simulator.run_all()


if __name__ == "__main__":
    main()
