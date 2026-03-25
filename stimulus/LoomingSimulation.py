import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from multiprocessing import Pool

GLOBAL_SIM = None

def init_worker(sim):
    global GLOBAL_SIM
    GLOBAL_SIM = sim

def process_one_position(pos):
    y, x = pos
    sim = GLOBAL_SIM

    name = f"x{x}_y{y}"
    frames, gif_frames = sim.generate_one_position(x, y)

    sim.save_gif(
        gif_frames,
        os.path.join(sim.output_root, "gif", f"{name}.gif")
    )
    sim.save_npz(
        frames, x, y,
        os.path.join(sim.output_root, "npz", f"{name}.npz")
    )

# ===================== Classical Looming Simulation =====================
class ClassicalLoomingSimulation:
    def __init__(self, mask_path,
                 tau=0.02,
                 r_max_px=20.0,
                 total_duration=5.0,
                 frame_rate=60,
                 pixels_per_degree=20.0,
                 gray_start=0.6,
                 gray_end=0.5,
                 output_root='./stimulus/LoomingDarkDisk'):

        self.frame_rate = frame_rate
        self.tau = tau
        self.r_max_px = r_max_px
        self.ppd = pixels_per_degree

        mask_data = np.load(mask_path)
        self.mask = mask_data['mask'].astype(np.uint8)
        self.height, self.width = self.mask.shape
        self.valid_positions = np.argwhere(self.mask == 1)

        self.r_start = 0.1

        self.R = r_max_px
        self.d_start = self.R / np.tan(self.r_start / self.R)
        self.d_end   = self.R / np.tan(self.r_max_px / self.R)

        self.gray_start_frames = int(gray_start * frame_rate)
        self.gray_end_frames   = int(gray_end * frame_rate)
        self.looming_frames    = int(tau * 10 * frame_rate)
        self.total_frames = self.gray_start_frames + self.looming_frames + self.gray_end_frames

        self.v = (self.d_start - self.d_end) / self.looming_frames * frame_rate

        tau_ms = int(round(tau*1000))
        self.output_root = os.path.join(output_root, f"tau_{tau_ms}ms")
        os.makedirs(os.path.join(self.output_root, "gif"), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, "npz"), exist_ok=True)

        print("\n================ Classical Looming =================")
        print(f"τ = {tau}s ({tau_ms} ms)")
        print(f"Frame rate = {frame_rate} Hz, Total frames = {self.total_frames}")
        print(f"Image size = {self.width} x {self.height}")
        print(f"Valid positions = {len(self.valid_positions)}")
        print(f"Max radius = {r_max_px} px")
        print(f"Output path = {self.output_root}")
        print(f"Computed speed v = {self.v:.4f}")
        print(f"Gray start = {gray_start}s, Gray end = {gray_end}s")
        print(f"Looming frames = {self.looming_frames}")
        print("===================================================")

    def looming_radius(self, frame_idx):
        t = frame_idx / self.frame_rate
        d = max(self.d_start - self.v * t, self.d_end)
        theta = 2.0 * np.arctan(self.R / d)
        r_pix = np.tan(theta / 2.0) * self.R
        return r_pix

    def generate_frame(self, x, y, frame_idx):
        img = np.ones((self.height, self.width), dtype=np.uint8) * 128
        radius = self.looming_radius(frame_idx)
        if radius < 1:
            img[y, x] = 0
        else:
            yy, xx = np.ogrid[:self.height, :self.width]
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)
            img[dist <= radius] = 0
        return img


    def generate_one_position(self, x, y):
        frames = np.zeros((self.total_frames, self.height, self.width), dtype=np.uint8)
        gif_frames = []
        for f in range(self.total_frames):
            if f < self.gray_start_frames:
                frame = np.ones((self.height, self.width), dtype=np.uint8) * 128
            elif f < self.gray_start_frames + self.looming_frames:
                frame_idx = f - self.gray_start_frames
                frame = self.generate_frame(x, y, frame_idx)
            else:
                frame = self.generate_frame(x, y, self.looming_frames - 1)

            frames[f] = frame
            gif_frames.append(Image.fromarray(frame))
        return frames, gif_frames

    def save_gif(self, gif_frames, path):
        duration_ms = int(1000 / self.frame_rate)
        gif_frames[0].save(
            path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0
        )

    def save_npz(self, frames, x, y, path):
        np.savez_compressed(
            path,
            frames=frames,
            x=x,
            y=y,
            frame_rate=self.frame_rate,
            tau=self.tau,
            R=self.R,
            v=self.v,
            r_start=self.r_start,
            r_max_px=self.r_max_px,
            gray_start_frames=self.gray_start_frames,
            gray_end_frames=self.gray_end_frames,
            looming_frames=self.looming_frames,
            total_frames=self.total_frames
        )

    def run_all_parallel(self, num_workers=8):
        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(self,)
        ) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        process_one_position,
                        self.valid_positions
                    ),
                    total=len(self.valid_positions),
                    desc="processing positions"
                )
            )

def main():
    mask_path = r'/home/jzyh/xjp_projects/FP_dev/results/combined_eye_mask_41x82.npz'
    output_root = './stimulus/LoomingDarkDisk_test'

    # τ = 20ms
    sim_20ms = ClassicalLoomingSimulation(
        mask_path=mask_path,
        tau=0.02,        # 20ms
        r_max_px=20.0,
        frame_rate=1000,
        pixels_per_degree=10.0,
        gray_start=0.6,
        gray_end=1,
        output_root=output_root
    )
    sim_20ms.run_all_parallel(num_workers=100)

    # τ = 80ms
    sim_80ms = ClassicalLoomingSimulation(
        mask_path=mask_path,
        tau=0.08,        # 80ms
        r_max_px=20.0,
        frame_rate=1000,
        pixels_per_degree=10.0,
        gray_start=0.6,
        gray_end=1,
        output_root=output_root
    )
    sim_80ms.run_all_parallel(num_workers=100)

if __name__ == "__main__":
    main()
