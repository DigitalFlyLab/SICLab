import glob
import numpy as np
import os
import matplotlib.pyplot as plt

class StimulusProcessor:
    def __init__(
        self,
        target_dt_ms=1.0,
        target_size=(41, 82),
        is_visual=False,
        fps=1000.0,
        downsample=1,
        save_dir="results"
    ):
        self.target_dt_ms = target_dt_ms
        self.target_size = target_size
        self.is_visual = is_visual
        self.fps = fps
        self.save_dir = save_dir
        self.downsample = downsample

        os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Load stimulus from npz
    # --------------------------------------------------
    def load_npz(self, filepath, verbose=False):
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        data = np.load(filepath)
        frames = data['frames']  # (T, H, W)
        # 转换为 (H, W, T) 并归一化到 0-1
        stimulus = np.transpose(frames, (1, 2, 0)).astype(np.float32) / 255.0
        if self.downsample > 1:
            stimulus = stimulus[:, :, ::self.downsample]
        if verbose:
            print(f"[{os.path.basename(filepath)}]")
            print(f"  Original frames: {frames.shape[0]}")
            print(f"  Stimulus shape (H, W, T): {stimulus.shape}")
            print(f"  Min / Max values: {stimulus.min():.3f} / {stimulus.max():.3f}")
            print(f"  Data type: {stimulus.dtype}")

        return stimulus

    # --------------------------------------------------
    # 2. Per-second projection
    # --------------------------------------------------
    def compute_secondly_projection(self, stimulus):
        H, W, T = stimulus.shape
        frames_per_sec = int(self.fps / 5)  # 每秒 5 帧?
        num_sec = T // frames_per_sec

        projections = []
        for s in range(num_sec):
            seg = stimulus[:, :, s*frames_per_sec:(s+1)*frames_per_sec]

            # 取这一秒中间那一帧
            mid_idx = frames_per_sec // 2
            frame = seg[:, :, mid_idx]

            projections.append(frame)

        return projections

    # --------------------------------------------------
    # 3. Visualization (ONE stimulus = ONE figure)
    # --------------------------------------------------
    def visualize(self, projections, stim_name):
        if not self.is_visual:
            return

        n = len(projections)
        fig, axes = plt.subplots(1, n, figsize=(1*n, 4))

        if n == 1:
            axes = [axes]

        for i, (proj, ax) in enumerate(zip(projections, axes)):
            im = ax.imshow(proj, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{(i+1)/5}s")
            ax.axis("off")

        plt.suptitle(stim_name)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 4. Single stimulus pipeline
    # --------------------------------------------------
    def process_npz(self, filepath, verbose=False):
        name = os.path.splitext(os.path.basename(filepath))[0]

        stimulus = self.load_npz(filepath, verbose=verbose)
        projections = self.compute_secondly_projection(stimulus)

        self.visualize(projections, name)

        return stimulus, projections

    # --------------------------------------------------
    # 5. Batch processing
    # --------------------------------------------------
    def process_folder(self, pattern="*.npz", verbose=False):
        dataset = {}

        for f in sorted(glob.glob(pattern)):
            print(f"Processing: {f}")
            stim, proj = self.process_npz(f, verbose=verbose)
            dataset[os.path.splitext(f)[0]] = stim

        return dataset
