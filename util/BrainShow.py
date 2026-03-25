import navis
import flybrains
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BrainShow:
    def __init__(self,
                 swc_path: str):
        """
        Brain visualization with response-based coloring.

        Args:
            swc_path: Directory containing all SWC files
        """
        self.swc_path = swc_path

        # Color definitions
        self.positive_color = '#800020'  # Wine red
        self.negative_color = '#0ABAB5'  # Tiffany blue

        # Load responses
        self.responses = None
        self.neuron_ids = None
        self.neuron_to_index = None

        # Placeholders for skeletons
        self.id_to_skeleton = {}  # Only store skeletons for specified IDs

        # Global response statistics
        self.global_max_abs_r = None  # Maximum absolute response for selected neurons
        
        print("BrainShow initialized")

    def load_responses(self, filepath: str, verbose: bool = True):
        """Load neuron responses from .npz file."""
        data = np.load(filepath, allow_pickle=True)
        self.responses = data['responses']          # (N, T)
        self.neuron_ids = data['neuron_ids'].astype(int)
        self.neuron_to_index = {
            nid: i for i, nid in enumerate(self.neuron_ids)
        }
        if verbose:
            print(f"Responses loaded: {self.responses.shape}")

    # ------------------------------------------------------------------
    # OPTIMIZED SWC LOADING - ONLY LOAD SPECIFIED IDS
    # ------------------------------------------------------------------
    def load_specific_skeletons(self, neuron_ids: List[int]):
        """
        Load only the SWC files for the specified neuron IDs.
        
        Args:
            neuron_ids: List of neuron IDs to load
        """
        print(f"Loading skeletons for {len(neuron_ids)} specified neurons...")
        
        # Clear previous skeletons
        self.id_to_skeleton = {}
        
        # Get all SWC files in directory
        swc_files = list(Path(self.swc_path).glob("*.swc"))
        
        # Filter files based on neuron IDs
        neuron_ids_set = set(neuron_ids)
        files_to_load = []
        
        # First pass: identify which files to load
        for swc_file in swc_files:
            # Extract ID from filename (assuming format: {id}.swc or similar)
            try:
                file_id = int(swc_file.stem.split('.')[0])
                if file_id in neuron_ids_set:
                    files_to_load.append(swc_file)
            except ValueError:
                # If filename doesn't contain ID, skip
                continue
        
        print(f"Found {len(files_to_load)} SWC files for specified neurons")
        
        # Load skeletons in parallel
        if files_to_load:
            skeletons = navis.read_swc(files_to_load, parallel=True)
            
            # Build ID -> skeleton mapping
            for sk in skeletons:
                try:
                    self.id_to_skeleton[int(sk.id)] = sk
                except (ValueError, AttributeError):
                    continue
        
        print(f"Loaded {len(self.id_to_skeleton)} skeletons for specified neurons")
        
        # Check for missing skeletons
        loaded_ids = set(self.id_to_skeleton.keys())
        missing_ids = neuron_ids_set - loaded_ids
        
        if missing_ids:
            print(f"Warning: {len(missing_ids)} neurons have no SWC files. Missing IDs: {list(missing_ids)[:10]}{'...' if len(missing_ids) > 10 else ''}")

    def _compute_global_max_response(self, valid_ids: np.ndarray):
        """
        Compute maximum positive and maximum negative response values
        for the selected neurons across all time frames.

        Args:
            valid_ids: Array of valid neuron IDs with both responses and skeletons
        """
        # Get indices for valid neurons in responses array
        idxs = np.array([self.neuron_to_index[nid] for nid in valid_ids])
        
        # Extract responses for selected neurons only
        selected_responses = self.responses[idxs, :]
        
        # Maximum positive value
        self.global_max_pos_r = np.max(selected_responses)
        
        # Maximum negative value (most negative, i.e., minimum value)
        self.global_max_neg_r = np.min(selected_responses)
        
        print(f"Global maximum positive response: {self.global_max_pos_r:.4f}")
        print(f"Global maximum negative response: {self.global_max_neg_r:.4f}")


    # ------------------------------------------------------------------
    def _compute_global_max_response(self, valid_ids: np.ndarray):
        """
        Compute robust maximum positive and maximum negative response values
        for the selected neurons across all time frames using the 95th percentile,
        to reduce the influence of outliers.

        Args:
            valid_ids: Array of valid neuron IDs with both responses and skeletons
        """
        # Get indices for valid neurons in responses array
        idxs = np.array([self.neuron_to_index[nid] for nid in valid_ids])
        
        # Extract responses for selected neurons only
        selected_responses = self.responses[idxs, :]
        
        # 95th percentile for positive responses
        pos_values = selected_responses[selected_responses > 0]
        if pos_values.size > 0:
            self.global_max_pos_r = np.percentile(pos_values, 98)
        else:
            self.global_max_pos_r = 0.0
        
        # 5th percentile for negative responses (most negative)
        neg_values = selected_responses[selected_responses < 0]
        if neg_values.size > 0:
            self.global_max_neg_r = np.percentile(neg_values, 2)
        else:
            self.global_max_neg_r = 0.0
        
        print(f"Robust 98th percentile positive response: {self.global_max_pos_r:.4f}")
        print(f"Robust 2th percentile negative response: {self.global_max_neg_r:.4f}")


    # ------------------------------------------------------------------
    def _compute_frame_activity(self, time_idx: int, valid_idxs: np.ndarray):
        """
        Vectorized computation of color and opacity for selected neurons,
        with positive and negative responses normalized separately.
        Uses sqrt scaling for opacity and threshold-linear rescaling.
        """
        # Get responses for selected neurons at this time frame
        responses_t = self.responses[valid_idxs, time_idx]

        opacity = np.abs(responses_t)

        # Threshold small values and linearly rescale
        threshold = 0.001
        target_min = 0.1
        max_alpha = 0.6

        mask = opacity >= threshold

        opacity_new = np.zeros_like(opacity)

        # linear projection: [threshold, max_alpha] -> [target_min, max_alpha]
        opacity_new[mask] = target_min + (opacity[mask] - threshold) * (max_alpha - target_min) / (max_alpha - threshold)

        opacity = opacity_new
        # Masks
        active_mask = opacity > 0
        positive_mask = responses_t > 0

        return active_mask, positive_mask, opacity

    # ------------------------------------------------------------------
    def plot_frames(self,
                    responses_path: str,
                    neuron_ids: list,
                    stimulus_path: str,
                    output_dir: str,
                    frame_interval: int = 10,
                    dpi: int = 300,
                    figsize=(20, 12),
                    video_fps: int = 10,
                    crf: int = 18):
        """
        Render neuron activity with stimulus preview.
        Stimulus loaded from MP4 (color).
        Shows dynamic neuron count curves for positive and negative responses separately.
        Left-bottom corner shows pixelated stimulus with eye mask overlay.
        """
        import os
        import gc
        import numpy as np
        import matplotlib.pyplot as plt
        import imageio
        import cv2
        from tqdm import tqdm
        import navis
        import flybrains

        frame_time_ms = 10
        os.makedirs(output_dir, exist_ok=True)

        # ----------------------------
        # Load responses
        # ----------------------------
        self.load_responses(responses_path)

        # ----------------------------
        # Load stimulus video
        # ----------------------------
        print("📹 Loading stimulus video...")
        cap = cv2.VideoCapture(stimulus_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {stimulus_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = 1000
        repeat_factor = max(1, int(round(target_fps / src_fps)))

        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for _ in range(repeat_factor):
                frames_list.append(frame)
        cap.release()

        if len(frames_list) == 0:
            raise ValueError("No frames loaded from stimulus video.")

        stim_data = np.stack(frames_list, axis=0)  # (T,H,W,3)

        # ----------------------------
        # Trim first 500 frames
        # ----------------------------
        if stim_data.shape[0] > 500:
            stim_data = stim_data[500:]
        n_stim_frames = stim_data.shape[0]

        # ----------------------------
        # Load skeletons
        # ----------------------------
        if not self.id_to_skeleton:
            self.load_specific_skeletons(neuron_ids)

        valid_ids = np.array([
            nid for nid in neuron_ids
            if nid in self.neuron_to_index and nid in self.id_to_skeleton
        ])
        if len(valid_ids) == 0:
            raise ValueError("No valid neuron IDs found!")

        self._compute_global_max_response(valid_ids)
        idxs = np.array([self.neuron_to_index[nid] for nid in valid_ids])
        skeleton_cache = {nid: self.id_to_skeleton[nid] for nid in valid_ids}

        flywire_mesh = flybrains.FLYWIRE.mesh
        n_time = self.responses.shape[1]
        time_points = range(0, n_time, frame_interval)

        views = {"view": ("x", "-y", "-z")}

        # ----------------------------
        # Load eye mask once
        # ----------------------------
        mask_npz_path = './results/combined_eye_mask_41x82.npz'
        mask_data = np.load(mask_npz_path)
        eye_mask = mask_data['mask'].astype(bool)
        mask_height, mask_width = eye_mask.shape

        # ----------------------------
        # Initialize neuron activity history
        # ----------------------------
        activity_history_pos = []
        activity_history_neg = []

        prev_stim_for_diff = None  # used for frame difference

        for view_name, view in views.items():
            view_dir = os.path.join(output_dir, view_name)
            os.makedirs(view_dir, exist_ok=True)
            mp4_path = os.path.join(view_dir, "animation.mp4")

            xlim = (40491.626953125, 955048.255859375)
            ylim = (455903.673046875, 13699.866015624997)

            # ----------------------------
            # Initialize figure
            # ----------------------------
            fig = plt.figure(figsize=figsize, facecolor="black")
            gs = fig.add_gridspec(
                2, 2,
                height_ratios=[3, 1],
                width_ratios=[1, 2],
                hspace=0.25,
                wspace=0.15
            )

            ax_stim = fig.add_subplot(gs[0, 0])
            ax_neuron = fig.add_subplot(gs[0, 1])
            ax_heatmap = fig.add_subplot(gs[1, 0])
            ax_activity = fig.add_subplot(gs[1, 1])

            # Initialize heatmap
            heatmap_img = ax_heatmap.imshow(
                np.zeros((mask_height, mask_width)),
                cmap='bwr', origin='lower', vmin=-255, vmax=255
            )
            ax_heatmap.set_facecolor('black')
            ax_heatmap.axis('off')
            ax_heatmap.set_title("Binocular visual contrast input", color="white", fontsize=10)

            writer = imageio.get_writer(
                mp4_path,
                format="FFMPEG",
                mode="I",
                fps=video_fps,
                codec="libx264",
                output_params=[
                    "-crf", str(crf),
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p"
                ]
            )

            pbar = tqdm(time_points, desc=view_name, dynamic_ncols=True)

            for t_idx, t in enumerate(pbar):
                time_sec = (t * frame_time_ms) / 1000.0

                # ----------------------------
                # Compute neuron activity
                # ----------------------------
                active_mask, positive_mask, opacity = self._compute_frame_activity(t, idxs)

                n_active_pos = np.sum(active_mask & positive_mask)
                n_active_neg = np.sum(active_mask & ~positive_mask)

                activity_history_pos.append(n_active_pos)
                activity_history_neg.append(n_active_neg)

                # ----------------------------
                # Stimulus frame index
                # ----------------------------
                stim_frame_idx = min(t_idx * frame_interval, n_stim_frames - 1)

                # ----------------------------
                # Plot stimulus
                # ----------------------------
                ax_stim.clear()
                ax_stim.set_facecolor("black")
                ax_stim.imshow(stim_data[stim_frame_idx])
                ax_stim.set_title(f"Stimulus t={time_sec:.2f}s", fontsize=12, color="white")
                ax_stim.axis("off")

                # ----------------------------
                # Plot neurons
                # ----------------------------
                ax_neuron.clear()
                ax_neuron.set_facecolor("black")
                ax_neuron.axis("off")
                navis.plot2d(flywire_mesh, view=view, ax=ax_neuron, color="#888888", alpha=0.15, linewidth=0.3)

                # Plot neurons
                if np.sum(active_mask) > 0:

                    visible_ids = valid_ids[active_mask]
                    visible_opacity = opacity[active_mask]
                    visible_positive = positive_mask[active_mask]

                    # Split positive and negative neurons
                    pos_ids = visible_ids[visible_positive]
                    pos_opacity = visible_opacity[visible_positive]

                    neg_ids = visible_ids[~visible_positive]
                    neg_opacity = visible_opacity[~visible_positive]

                    # Batch plot positive neurons
                    if len(pos_ids) > 0:
                        pos_skeletons = [skeleton_cache[nid] for nid in pos_ids]
                        navis.plot2d(
                            pos_skeletons,
                            view=view,
                            ax=ax_neuron,
                            color="red",
                            alpha=pos_opacity
                        )

                    # Batch plot negative neurons
                    if len(neg_ids) > 0:
                        neg_skeletons = [skeleton_cache[nid] for nid in neg_ids]
                        navis.plot2d(
                            neg_skeletons,
                            view=view,
                            ax=ax_neuron,
                            color="blue",
                            alpha=neg_opacity
                        )
                        
                ax_neuron.set_xlim(xlim)
                ax_neuron.set_ylim(ylim)
                ax_neuron.set_title(f"Drosophila brain activity: t={time_sec:.2f}s | +:{n_active_pos} -:{n_active_neg}", fontsize=12, color="white")

                # ----------------------------
                # Update activity axis
                # ----------------------------
                ax_activity.clear()
                ax_activity.set_facecolor("black")
                times_sec = frame_interval * np.arange(len(activity_history_pos)) * (frame_time_ms / 1000.0)
                ax_activity.plot(times_sec, activity_history_pos, color="red", lw=1.5, label="Positive neurons")
                ax_activity.plot(times_sec, activity_history_neg, color="blue", lw=1.5, label="Negative neurons")
                ax_activity.plot(times_sec[-1], activity_history_pos[-1], "ro")
                ax_activity.plot(times_sec[-1], activity_history_neg[-1], "bo")
                ax_activity.set_ylabel("Active neurons", color="white")
                ax_activity.set_xlabel("Time (s)", color="white")
                ax_activity.tick_params(colors="white")
                ax_activity.set_title("Neuron activity over time", color="white")
                ax_activity.legend(facecolor="black", labelcolor="white")

                # ----------------------------
                # Update heatmap
                # ----------------------------
                curr_stim_gray = cv2.cvtColor(
                    cv2.resize(stim_data[stim_frame_idx], (mask_width, mask_height), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_RGB2GRAY
                ).astype(np.float32)

                if prev_stim_for_diff is None:
                    stim_diff_display = np.zeros_like(curr_stim_gray)
                else:
                    stim_diff_display = curr_stim_gray - prev_stim_for_diff

                stim_diff_masked = np.ma.masked_where(~eye_mask, stim_diff_display)
                prev_stim_for_diff = curr_stim_gray.copy()
                heatmap_img.set_data(stim_diff_masked)

                # ----------------------------
                # Capture frame
                # ----------------------------
                fig.canvas.draw()
                buffer = np.asarray(fig.canvas.buffer_rgba())
                frame = buffer[:, :, :3].copy()
                writer.append_data(frame)

                if t_idx % (frame_interval * 5) == 0:
                    gc.collect()

            writer.close()
            plt.close(fig)
            gc.collect()

            size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
            print(f"✓ MP4 created: {mp4_path} ({size_mb:.2f} MB, FPS={video_fps})")

        print("\n✅ All views rendered successfully.")
