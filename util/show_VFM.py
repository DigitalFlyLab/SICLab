import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import issparse
from tqdm import tqdm
import os
from scipy.ndimage import shift

class VFMShow:
    def __init__(self, coord_path: str):
        """
        Initialize heatmap generator.

        Args:
            coord_path: Path to neuron coordinates CSV. Expected columns include 'root_id', 'x', 'y'.
        """
        # Load coordinate table and build quick lookup
        self.coord_df = pd.read_csv(coord_path)
        self._build_coord_mapping()

        # Load neuron type table for lookups used elsewhere
        self.neuron_types_df = pd.read_csv('../data/visual_neuron_types.txt')

        # Colormaps for excitatory (pos) and inhibitory (neg) displays
        self.exc_cmap = LinearSegmentedColormap.from_list('exc', ['#FFFFFF', '#FF0000'])
        self.inh_cmap = LinearSegmentedColormap.from_list('inh', ['#FFFFFF', '#0000FF'])

        # Fixed plotting grid parameters
        self.xlim = (-20, 20)
        self.ylim = (-20, 20)
        self.resolution = 1

    def hex_to_cartesian(self, p: float, q: float) -> tuple:
        """
        Convert hex coordinates (p,q) to Cartesian (x,y).
        The conversion follows a simple odd-row offset scheme.
        """
        x = 2 * p + 1 if q % 2 == 1 else 2 * p
        y = q / 2
        return x, y

    def _build_coord_mapping(self):
        """
        Build a mapping from root_id to Cartesian coordinates.
        Rows with missing or invalid coordinates are skipped.
        """
        self.coord_map = {}
        for _, row in self.coord_df.iterrows():
            try:
                x, y = self.hex_to_cartesian(float(row['x']), float(row['y']))
                self.coord_map[int(row['root_id'])] = (x, y)
            except (ValueError, KeyError):
                # Skip rows with invalid data
                continue

    def generate_all_matrices(self, neuron_types: list=['l1', 'l2', 'l3'], side: str="right", blocked_types: list=None):
        """
        PyTorch-accelerated version that rasterizes source->target matrices without centering.
        """
        import torch
        import os
        import numpy as np
        from scipy.sparse import issparse
        from tqdm import tqdm

        # Calculate grid dimensions
        x_bins = int((self.xlim[1] - self.xlim[0]) / self.resolution) + 1
        y_bins = int((self.ylim[1] - self.ylim[0]) / self.resolution) + 1

        print(f"\nProcessing all layers for {side} hemisphere (PyTorch accelerated)...")

        # Create output directory
        os.makedirs(f'./output/VFM', exist_ok=True)

        # Load layer data
        all_data = {}
        layers = [neuron_type.lower() for neuron_type in neuron_types]
        blocked_tag = "_block" + "_".join(blocked_types) if blocked_types else ""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        for layer in layers:
            pos_path = f'./output/{layer}_{side}{blocked_tag}_excitatory.npz'
            neg_path = f'./output/{layer}_{side}{blocked_tag}_inhibitory.npz'
            try:
                exc_data = np.load(pos_path, allow_pickle=True)
                inh_data = np.load(neg_path, allow_pickle=True)

                # Convert to PyTorch tensors
                all_data[layer] = {
                    'exc': torch.from_numpy(exc_data['weight_matrix'].toarray() if issparse(exc_data['weight_matrix']) else exc_data['weight_matrix']).float().to(device),
                    'inh': torch.from_numpy(inh_data['weight_matrix'].toarray() if issparse(inh_data['weight_matrix']) else inh_data['weight_matrix']).float().to(device),
                    'source_ids': exc_data['source_ids'],
                    'target_ids': exc_data['target_ids']
                }
            except FileNotFoundError:
                print(f"Warning: Missing data for {layer}_{side}")
                continue

        if not all_data:
            return

        target_ids = next(iter(all_data.values()))['target_ids']
        num_targets = len(target_ids)

        exc_coeff = 1.0
        inh_coeff = 1.0

        visual_metrices = {}

        for layer in layers:
            if layer not in all_data:
                continue

            print(f"\nProcessing layer {layer}...")

            src_ids = all_data[layer]['source_ids']
            exc_mat = all_data[layer]['exc']  # shape: (num_sources, num_targets)
            inh_mat = all_data[layer]['inh']

            # Build coordinate mapping for valid source neurons
            valid_indices = []
            x_coords = []
            y_coords = []

            for src_idx, src_id in enumerate(src_ids):
                if src_id in self.coord_map:
                    x, y = self.coord_map[src_id]
                    x_idx = int((x - self.xlim[0]) / self.resolution)
                    y_idx = int((y - self.ylim[0]) / self.resolution)

                    if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                        valid_indices.append(src_idx)
                        x_coords.append(x_idx)
                        y_coords.append(y_idx)

            if not valid_indices:
                continue

            # Convert to tensors
            valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
            x_coords = torch.tensor(x_coords, dtype=torch.long, device=device)
            y_coords = torch.tensor(y_coords, dtype=torch.long, device=device)

            # Extract weights for valid source neurons
            exc_weights = exc_mat[valid_indices] * exc_coeff
            inh_weights = inh_mat[valid_indices] * inh_coeff

            # Initialize output matrices: (num_targets, y_bins, x_bins)
            exc_matrices = torch.zeros((num_targets, y_bins, x_bins), dtype=torch.float32, device=device)
            inh_matrices = torch.zeros((num_targets, y_bins, x_bins), dtype=torch.float32, device=device)

            # Fill all target neuron grids
            for i, (y_idx, x_idx) in enumerate(zip(y_coords, x_coords)):
                exc_matrices[:, y_idx, x_idx] = exc_weights[i]  # Broadcast to all targets
                inh_matrices[:, y_idx, x_idx] = inh_weights[i]

            # Convert back to CPU numpy arrays
            visual_metrices[layer] = {
                'exc': exc_matrices.cpu().numpy(),
                'inh': inh_matrices.cpu().numpy(),
                'target_ids': target_ids
            }

        # Save output
        print("\nSaving output...")
        for layer in layers:
            if layer not in visual_metrices:
                continue

            np.savez_compressed(
                f'./output/VFM/{layer}_{side}.npz',
                exc=visual_metrices[layer]['exc'].astype(np.float16),
                inh=visual_metrices[layer]['inh'].astype(np.float16),
                target_ids=visual_metrices[layer]['target_ids']
            )

        print("All raster matrices saved (no centering).")

    def plot_single_neuron_all_layers(self, neuron_id: int, normalize: bool = False):

        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        neuron_ids = [neuron_id]
        layers_list = ['l1', 'l2', 'l3']

        mask_path = './output/combined_eye_mask_41x82.npz'
        mask_data = np.load(mask_path)
        mask = mask_data['mask'].astype(np.uint8)  # (H, W)
        height, width = mask.shape

        combined_weights = {}
        global_max = np.zeros(len(neuron_ids), dtype=np.float32)

        vmax_global = 0

        for neuron_type in layers_list:
            layer = neuron_type.upper()
            combined_weights[layer] = np.full(
                (len(neuron_ids), height, width),
                np.nan,
                dtype=np.float32
            )

            for out_idx, nid in enumerate(neuron_ids):
                concat_data = np.full((height, width), np.nan, dtype=np.float32)

                for side in ['left', 'right']:
                    weight_path = f'./output/VFM/{neuron_type}_{side}.npz'
                    if not os.path.exists(weight_path):
                        continue

                    data = np.load(weight_path)
                    target_ids = data['target_ids']
                    exc_data = data['exc']
                    inh_data = data['inh']

                    idx = np.where(target_ids == nid)[0]
                    if len(idx) == 0:
                        continue
                    idx = idx[0]

                    combined = (exc_data[idx] + inh_data[idx]).astype(np.float32)

                    if side == 'left':
                        concat_data[:, :width // 2] = combined
                    else:
                        concat_data[:, width // 2:] = np.fliplr(combined)

                concat_data_masked = np.full_like(concat_data, np.nan, dtype=np.float32)
                concat_data_masked[mask == 1] = concat_data[mask == 1]

                combined_weights[layer][out_idx] = concat_data_masked
                vmax_global = max(vmax_global, np.nanmax(np.abs(combined_weights[layer][out_idx])))

        eps = 1e-8

        if normalize:
            if vmax_global > eps:
                scale = vmax_global * (1-0.3*np.log(vmax_global + eps))
                if scale > eps:
                    for layer in [l.upper() for l in layers_list]:
                        for out_idx in range(len(neuron_ids)):
                            combined_weights[layer][out_idx] /= scale
                else:
                    for layer in [l.upper() for l in layers_list]:
                        for out_idx in range(len(neuron_ids)):
                            combined_weights[layer][out_idx] *= 0.0

        layers = ['L1', 'L2', 'L3']
        num_layers = len(layers)

        fig, axes = plt.subplots(
            num_layers, 1,
            figsize=(5, 2 * num_layers)
        )
        if num_layers == 1:
            axes = [axes]

        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad('white') 

        idx = 0
        for i, layer in enumerate(layers):
            ax = axes[i]
            data = combined_weights[layer][idx]

            vmax = np.nanmax(np.abs(data))
            if not np.isfinite(vmax):
                continue

            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            im = ax.imshow(
                data,
                cmap=cmap,
                norm=norm,
                origin='lower',
                aspect='auto'
            )

            ax.axvline(x=data.shape[1] // 2, color='black', linestyle='--')
            ax.set_xticks([])
            ax.set_yticks([])

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=12)  
            
        plt.tight_layout()
        plt.show()

    def plot_neuron_umap_from_profiles(
        self,
        tasks: dict,
        title: str = '',
        crop_size: int = 21,
        n_neighbors: int = 100,
        min_dist: float = 0.7,
        scatter_size: float = 10,
        random_state: int = 42,
        save_name: str = "neuron_umap",
        side_list: list = ['left', 'right', 'both'] 
    ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import umap
        from tqdm import tqdm
        import scienceplots
        import matplotlib.colors as mcolors

        plt.style.use(['science', 'nature', 'no-latex'])
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 14,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.linewidth": 1.5,
        })

        layers_list = ['l1', 'l2', 'l3']
        half = crop_size // 2
        save_dir = './output/VFM_umap'
        os.makedirs(save_dir, exist_ok=True)

        mask = np.load('./output/combined_eye_mask_41x82.npz')['mask'].astype(bool)
        H, W = mask.shape
        yy, xx = np.where(mask)

        # --------------------------
        # Load matrices
        # --------------------------
        matrices = {}
        for layer in layers_list:
            matrices[layer] = {}
            for s in ['left', 'right']:
                path = f'./output/VFM/{layer}_{s}.npz'
                if os.path.exists(path):
                    d = np.load(path)
                    matrices[layer][s] = {
                        'target_ids': d['target_ids'],
                        'w': (d['exc'] + d['inh']).astype(np.float32)
                    }
                else:
                    matrices[layer][s] = None

        # --------------------------
        # Collect features
        # --------------------------
        features_dict = {side: {layer: [] for layer in layers_list + ['all']}
                        for side in ['left', 'right', 'both']}
        labels_dict = {side: {layer: [] for layer in layers_list + ['all']}
                    for side in ['left', 'right', 'both']}
        visited = set()

        for (neuron_type, side), neuron_ids in tqdm(tasks.items(), desc='Extracting features'):
            for nid in neuron_ids:
                if nid in visited:
                    continue
                visited.add(nid)

                full_layers = []
                max_val = 0.0
                cx = cy = None

                for layer in layers_list:
                    full = np.zeros((H, W), dtype=np.float32)
                    for s in ['left', 'right']:
                        entry = matrices[layer][s]
                        if entry is None:
                            continue
                        ids = entry['target_ids']
                        idx = np.where(ids == nid)[0]
                        if idx.size == 0:
                            continue
                        w = entry['w'][idx[0]]
                        if s == 'left':
                            full[:, :W//2] = w
                        else:
                            full[:, W//2:] = np.fliplr(w)

                    full_layers.append(full)

                    abs_valid = np.abs(full)[mask]
                    if abs_valid.size == 0:
                        continue
                    if abs_valid.max() > max_val:
                        max_val = abs_valid.max()
                        idx_max = np.argmax(abs_valid)
                        cy = yy[idx_max]
                        cx = xx[idx_max]

                if max_val < 1e-12:
                    continue

                xs = xx - cx + half
                ys = yy - cy + half
                inside = (xs >= 0) & (xs < crop_size) & (ys >= 0) & (ys < crop_size)

                layer_feats = []
                for full in full_layers:
                    patch = np.zeros((crop_size, crop_size), dtype=np.float32)
                    patch[ys[inside], xs[inside]] = full[mask][inside]
                    layer_feats.append(patch.flatten())

                feat_concat = np.concatenate(layer_feats)

                for i, layer in enumerate(layers_list):
                    features_dict[side][layer].append(layer_feats[i])
                    labels_dict[side][layer].append(neuron_type)

                    features_dict['both'][layer].append(layer_feats[i])
                    labels_dict['both'][layer].append(neuron_type)

                features_dict[side]['all'].append(feat_concat)
                labels_dict[side]['all'].append(neuron_type)

                features_dict['both']['all'].append(feat_concat)
                labels_dict['both']['all'].append(neuron_type)

        embedding_dict = {}
        for side in side_list:
            embedding_dict[side] = {}
            for layer in layers_list + ['all']:
                feats = np.asarray(features_dict[side][layer])
                if feats.shape[0] == 0:
                    embedding_dict[side][layer] = None
                    continue
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=2,
                    random_state=random_state
                )
                embedding_dict[side][layer] = reducer.fit_transform(feats)

        nature_40 = [
            "#4C72B0","#55A868","#C44E52","#8172B3","#CCB974","#64B5CD",
            "#8C8C8C","#E17C05","#5DA5DA","#B276B2","#F17CB0","#60BD68",
            "#F15854","#4D4D4D","#B2912F","#499894","#E15759","#79706E",
            "#86BCB6","#FF9DA7","#9C755F","#3182bd","#31a354","#74c476","#a1d99b",
            "#756bb1","#9e9ac8","#bcbddc","#636363"
        ]

        # --------------------------
        # Plot
        # --------------------------
        valid_sides = [s for s in ['left', 'right', 'both'] if s in side_list]
        n_rows = len(valid_sides)
        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 5*n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        layer_names = ['L1', 'L2', 'L3', 'All']
        scatter_handles = []

        for row_idx, side in enumerate(valid_sides):
            for col_idx, layer_name in enumerate(layer_names):
                ax = axes[row_idx, col_idx]
                key = layer_name.lower() if layer_name != 'All' else 'all'
                emb = embedding_dict[side][key]
                lbls = labels_dict[side][key]

                if emb is None or len(lbls) == 0:
                    ax.axis('off')
                    continue

                unique_labels = sorted(set(lbls))
                num_types = len(unique_labels)

                if num_types <= len(nature_40):
                    colors_list = nature_40[:num_types]
                else:
                    n_colors = len(nature_40)
                    colors_list = []
                    for i in range(num_types):
                        base_color = nature_40[i % n_colors]
                        rgb = np.array(mcolors.to_rgb(base_color))
                        factor = 0.9 ** (i // 40)
                        colors_list.append(rgb * factor)

                for i, t in enumerate(unique_labels):
                    idx = [j for j, l in enumerate(lbls) if l == t]
                    sc = ax.scatter(
                        emb[idx, 0],
                        emb[idx, 1],
                        s=scatter_size,
                        alpha=0.3,
                        color=colors_list[i],
                        edgecolors='none',
                        linewidths=0,
                        label=t if row_idx == n_rows-1 and layer_name == 'All' else None
                    )
                    if row_idx == n_rows-1 and layer_name == 'All':
                        scatter_handles.append(sc)

                if row_idx == 0:
                    ax.set_title(layer_name)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')

        row_labels = {'left':'Left','right':'Right','both':'Both'}
        for i, side in enumerate(valid_sides):
            y = 1 - (i + 0.5)/n_rows
            fig.text(0.02, y, row_labels[side]+title, rotation=90, va='center', ha='center')

        from matplotlib.lines import Line2D
        labels = [s.get_label() for s in scatter_handles]
        colors = [s.get_facecolor()[0] for s in scatter_handles]
        proxy_handles = [Line2D([0], [0], marker='o', color='none', 
                                markerfacecolor=c, alpha=1, markersize=10)
                        for c in colors]
        n_per_col = 60
        N = len(labels)
        import math

        ncol = math.ceil(N / n_per_col)

        new_labels = []
        new_handles = []

        for r in range(n_per_col):
            for c in range(ncol):
                idx = c * n_per_col + r
                if idx < N:
                    new_labels.append(labels[idx])
                    new_handles.append(proxy_handles[idx])
        
        fig.legend(
            handles=new_handles,
            labels=new_labels,
            loc='center left',
            bbox_to_anchor=(1, 0.5),  
            bbox_transform=fig.transFigure,
            frameon=False,
            markerscale=2,
            ncol=ncol
        )

        plt.tight_layout(rect=[0.05, 0.05, 0.93, 0.95])
        plt.savefig(os.path.join(save_dir, f'{save_name}.pdf'), bbox_inches='tight')
        plt.show()

    def plot_neuron_umap_from_profiles(
        self,
        tasks: dict,
        title: str = '',
        crop_size: int = 21,
        n_neighbors: int = 100,
        min_dist: float = 0.7,
        scatter_size: float = 10,
        random_state: int = 42,
        save_name: str = "neuron_umap",
        side_list: list = ['left', 'right', 'both'] 
    ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import umap
        from tqdm import tqdm
        import scienceplots
        import matplotlib.colors as mcolors
        from sklearn.metrics import silhouette_score

        # -------------------------- Style --------------------------
        plt.style.use(['science', 'nature', 'no-latex'])
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 14,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.linewidth": 1.5,
        })

        layers_list = ['l1', 'l2', 'l3']
        half = crop_size // 2
        save_dir = './output/VFM_umap'
        os.makedirs(save_dir, exist_ok=True)

        mask = np.load('./output/combined_eye_mask_41x82.npz')['mask'].astype(bool)
        H, W = mask.shape
        yy, xx = np.where(mask)

        # -------------------------- Load matrices --------------------------
        matrices = {}
        for layer in layers_list:
            matrices[layer] = {}
            for s in ['left', 'right']:
                path = f'./output/VFM/{layer}_{s}.npz'
                if os.path.exists(path):
                    d = np.load(path)
                    matrices[layer][s] = {
                        'target_ids': d['target_ids'],
                        'w': (d['exc'] + d['inh']).astype(np.float32)
                    }
                else:
                    matrices[layer][s] = None

        # -------------------------- Collect features --------------------------
        features_dict = {side: {layer: [] for layer in layers_list + ['all']}
                        for side in ['left', 'right', 'both']}
        labels_dict = {side: {layer: [] for layer in layers_list + ['all']}
                    for side in ['left', 'right', 'both']}
        visited = set()

        for (neuron_type, side), neuron_ids in tqdm(tasks.items(), desc='Extracting features'):
            for nid in neuron_ids:
                if nid in visited:
                    continue
                visited.add(nid)

                full_layers = []
                max_val = 0.0
                cx = cy = None

                for layer in layers_list:
                    full = np.zeros((H, W), dtype=np.float32)
                    for s in ['left', 'right']:
                        entry = matrices[layer][s]
                        if entry is None:
                            continue
                        ids = entry['target_ids']
                        idx = np.where(ids == nid)[0]
                        if idx.size == 0:
                            continue
                        w = entry['w'][idx[0]]
                        if s == 'left':
                            full[:, :W//2] = w
                        else:
                            full[:, W//2:] = np.fliplr(w)

                    full_layers.append(full)

                    abs_valid = np.abs(full)[mask]
                    if abs_valid.size == 0:
                        continue
                    if abs_valid.max() > max_val:
                        max_val = abs_valid.max()
                        idx_max = np.argmax(abs_valid)
                        cy = yy[idx_max]
                        cx = xx[idx_max]

                if max_val < 1e-12:
                    continue

                xs = xx - cx + half
                ys = yy - cy + half
                inside = (xs >= 0) & (xs < crop_size) & (ys >= 0) & (ys < crop_size)

                layer_feats = []
                for full in full_layers:
                    patch = np.zeros((crop_size, crop_size), dtype=np.float32)
                    patch[ys[inside], xs[inside]] = full[mask][inside]
                    layer_feats.append(patch.flatten())

                feat_concat = np.concatenate(layer_feats)

                for i, layer in enumerate(layers_list):
                    features_dict[side][layer].append(layer_feats[i])
                    labels_dict[side][layer].append(neuron_type)

                    features_dict['both'][layer].append(layer_feats[i])
                    labels_dict['both'][layer].append(neuron_type)

                features_dict[side]['all'].append(feat_concat)
                labels_dict[side]['all'].append(neuron_type)

                features_dict['both']['all'].append(feat_concat)
                labels_dict['both']['all'].append(neuron_type)

        embedding_dict = {}
        silhouette_dict = {} 
        for side in side_list:
            embedding_dict[side] = {}
            silhouette_dict[side] = {}
            for layer in layers_list + ['all']:
                feats = np.asarray(features_dict[side][layer])
                lbls = np.asarray(labels_dict[side][layer])
                if feats.shape[0] == 0:
                    embedding_dict[side][layer] = None
                    silhouette_dict[side][layer] = None
                    continue
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=2,
                    random_state=random_state
                )
                embedding_dict[side][layer] = reducer.fit_transform(feats)
                try:
                    score = silhouette_score(feats, lbls)
                except Exception as e:
                    score = None
                silhouette_dict[side][layer] = score
                if score is not None:
                    print(f"Silhouette Score - side: {side}, layer: {layer} -> {score:.3f}")

        nature_40 = [
            "#4C72B0","#55A868","#C44E52","#8172B3","#CCB974","#64B5CD",
            "#8C8C8C","#E17C05","#5DA5DA","#B276B2","#F17CB0","#60BD68",
            "#F15854","#4D4D4D","#B2912F","#499894","#E15759","#79706E",
            "#86BCB6","#FF9DA7","#9C755F","#3182bd","#31a354","#74c476","#a1d99b",
            "#756bb1","#9e9ac8","#bcbddc","#636363"
        ]

        valid_sides = [s for s in ['left', 'right', 'both'] if s in side_list]
        n_rows = len(valid_sides)
        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 5*n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)  

        layer_names = ['L1', 'L2', 'L3', 'All']
        scatter_handles = []

        for row_idx, side in enumerate(valid_sides):
            for col_idx, layer_name in enumerate(layer_names):
                ax = axes[row_idx, col_idx]
                key = layer_name.lower() if layer_name != 'All' else 'all'
                emb = embedding_dict[side][key]
                lbls = labels_dict[side][key]
                score = silhouette_dict[side][key]  

                if emb is None or len(lbls) == 0:
                    ax.axis('off')
                    continue

                unique_labels = sorted(set(lbls))
                num_types = len(unique_labels)

                if num_types <= len(nature_40):
                    colors_list = nature_40[:num_types]
                else:
                    n_colors = len(nature_40)
                    colors_list = []
                    for i in range(num_types):
                        base_color = nature_40[i % n_colors]
                        rgb = np.array(mcolors.to_rgb(base_color))
                        factor = 0.9 ** (i // 40)
                        colors_list.append(rgb * factor)

                for i, t in enumerate(unique_labels):
                    idx = [j for j, l in enumerate(lbls) if l == t]
                    sc = ax.scatter(
                        emb[idx, 0],
                        emb[idx, 1],
                        s=scatter_size,
                        alpha=0.3,
                        color=colors_list[i],
                        edgecolors='none',
                        linewidths=0,
                        label=t if row_idx == n_rows-1 and layer_name == 'All' else None
                    )
                    if row_idx == n_rows-1 and layer_name == 'All':
                        scatter_handles.append(sc)

                if score is not None:
                    ax.set_title(f"{layer_name}\nSilhouette: {score:.3f}")
                else:
                    ax.set_title(f"{layer_name}\nSilhouette: N/A")

                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')

        row_labels = {'left':'Left','right':'Right','both':'Both'}
        for i, side in enumerate(valid_sides):
            y = 1 - (i + 0.5)/n_rows
            fig.text(0.02, y, row_labels[side]+title, rotation=90, va='center', ha='center')

        from matplotlib.lines import Line2D
        labels = [s.get_label() for s in scatter_handles]
        colors = [s.get_facecolor()[0] for s in scatter_handles]
        proxy_handles = [Line2D([0], [0], marker='o', color='none', 
                                markerfacecolor=c, alpha=1, markersize=10)
                        for c in colors]
        n_per_col = 60
        N = len(labels)
        import math
        ncol = math.ceil(N / n_per_col)

        new_labels = []
        new_handles = []
        for r in range(n_per_col):
            for c in range(ncol):
                idx = c * n_per_col + r
                if idx < N:
                    new_labels.append(labels[idx])
                    new_handles.append(proxy_handles[idx])
        
        fig.legend(
            handles=new_handles,
            labels=new_labels,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            bbox_transform=fig.transFigure,
            frameon=False,
            markerscale=2,
            ncol=ncol
        )

        plt.tight_layout(rect=[0.05, 0.05, 0.93, 0.95])
        plt.savefig(os.path.join(save_dir, f'{save_name}.pdf'), bbox_inches='tight')
        plt.show()

        return embedding_dict, silhouette_dict

    def plot_type_similarity_heatmap(
        self,
        tasks: dict,
        crop_size: int = 21,
        save_name: str = "type_similarity_heatmap_upper_triangular_labels_custom"
    ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import scienceplots

        # --------------------------
        # Style
        # --------------------------
        plt.style.use(['science', 'nature', 'no-latex'])
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.5,
        })

        layers_list = ['l1', 'l2', 'l3']
        half = crop_size // 2

        save_dir = './output/VFM_correlation'
        os.makedirs(save_dir, exist_ok=True)

        mask = np.load('./output/combined_eye_mask_41x82.npz')['mask'].astype(bool)
        H, W = mask.shape
        yy, xx = np.where(mask)

        matrices = {}
        for layer in layers_list:
            matrices[layer] = {}
            for s in ['left', 'right']:
                path = f'./output/VFM/{layer}_{s}.npz'
                if os.path.exists(path):
                    d = np.load(path)
                    matrices[layer][s] = {
                        'target_ids': d['target_ids'],
                        'w': (d['exc'] + d['inh']).astype(np.float32)
                    }
                else:
                    matrices[layer][s] = None

        features_dict = {side: {layer: {} for layer in layers_list + ['all']} for side in ['left', 'right', 'both']}
        visited = set()
        for (neuron_type, side), neuron_ids in tasks.items():
            for nid in neuron_ids:
                if nid in visited:
                    continue
                visited.add(nid)

                full_layers = []
                max_val = 0.0
                cx = cy = None

                for layer in layers_list:
                    full = np.zeros((H, W), dtype=np.float32)
                    for s in ['left', 'right']:
                        entry = matrices[layer][s]
                        if entry is None:
                            continue
                        ids = entry['target_ids']
                        idx = np.where(ids == nid)[0]
                        if idx.size == 0:
                            continue
                        w = entry['w'][idx[0]]
                        if s == 'left':
                            full[:, :W//2] = w
                        else:
                            full[:, W//2:] = np.fliplr(w)
                    full_layers.append(full)

                    abs_valid = np.abs(full)[mask]
                    if abs_valid.size == 0:
                        continue
                    if abs_valid.max() > max_val:
                        max_val = abs_valid.max()
                        idx_max = np.argmax(abs_valid)
                        cy = yy[idx_max]
                        cx = xx[idx_max]

                if max_val < 1e-12:
                    continue

                xs = xx - cx + half
                ys = yy - cy + half
                inside = (xs >= 0) & (xs < crop_size) & (ys >= 0) & (ys < crop_size)

                layer_feats = []
                for full in full_layers:
                    patch = np.zeros((crop_size, crop_size), dtype=np.float32)
                    patch[ys[inside], xs[inside]] = full[mask][inside]
                    layer_feats.append(patch.flatten())

                feat_concat = np.concatenate(layer_feats)
                for i, layer in enumerate(layers_list):
                    for s in [side, 'both']:
                        features_dict[s][layer].setdefault(neuron_type, []).append(layer_feats[i])
                for s in [side, 'both']:
                    features_dict[s]['all'].setdefault(neuron_type, []).append(feat_concat)

        fig, axes = plt.subplots(3, 1, figsize=(6, 18))
        sides = ['left', 'right', 'both']
        cmap = 'coolwarm'

        for ax, side in zip(axes, sides):
            layer_key = 'all'
            types = sorted(features_dict[side][layer_key].keys())
            n = len(types)
            if n == 0:
                ax.axis('off')
                continue

            mean_features = np.array([np.array(features_dict[side][layer_key][t]).mean(axis=0) for t in types])
            corr_matrix = np.corrcoef(mean_features)

            mask_triu_diag = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
            corr_upper = np.full_like(corr_matrix, np.nan)
            corr_upper[mask_triu_diag] = corr_matrix[mask_triu_diag]

            im = ax.imshow(corr_upper, cmap=cmap, vmin=-1, vmax=1, origin='upper')

            ax.axis('off')

            for i, t in enumerate(types):
                ax.text(i - 0.5, i, t, ha='right', va='center', fontsize=10) 

            for i, t in enumerate(types):
                ax.text(i, -0.6, t, ha='center', va='bottom', rotation=90, fontsize=10)  

            ax.set_xlim(-0.5, n-0.5)
            ax.set_ylim(n-0.5, -0.5)

            for spine in ['left','right','top','bottom']:
                ax.spines[spine].set_visible(False)

        cbar_ax = fig.add_axes([0.92, 0.4, 0.06, 0.3])
        fig.colorbar(im, cax=cbar_ax, label='Pearson r')

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(save_dir, f"{save_name}.pdf"), bbox_inches='tight')
        plt.show()