import numpy as np
import torch
from scipy.signal import butter, lfilter
from typing import List, Dict
import os
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class SICModelTorch:
    def __init__(self, subtype=None, device=None, matrix_dir="neuron_matrices_centered", t_step=10, rate=100, output_dir="./results/neuron_responses"):
        """
        Initialize the SIC model with PyTorch acceleration for batch processing.
        
        Args:
            subtype: Optional subtype specification
            device: Torch device to use. If None, will auto-select CUDA if available
            matrix_dir: Directory name for weight matrices (default: "neuron_matrices_centered")
        """
        self.NEURON_GRID = (41, 82)
        self.TIME_STEP = t_step
        self.L1_DECAY_TAU = 200
        self.L2_DECAY_TAU = 200
        self.L3_DECAY_TAU = 200
        self.LOW_PASS_CUTOFF = 10
        self.SAMPLING_RATE = rate
        self.b_lp, self.a_lp = butter(2, self.LOW_PASS_CUTOFF, btype='low', fs=self.SAMPLING_RATE)
        self.subtype = subtype
        self.matrix_dir = matrix_dir
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            print(f"Using device: {self.device}")
                
        # Create base output directory
        self.base_output_dir = output_dir
        os.makedirs(self.base_output_dir, exist_ok=True)

    def _create_output_dir(self):
        """
        Create a timestamped output directory for saving neuron responses.
        
        Returns:
            str: Full path to the created directory
        """
        timestamp = datetime.now().strftime("%Y%m")
        output_dir = os.path.join(self.base_output_dir, f"{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def load_weights(
        self,
        neuron_ids: List[int],
        matrix_dir: str = None,
        normalize: bool = False,
        center_to_origin: bool = False,
    ) -> Dict:
        """
        Load weights for multiple neurons from consolidated files.
        Combines left and right weights by horizontally flipping right weights,
        merges excitatory and inhibitory weights immediately,
        and optionally shifts weights so that the GLOBAL abs-max location across
        L1/L2/L3 is moved to the center (crop + zero padding, no wrapping).
        """

        import numpy as np
        import torch
        from typing import List, Dict

        neuron_types: List[str] = ['l1', 'l2', 'l3']
        neuron_ids = list(neuron_ids) if not isinstance(neuron_ids, list) else neuron_ids
        num_neurons = len(neuron_ids)

        current_matrix_dir = matrix_dir if matrix_dir is not None else self.matrix_dir

        weights = {
            'neuron_ids': neuron_ids,
            'indices': {nid: idx for idx, nid in enumerate(neuron_ids)},
            'matrix_dir': current_matrix_dir,
            'side': 'combined'
        }

        height, width = self.NEURON_GRID
        combined_weights = {}

        # --------------------------------------------------
        # PHASE 1: Load and combine left/right weights
        # --------------------------------------------------
        global_max_per_neuron = np.zeros(num_neurons, dtype=np.float32)

        for neuron_type in neuron_types:
            layer = neuron_type.upper()
            combined_weights[layer] = {
                'combined': np.zeros((num_neurons, height, width), dtype=np.float32),
                'found': np.zeros(num_neurons, dtype=bool)
            }

            for side in ['left', 'right']:
                path = f'./output/VFM/{neuron_type}_{side}.npz'

                try:
                    with np.load(path) as data:
                        target_ids = data['target_ids']
                        exc_data = data['exc']
                        inh_data = data['inh']

                        for out_idx, nid in enumerate(neuron_ids):
                            idx = np.where(np.asarray(target_ids) == nid)[0]
                            if idx.size == 0:
                                continue

                            idx = idx[0]
                            exc_weight = exc_data[idx].astype(np.float32)
                            inh_weight = inh_data[idx].astype(np.float32)
                            combined = exc_weight + inh_weight

                            global_max_per_neuron[out_idx] = max(
                                global_max_per_neuron[out_idx],
                                np.max(np.abs(exc_weight)),
                                np.max(np.abs(inh_weight))
                            )

                            if side == 'left':
                                combined_weights[layer]['combined'][out_idx, :, :width // 2] = combined
                            else:
                                combined_weights[layer]['combined'][out_idx, :, width // 2:] = np.fliplr(combined)

                            combined_weights[layer]['found'][out_idx] = True

                except FileNotFoundError:
                    print(f"Warning: Could not find weights file for {layer} {side} at {path}")

        # --------------------------------------------------
        # PHASE 2: Optional normalization
        # --------------------------------------------------
        eps = 1e-8

        for layer in ['L1', 'L2', 'L3']:
            batch = np.zeros((num_neurons, height, width), dtype=np.float32)

            for n in range(num_neurons):
                if not combined_weights[layer]['found'][n]:
                    continue

                max_val = global_max_per_neuron[n]

                if not normalize or max_val < eps:
                    batch[n] = combined_weights[layer]['combined'][n]
                    continue

                scale = max_val * (1 - 0.3 * np.log(max_val + eps))
                batch[n] = (
                    0.0 if scale < eps else combined_weights[layer]['combined'][n] / scale
                )

            weights[layer] = batch

        # --------------------------------------------------
        # PHASE 3: Shift by GLOBAL abs-max across L1/L2/L3
        # --------------------------------------------------
        if center_to_origin:
            target_y = height // 2
            target_x = width // 2

            for n in range(num_neurons):
                max_val = 0.0
                cx = cy = None

                # global abs max
                for layer in ['L1', 'L2', 'L3']:
                    w = weights[layer][n]
                    abs_w = np.abs(w)

                    if abs_w.max() > max_val:
                        max_val = abs_w.max()
                        cy, cx = np.unravel_index(np.argmax(abs_w), abs_w.shape)

                if max_val < eps:
                    continue

                shift_y = target_y - cy
                shift_x = target_x - cx

                for layer in ['L1', 'L2', 'L3']:
                    src = weights[layer][n]
                    dst = np.zeros_like(src)

                    y0_src = max(0, -shift_y)
                    y1_src = min(height, height - shift_y)
                    x0_src = max(0, -shift_x)
                    x1_src = min(width, width - shift_x)

                    y0_dst = max(0, shift_y)
                    y1_dst = min(height, height + shift_y)
                    x0_dst = max(0, shift_x)
                    x1_dst = min(width, width + shift_x)

                    dst[y0_dst:y1_dst, x0_dst:x1_dst] = src[y0_src:y1_src, x0_src:x1_src]
                    weights[layer][n] = dst

        # --------------------------------------------------
        # Convert to torch
        # --------------------------------------------------
        for layer in ['L1', 'L2', 'L3']:
            weights[layer] = torch.from_numpy(weights[layer]).to(self.device)

        return weights



    def _silu(self, x):
        """SiLU (Swish) activation function (torch or numpy)"""
        if isinstance(x, torch.Tensor):
            return torch.nn.functional.silu(x)
        else:
            return x / (1.0 + np.exp(-x))

    class NeuronLayer:
        """Single neuron layer (L1, L2, or L3) without batch dimension"""
        
        def __init__(self, layer_type, grid_shape, device):
            self.response = torch.zeros(grid_shape, dtype=torch.float32, device=device)
            self.layer_type = layer_type
            self.device = device
            
            if layer_type in ['L1', 'L3']:
                self.static_component = torch.zeros(grid_shape, dtype=torch.float32, device=device)
            else:
                self.static_component = None
            
            # Parameters for sigmoid response
            self.params = {
                'L1_B': {'a': -0.917, 'b': 1.992},
                'L1_D': {'a': -2.326, 'b': -5.377},
                'L2_B': {'a': -0.814, 'b': 1.950},
                'L2_D': {'a': -2.044, 'b': -3.623}
            }

        def _sigmoid(self, x, a, b):
            """Sigmoid function for layer response"""
            return a * x / (1 + torch.abs(b * x))

        def _contrast(self, current, last):
            """Compute contrast between current and last stimulus"""
            contrast = torch.zeros_like(current)
            mask = last != 0
            contrast[mask] = (current[mask] - last[mask]) / last[mask]
            contrast[(last == 0) & (current > 0)] = 10000
            return contrast

        def update(self, current, last, dt, tau):
            """
            Update layer response based on current and last stimulus.
            
            Args:
                current: Current stimulus (H, W)
                last: Last stimulus (H, W)
                dt: Time step
                tau: Decay time constant
            
            Returns:
                Updated response (H, W)
            """
            contrast = self._contrast(current, last)
            direction = torch.where(current > last, 
                                   torch.ones_like(current), 
                                   torch.zeros_like(current))  # 1 for 'B', 0 for 'D'
            delta = torch.zeros_like(contrast)
            
            if self.layer_type == 'L1':
                # Static component for L1
                self.static_component = 0.35 * torch.exp(-2.36 * current) + 0.08
                
                # Dynamic component
                for d, val in [('B', 1), ('D', 0)]:
                    mask = (direction == val)
                    p = self.params[f'L1_{d}']
                    delta[mask] = self._sigmoid(contrast[mask], p['a'], p['b'])
                
                self.response = self.response * torch.exp(torch.tensor(-dt / tau, device=self.device)) + delta
                return self.response+self.static_component
            
            elif self.layer_type == 'L2':
                # Only dynamic component for L2
                for d, val in [('B', 1), ('D', 0)]:
                    mask = (direction == val)
                    p = self.params[f'L2_{d}']
                    delta[mask] = self._sigmoid(contrast[mask], p['a'], p['b'])
                
                self.response = self.response * torch.exp(torch.tensor(-dt / tau, device=self.device)) + delta
                return self.response
            
            elif self.layer_type == 'L3':
                # Static target for L3
                target = 0.62 * torch.exp(-2.90 * current) + 0.04

                decay = torch.exp(torch.tensor(-dt / tau, device=self.device))
                self.response = self.response * decay + target * (1 - decay)

                return self.response


    def _can_index_by_neuron(self, v, n_neurons):
        if isinstance(v, np.ndarray):
            return v.ndim >= 1 and v.shape[0] == n_neurons
        if torch.is_tensor(v):
            return v.dim() >= 1 and v.shape[0] == n_neurons
        if isinstance(v, (list, tuple)):
            return len(v) == n_neurons
        return False

    def calculate_response_baseline(
        self,
        stim: np.ndarray,
        weights: Dict,
        save_results: bool = True,
        description: str = "",
        baseline_steps: int = 50,
        stim_name: str = '',
        responce_threshold: float = 0.001,
        downsample: bool = True
    ) -> np.ndarray:
        """
        Calculate baseline-subtracted responses with integer TIME_STEP sampling.

        Pipeline:
        Linear convolution → temporal LP filter → baseline subtraction → SiLU

        Returns:
            Baseline-subtracted, activated response array (N, T_new - baseline_steps)
        """
        import time
        import torch
        from scipy.signal import lfilter

        start_time = time.time()
        H, W, T_orig = stim.shape
        num_neurons = len(weights['neuron_ids'])
        
        # --- downsample AFTER LP ---
        if downsample:
            factor = max(1, self.TIME_STEP)
        else:
            factor = 1

        stim_resampled = stim[:, :, ::factor]
        n_steps = stim_resampled.shape[2]

        if n_steps <= baseline_steps:
            raise ValueError(
                f"Total stimulus length {n_steps} must be greater than baseline_steps {baseline_steps}"
            )

        # --- Initialize layers ---
        layers = {
            name: self.NeuronLayer(name, self.NEURON_GRID, self.device)
            for name in ['L1', 'L2', 'L3']
        }

        stim_tensor = torch.from_numpy(stim_resampled.astype(np.float32)).to(self.device)
        last_stim = stim_tensor[:, :, 0]

        # --- Allocate response tensor ---
        full_responses_complete = torch.zeros(
            (num_neurons, n_steps),
            dtype=torch.float32,
            device=self.device
        )
        # --- Time loop (no gradients to save memory) ---
        with torch.no_grad():
            for t in range(n_steps):
                current_stim = stim_tensor[:, :, t]
                layer_data = {}

                # Update each layer
                for name, layer in layers.items():
                    tau = getattr(self, f'{name}_DECAY_TAU')
                    layer_data[name] = layer.update(
                        current_stim, last_stim, self.TIME_STEP, tau
                    )

                # Linear weighted sum (exc + inh already combined)
                raw_response = torch.zeros(
                    num_neurons, dtype=torch.float32, device=self.device
                )

                for name in ['L1', 'L2', 'L3']:
                    raw_response += (layer_data[name].unsqueeze(0) * weights[name]).sum(dim=(1, 2))


                full_responses_complete[:, t] = raw_response
                last_stim = current_stim

                # Free intermediates
                del layer_data, raw_response
                torch.cuda.empty_cache()

        # --- Move to CPU ---
        full_responses_cpu = full_responses_complete.cpu().numpy()

        # --- Remove baseline steps (still linear domain) ---
        responses_truncated = full_responses_cpu[:, baseline_steps:]
        if responses_truncated.shape[1] == 0:
            raise ValueError(
                f"Truncation resulted in empty response array. "
                f"Check baseline_steps ({baseline_steps}) vs total steps ({n_steps})"
            )

        # --- Baseline subtraction (linear domain) ---
        baseline_values = responses_truncated[:, 0].copy()
        responses_baseline_sub = responses_truncated - baseline_values[:, np.newaxis]

        # ==================================================
        # ✅ NEW: ReLU BEFORE low-pass
        # ==================================================
        responses_relu = self._silu(responses_baseline_sub)

        # --- Low-pass filtering AFTER ReLU ---
        lp_responses = np.zeros_like(responses_relu)
        for i in range(num_neurons):
            lp_responses[i] = lfilter(self.b_lp, self.a_lp, responses_relu[i])

        # --- Optional activation ---
        final_responses = lp_responses

        # --- Save results if requested ---
        if save_results:
            max_abs_responses = np.max(np.abs(final_responses), axis=1)
            valid_neuron_idx = np.where(max_abs_responses >= responce_threshold)[0]

            if valid_neuron_idx.size > 0:
                final_valid = final_responses[valid_neuron_idx]
                baseline_values_valid = baseline_values[valid_neuron_idx]

                weights_valid = {}
                n_neurons_total = final_responses.shape[0]
                for k, val in weights.items():
                    if k == 'neuron_ids':
                        if isinstance(val, np.ndarray):
                            weights_valid[k] = val[valid_neuron_idx]
                        else:
                            weights_valid[k] = [val[i] for i in valid_neuron_idx]
                    else:
                        if isinstance(val, torch.Tensor) and val.shape[0] == n_neurons_total:
                            weights_valid[k] = val[valid_neuron_idx]
                        else:
                            weights_valid[k] = val

                self._save_responses_baseline(
                    final_valid,
                    responses_truncated[valid_neuron_idx],
                    baseline_values_valid,
                    weights_valid,
                    time.time() - start_time,
                    stim_resampled.shape,
                    description,
                    baseline_steps,
                    stim_name
                )

        return final_responses

    def _save_responses_baseline(self, responses: np.ndarray, raw_responses: np.ndarray, 
                                baseline_values: np.ndarray, weights: Dict, 
                                execution_time: float, original_stim_shape: tuple, 
                                description: str = "", baseline_steps: int = 300, stim_name: str = ''):
        """
        Save baseline-subtracted neuron responses to disk.
        
        Args:
            responses: Baseline-subtracted response array of shape (N, T-baseline_steps)
            raw_responses: Raw responses before baseline subtraction (N, T-baseline_steps)
            baseline_values: Baseline values subtracted from each neuron (N,)
            weights: Weight dictionary containing metadata
            execution_time: Total execution time in seconds
            original_stim_shape: Original shape of stimulus (H, W, T)
            description: Optional description
            baseline_steps: Number of baseline steps that were removed
        """
        # Create output directory
        output_dir = self._create_output_dir()
        
        # Create metadata with baseline information
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'num_neurons': len(weights['neuron_ids']),
            'original_stimulus_shape': original_stim_shape,
            'truncated_stimulus_length': original_stim_shape[2] - baseline_steps,
            'baseline_steps_removed': baseline_steps,
            'device': str(self.device),
            'description': description,
            'neuron_ids': weights['neuron_ids'].tolist() if isinstance(weights['neuron_ids'], np.ndarray) else weights['neuron_ids'],
            'response_shape': responses.shape,
            'resampled_length': responses.shape[1] + baseline_steps,
            'sampling_rate': self.SAMPLING_RATE,
            'matrix_dir': weights.get('matrix_dir', self.matrix_dir),
            'side': weights.get('side', 'right'),
            'model_parameters': {
                'L1_DECAY_TAU': self.L1_DECAY_TAU,
                'L2_DECAY_TAU': self.L2_DECAY_TAU,
                'L3_DECAY_TAU': self.L3_DECAY_TAU,
                'TIME_STEP': self.TIME_STEP,
                'NEURON_GRID': self.NEURON_GRID
            },
            'baseline_processing': {
                'method': 'subtract_first_step',
                'baseline_step_relative': 0,  # Step 0 after truncation
                'baseline_values': baseline_values.tolist()
            }
        }
        
        # Save as a single .npz file with both baseline-subtracted and raw responses
        save_path = os.path.join(output_dir, f"{stim_name}.npz")
        np.savez_compressed(
            save_path,
            responses=responses,
            baseline_values=baseline_values,
            neuron_ids=weights['neuron_ids'],
            metadata=metadata,
        )

    def calculate_layer_responses(self, stim: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate the average response of each layer (L1, L2, L3) over the neuron grid.
        This is useful for visualizing layer-level dynamics without specific neuron weights.
        
        Args:
            stim: Stimulus array of shape (H, W, T)
        
        Returns:
            Dictionary with keys 'L1', 'L2', 'L3' containing response arrays of shape (T,)
        """
        n_steps = stim.shape[2]
        
        # Initialize layers (no batch dimension needed)
        layers = {
            name: self.NeuronLayer(name, self.NEURON_GRID, self.device)
            for name in ['L1', 'L2', 'L3']
        }
        
        # Convert stimulus to torch
        stim_tensor = torch.from_numpy(stim.astype(np.float32)).to(self.device)
        last_stim = stim_tensor[:, :, 0]  # (H, W)
        
        # Storage
        layer_responses = {name: np.zeros(n_steps, dtype=np.float32) for name in layers}
        
        # Time loop
        for t in range(n_steps):
            current_stim = stim_tensor[:, :, t]  # (H, W)
            
            # Update each layer
            for name, layer in layers.items():
                tau = getattr(self, f'{name}_DECAY_TAU')
                layer_data = layer.update(current_stim, last_stim, self.TIME_STEP, tau)
                
                # Take mean over spatial dimensions
                layer_responses[name][t] = torch.mean(layer_data).cpu().item()
            
            last_stim = current_stim
        
        return layer_responses