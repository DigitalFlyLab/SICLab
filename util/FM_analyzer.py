import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import warnings

class FMAnalyzer:
    def __init__(self, matrix_npz_path, neuron_ids_path, neuron_types_path):
        """
        Initialize analyzer (supports only new .npz format)
        :param matrix_npz_path: Path to sparse matrix .npz file (must contain data/indices/indptr/shape)
        :param neuron_ids_path: Path to neuron ID list .npy file
        :param neuron_types_path: Path to neuron type CSV file
        """
        # Load sparse matrix
        try:
            loaded = np.load(matrix_npz_path)
            self.W = csr_matrix(
                (loaded['data'], loaded['indices'], loaded['indptr']),
                shape=loaded['shape']
            )
        except KeyError as e:
            available_keys = loaded.files if 'loaded' in locals() else []
            raise ValueError(
                f"NPZ file format error! Required keys: data, indices, indptr, shape\n"
                f"Found keys: {available_keys}"
            ) from e

        # Load neuron ID mapping
        self.neuron_ids = np.load(neuron_ids_path, allow_pickle=True)
        self.id_to_idx = {nid: idx for idx, nid in enumerate(self.neuron_ids)}
        self.idx_to_id = {idx: nid for idx, nid in enumerate(self.neuron_ids)}
        
        # Load neuron type data
        self.neuron_types_df = pd.read_csv(neuron_types_path)
        self._validate_neuron_types()

    def _validate_neuron_types(self):
        """Validate neuron type data integrity"""
        required_columns = {'root_id', 'type', 'side'}
        if not required_columns.issubset(self.neuron_types_df.columns):
            missing = required_columns - set(self.neuron_types_df.columns)
            raise ValueError(f"Neuron type file missing required columns: {missing}")

    def get_neuron_ids_by_type(self, neuron_type, side='right'):
        """
        Get neuron IDs of specified type
        :param neuron_type: Neuron type (e.g., L1/L2/L3)
        :param side: Brain side (left/right)
        :return: List of neuron IDs
        """
        df = self.neuron_types_df
        filtered = df[(df['type'] == neuron_type) & (df['side'] == side)]
        return filtered['root_id'].tolist()

    def set_blocked_types(self, blocked_types, side='right'):
        """
        Set neuron types to block (e.g., ['Tm3', 'T5b']),
        these nodes will not be allowed in path traversal.
        """
        df = self.neuron_types_df
        blocked = df[(df['type'].isin(blocked_types)) & (df['side'] == side)]['root_id'].tolist()
        blocked_idx = {self.id_to_idx[nid] for nid in blocked if nid in self.id_to_idx}
        self.blocked_idx = blocked_idx

    def _single_source_search(self, args):
        source_id, max_depth, min_weight = args
        
        pos_weights = np.zeros(len(self.neuron_ids), dtype=np.float64)
        neg_weights = np.zeros(len(self.neuron_ids), dtype=np.float64)
        if source_id not in self.id_to_idx:
            return pos_weights, neg_weights
        source_idx = self.id_to_idx[source_id]

        if hasattr(self, 'blocked_idx') and source_idx in self.blocked_idx:
            return pos_weights, neg_weights

        from collections import deque
        queue = deque()
        queue.append((source_idx, 1.0, 0))
        visited_edges = set()

        while queue:
            current_idx, current_weight, depth = queue.popleft()

            if depth > max_depth or abs(current_weight) < min_weight:
                continue

            if hasattr(self, 'blocked_idx') and current_idx in self.blocked_idx:
                continue

            if current_weight > 0:
                pos_weights[current_idx] += current_weight
            else:
                neg_weights[current_idx] += current_weight

            for j in range(self.W.indptr[current_idx], self.W.indptr[current_idx + 1]):
                post_idx = self.W.indices[j]
                conn_weight = self.W.data[j]
                edge_key = (current_idx, post_idx)
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                if hasattr(self, 'blocked_idx') and post_idx in self.blocked_idx:
                    continue

                new_weight = current_weight * conn_weight
                if abs(new_weight) >= min_weight:
                    queue.append((post_idx, new_weight, depth + 1))

        return pos_weights, neg_weights

    def compute_and_save_weights(self, neuron_types, side='right', blocked_types=None,
                                 max_depth=50, min_weight=1e-6, num_processes=None, chunk_size=10):
        """
        Add blocked_types parameter to block specified neuron types in path traversal.
        """
        if blocked_types:
            self.set_blocked_types(blocked_types, side)

        os.makedirs("./output", exist_ok=True)
        num_processes = num_processes or cpu_count()

        for neuron_type in neuron_types:
            print(f"\n{'='*50}\nProcessing {neuron_type} neurons (with block={blocked_types})\n{'='*50}")

            source_ids = self.get_neuron_ids_by_type(neuron_type, side)
            if not source_ids:
                print(f"Warning: No {neuron_type} neurons found")
                continue

            n_sources = len(source_ids)
            n_targets = len(self.neuron_ids)
            pos_matrix = np.zeros((n_sources, n_targets), dtype=np.float64)
            neg_matrix = np.zeros((n_sources, n_targets), dtype=np.float64)

            tasks = [(src_id, max_depth, min_weight) for src_id in source_ids]
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(self._single_source_search, tasks, chunksize=chunk_size),
                    total=n_sources,
                    desc=f"Processing {neuron_type}"
                ))
            for i, (pos_res, neg_res) in enumerate(results):
                pos_matrix[i] = pos_res
                neg_matrix[i] = neg_res

            blocked_tag = "_block" + "_".join(blocked_types) if blocked_types else ""
            base_path = f"./output/{neuron_type.lower()}_{side}{blocked_tag}"

            np.savez(
                f"{base_path}_excitatory.npz",
                source_ids=source_ids,
                target_ids=self.neuron_ids,
                weight_matrix=pos_matrix,
                metadata={
                    'neuron_type': neuron_type,
                    'side': side,
                    'max_depth': max_depth,
                    'min_weight': min_weight,
                    'blocked_types': blocked_types
                }
            )
            np.savez(
                f"{base_path}_inhibitory.npz",
                source_ids=source_ids,
                target_ids=self.neuron_ids,
                weight_matrix=neg_matrix,
                metadata={
                    'neuron_type': neuron_type,
                    'side': side,
                    'max_depth': max_depth,
                    'min_weight': min_weight,
                    'blocked_types': blocked_types
                }
            )
            print(f"Results saved to {base_path}_excitatory.npz and {base_path}_inhibitory.npz")

    def compute_from_specific_ids(self, source_ids,
                                blocked_types=None,
                                side='right',
                                max_depth=50,
                                min_weight=1e-6,
                                num_processes=None,
                                specific_type='sugar',
                                chunk_size=10):
        """
        Directly compute weights from specified neuron IDs
        """
        if blocked_types:
            self.set_blocked_types(blocked_types, side)

        os.makedirs("./output", exist_ok=True)
        num_processes = num_processes or cpu_count()

        source_ids = [nid for nid in source_ids if nid in self.id_to_idx]

        if not source_ids:
            print("No valid source IDs found in matrix.")
            return

        n_sources = len(source_ids)
        n_targets = len(self.neuron_ids)

        pos_matrix = np.zeros((n_sources, n_targets), dtype=np.float64)
        neg_matrix = np.zeros((n_sources, n_targets), dtype=np.float64)

        tasks = [(src_id, max_depth, min_weight) for src_id in source_ids]

        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(self._single_source_search, tasks, chunksize=chunk_size),
                total=n_sources,
                desc=f"Processing {specific_type} neurons"
            ))

        for i, (pos_res, neg_res) in enumerate(results):
            pos_matrix[i] = pos_res
            neg_matrix[i] = neg_res

        base_path = f"./output/{specific_type}"

        np.savez(
            f"{base_path}_excitatory.npz",
            source_ids=source_ids,
            target_ids=self.neuron_ids,
            weight_matrix=pos_matrix,
            metadata={
                'max_depth': max_depth,
                'min_weight': min_weight,
                'blocked_types': blocked_types
            }
        )

        np.savez(
            f"{base_path}_inhibitory.npz",
            source_ids=source_ids,
            target_ids=self.neuron_ids,
            weight_matrix=neg_matrix,
            metadata={
                'max_depth': max_depth,
                'min_weight': min_weight,
                'blocked_types': blocked_types
            }
        )

        print(f"Results saved to ./output/{specific_type}_excitatory.npz and {specific_type}_inhibitory.npz")

    def export_combined_summary_csv(
        self,
        side,
        neuron_types,
        pos_npz_paths,
        neg_npz_paths,
        cell_type_path="data/cell_type.txt",
        output_csv="combined_summary.csv",
        max_depth=100
    ):
        """
        Export a combined summary CSV including:
        - min_depth
        - max_downstream_weight (as post neuron)
        - max_upstream_weight (all upstream connections)
        - max_as_source_weight (as pre neuron connecting to targets)
        - upstream/downstream statistics (nonzero connections)
        - primary_type, additional_types
        """

        import pandas as pd
        import numpy as np
        from collections import deque

        n_neurons = len(self.neuron_ids)

        # ------------------------
        # Load cell type info
        # ------------------------
        cell_df = pd.read_csv(
            cell_type_path,
            header=None,
            names=["root_id", "primary_type", "additional_types"]
        )

        type_map = {
            str(row.root_id): (
                row.primary_type,
                row.additional_types if pd.notna(row.additional_types) else ""
            )
            for _, row in cell_df.iterrows()
        }

        # ------------------------
        # Compute max_downstream_weight from pos+neg matrices
        # ------------------------
        combined_matrix = None

        for pos_path, neg_path in zip(pos_npz_paths, neg_npz_paths):

            pos_data = np.load(pos_path, allow_pickle=True)
            neg_data = np.load(neg_path, allow_pickle=True)

            total_matrix = pos_data["weight_matrix"] + neg_data["weight_matrix"]

            if combined_matrix is None:
                combined_matrix = total_matrix
            else:
                combined_matrix = np.vstack((combined_matrix, total_matrix))

        max_downstream = np.max(np.abs(combined_matrix), axis=0)

        # ------------------------
        # Compute min_depth using BFS
        # ------------------------
        depths = np.full(n_neurons, np.inf)

        queue = deque()

        for neuron_type in neuron_types:

            source_ids = self.get_neuron_ids_by_type(neuron_type, side=side)

            for nid in source_ids:

                if nid in self.id_to_idx:

                    idx = self.id_to_idx[nid]

                    depths[idx] = 0
                    queue.append(idx)

        while queue:

            current = queue.popleft()

            if depths[current] >= max_depth:
                continue

            for j in range(self.W.indptr[current], self.W.indptr[current + 1]):

                post = self.W.indices[j]

                if depths[post] > depths[current] + 1:

                    depths[post] = depths[current] + 1
                    queue.append(post)

        # ------------------------
        # Upstream statistics (incoming edges)
        # ------------------------
        W_csc = self.W.tocsc()

        max_upstream = np.zeros(n_neurons)
        mean_upstream = np.zeros(n_neurons)
        count_upstream_nonzero = np.zeros(n_neurons)

        for idx in range(n_neurons):

            col_start = W_csc.indptr[idx]
            col_end = W_csc.indptr[idx + 1]

            if col_end > col_start:

                weights = np.abs(W_csc.data[col_start:col_end])

                max_upstream[idx] = np.max(weights)
                mean_upstream[idx] = np.mean(weights)
                count_upstream_nonzero[idx] = len(weights)

        # ------------------------
        # Downstream statistics (outgoing edges)
        # ------------------------
        max_as_source = np.zeros(n_neurons)
        mean_downstream = np.zeros(n_neurons)
        count_downstream_nonzero = np.zeros(n_neurons)

        for idx in range(n_neurons):

            row_start = self.W.indptr[idx]
            row_end = self.W.indptr[idx + 1]

            if row_end > row_start:

                weights = np.abs(self.W.data[row_start:row_end])

                max_as_source[idx] = np.max(weights)
                mean_downstream[idx] = np.mean(weights)
                count_downstream_nonzero[idx] = len(weights)

        # ------------------------
        # Build CSV
        # ------------------------
        rows = []

        for idx, nid in enumerate(self.neuron_ids):

            rid = str(nid)

            primary_type, additional_type = ("", "")

            if rid in type_map:
                primary_type, additional_type = type_map[rid]

            rows.append({
                "root_id": rid,
                "min_depth": int(depths[idx]) if depths[idx] != np.inf else -1,

                "max_FM_weight": float(max_downstream[idx]),
                "max_postsynaptic_weight": float(max_upstream[idx]),
                "max_presynaptic_weight": float(max_as_source[idx]),

                "mean_upstream_weight": float(mean_upstream[idx]),
                "count_upstream_nonzero": int(count_upstream_nonzero[idx]),

                "mean_downstream_weight": float(mean_downstream[idx]),
                "count_downstream_nonzero": int(count_downstream_nonzero[idx]),

                "primary_type": primary_type,
                "additional_types": additional_type
            })

        df_out = pd.DataFrame(rows)

        df_out.to_csv(output_csv, index=False)

        print(f"Combined summary saved to {output_csv}")