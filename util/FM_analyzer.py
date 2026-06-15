import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import warnings
from numba import njit
@njit(cache=True)
def _single_source_search_numba_core(
    source_idx,
    max_depth,
    min_weight,
    n,
    W_indptr,
    W_indices,
    W_data,
    blocked_mask,
    has_blocked
):
    pos_weights = np.zeros(n, dtype=np.float64)
    neg_weights = np.zeros(n, dtype=np.float64)

    if has_blocked and blocked_mask[source_idx]:
        return pos_weights, neg_weights

    nodes = [source_idx]
    weights = [1.0]
    depths = [0]

    head = 0

    while head < len(nodes):

        current_idx = nodes[head]
        current_weight = weights[head]
        depth = depths[head]
        head += 1

        if depth > max_depth or abs(current_weight) < min_weight:
            continue

        if has_blocked and blocked_mask[current_idx]:
            continue

        if current_weight > 0:
            pos_weights[current_idx] += current_weight
        else:
            neg_weights[current_idx] += current_weight

        if depth >= max_depth:
            continue

        start = W_indptr[current_idx]
        end = W_indptr[current_idx + 1]

        next_depth = depth + 1

        for j in range(start, end):

            post_idx = W_indices[j]

            if has_blocked and blocked_mask[post_idx]:
                continue

            new_weight = current_weight * W_data[j]

            if abs(new_weight) >= min_weight:
                nodes.append(post_idx)
                weights.append(new_weight)
                depths.append(next_depth)

    return pos_weights, neg_weights
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
            self.W = self.W.tocsr()
            self.W.sum_duplicates()
            self.W.sort_indices()
            self.W.eliminate_zeros()
            
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

    def set_blocked_types(self, blocked_types):
        """
        Set neuron types to block.
        These nodes will not be allowed in path traversal.

        Block all neurons whose type is in blocked_types,
        regardless of side.
        """
        df = self.neuron_types_df

        # =========================================================
        # Block all sides
        # =========================================================
        blocked_df = df[
            df['type'].isin(blocked_types)
        ]

        blocked = blocked_df['root_id'].tolist()

        blocked_idx = {
            self.id_to_idx[nid]
            for nid in blocked
            if nid in self.id_to_idx
        }

        self.blocked_idx = blocked_idx

        # Numba-friendly boolean mask
        blocked_mask = np.zeros(len(self.neuron_ids), dtype=np.bool_)

        for idx in blocked_idx:
            blocked_mask[idx] = True

        self.blocked_mask = blocked_mask

        # =========================================================
        # Print block information
        # =========================================================
        print("\n" + "=" * 60)
        print("Blocked neuron information")
        print("=" * 60)
        print(f"Blocked types: {blocked_types}")
        print("Blocked side: all")
        print(f"Matched neurons in neuron_types_df: {len(blocked):,}")
        print(f"Matched neurons in connectome matrix: {len(blocked_idx):,}")
        print(f"Final blocked_mask count: {np.sum(blocked_mask):,}")

        print("\nBlocked count by type:")
        type_counts = blocked_df["type"].value_counts()

        for t in blocked_types:
            count_df = int(type_counts.get(t, 0))
            ids_t = blocked_df.loc[blocked_df["type"] == t, "root_id"].tolist()
            count_matrix = sum(nid in self.id_to_idx for nid in ids_t)

            print(
                f"  {t}: "
                f"{count_df:,} in neuron_types_df, "
                f"{count_matrix:,} in connectome matrix"
            )

        if "side" in blocked_df.columns:
            print("\nBlocked count by side:")
            side_counts = blocked_df["side"].value_counts()

            for s, c in side_counts.items():
                ids_s = blocked_df.loc[blocked_df["side"] == s, "root_id"].tolist()
                count_matrix_s = sum(nid in self.id_to_idx for nid in ids_s)

                print(
                    f"  {s}: "
                    f"{int(c):,} in neuron_types_df, "
                    f"{count_matrix_s:,} in connectome matrix"
                )

        print("=" * 60 + "\n")

    def _single_source_search(self, args):
        source_id, max_depth, min_weight = args

        n = len(self.neuron_ids)

        pos_weights = np.zeros(n, dtype=np.float64)
        neg_weights = np.zeros(n, dtype=np.float64)

        if source_id not in self.id_to_idx:
            return pos_weights, neg_weights

        source_idx = self.id_to_idx[source_id]

        # -----------------------------
        # CSR local bindings
        # -----------------------------
        W_indptr = self.W.indptr
        W_indices = self.W.indices
        W_data = self.W.data

        # -----------------------------
        # blocked mask
        # -----------------------------
        if hasattr(self, 'blocked_mask'):
            blocked_mask = self.blocked_mask
            has_blocked = True
        else:
            blocked_mask = np.zeros(n, dtype=np.bool_)
            has_blocked = False

        return _single_source_search_numba_core(
            source_idx,
            max_depth,
            min_weight,
            n,
            W_indptr,
            W_indices,
            W_data,
            blocked_mask,
            has_blocked
        )

    def compute_and_save_weights(self, neuron_types, side='right', blocked_types=None,
                                 max_depth=100, min_weight=1e-6, num_processes=None, chunk_size=10):
        if blocked_types:
            self.set_blocked_types(blocked_types)

        os.makedirs("./preprocess", exist_ok=True)
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
            base_path = f"./preprocess/{neuron_type.lower()}_{side}{blocked_tag}"

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
                                max_depth=100,
                                min_weight=1e-6,
                                num_processes=None,
                                specific_type='sugar',
                                chunk_size=10):
        """
        Directly compute weights from specified neuron IDs
        """
        if blocked_types:
            self.set_blocked_types(blocked_types)

        os.makedirs("./preprocess", exist_ok=True)
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

        base_path = f"./preprocess/{specific_type}"

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

        print(f"Results saved to ./preprocess/{specific_type}_excitatory.npz and {specific_type}_inhibitory.npz")

    def compute_weights_by_ids(self, source_ids, target_ids=None,
                            blocked_types=None, max_depth=100, min_weight=1e-6,
                            num_processes=None, chunk_size=10, save_prefix=None):
        """
        Compute weight matrices from explicit source IDs to target IDs.
        
        :param source_ids: List of neuron IDs to use as sources
        :param target_ids: Optional list of neuron IDs to restrict targets
        :param blocked_types: Optional list of neuron types to block during traversal
        :param max_depth: Max traversal depth
        :param min_weight: Minimum weight threshold
        :param num_processes: Number of parallel processes
        :param chunk_size: Chunk size for multiprocessing
        :param save_prefix: Optional prefix for saving results
        """
        if blocked_types:
            self.set_blocked_types(blocked_types)

        if target_ids is None:
            target_ids = self.neuron_ids

        target_idx_set = {self.id_to_idx[tid] for tid in target_ids if tid in self.id_to_idx}

        os.makedirs("./preprocess", exist_ok=True)
        num_processes = num_processes or cpu_count()
        n_sources = len(source_ids)
        n_targets = len(target_ids)

        pos_matrix = np.zeros((n_sources, n_targets), dtype=np.float64)
        neg_matrix = np.zeros((n_sources, n_targets), dtype=np.float64)

        tasks = [(src_id, max_depth, min_weight) for src_id in source_ids]
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(self._single_source_search, tasks, chunksize=chunk_size),
                total=n_sources,
                desc=f"Processing sources"
            ))

        for i, (pos_res, neg_res) in enumerate(results):
            # Only keep entries for the specified target IDs
            pos_matrix[i] = np.array([pos_res[self.id_to_idx[tid]] if tid in self.id_to_idx else 0
                                    for tid in target_ids])
            neg_matrix[i] = np.array([neg_res[self.id_to_idx[tid]] if tid in self.id_to_idx else 0
                                    for tid in target_ids])

        # Save results if prefix is given
        if save_prefix:
            blocked_tag = "_block" + "_".join(blocked_types) if blocked_types else ""
            base_path = f"./preprocess/{save_prefix}{blocked_tag}"

            np.savez(
                f"{base_path}_excitatory.npz",
                source_ids=source_ids,
                target_ids=target_ids,
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
                target_ids=target_ids,
                weight_matrix=neg_matrix,
                metadata={
                    'max_depth': max_depth,
                    'min_weight': min_weight,
                    'blocked_types': blocked_types
                }
            )
            print(f"Results saved to {base_path}_excitatory.npz and {base_path}_inhibitory.npz")

        return pos_matrix, neg_matrix
    
    def export_combined_summary_csv(
        self,
        side,
        neuron_types,
        pos_npz_paths,
        neg_npz_paths,
        cell_type_path="data/cell_type.txt",
        preprocess_csv="combined_summary.csv",
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

        df_out.to_csv(preprocess_csv, index=False)

        print(f"Combined summary saved to {preprocess_csv}")