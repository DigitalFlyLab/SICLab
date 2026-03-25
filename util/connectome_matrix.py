import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed
from tqdm import tqdm

class ConnectomeMatrixExporter:
    def __init__(self, connections_path, n_jobs=-1, results_dir="results"):
        self.connections_path = connections_path
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.results_dir = results_dir
        self.W = None
        self.neuron_ids = []
        self.id_to_idx = {}

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self):
        conn_df = pd.read_csv(self.connections_path, 
                              usecols=['pre_root_id', 'post_root_id', 'neuropil', 
                                       'syn_count', 'nt_type'])
        
        conn_df = conn_df.groupby(['pre_root_id', 'post_root_id', 'nt_type'], 
                                  as_index=False)['syn_count'].sum()
        
        input_synapses_df = conn_df.groupby('post_root_id')['syn_count'].sum().reset_index()
        input_synapses = input_synapses_df.set_index('post_root_id')['syn_count'].to_dict()
        return conn_df, input_synapses

    def build_connectome_matrix(self):
        conn_df, input_synapses = self.load_data()
        
        # Build ID map (ensure neuron_ids are sorted)
        all_ids = set(conn_df['pre_root_id']).union(set(conn_df['post_root_id']))
        self.neuron_ids = sorted(all_ids)
        self.id_to_idx = {n: i for i, n in enumerate(self.neuron_ids)}
        
        # Parallel computation of raw weights
        print("Computing connectome weights...")
        self.W = self._build_weights_parallel(conn_df, input_synapses)
        return self.W

    def _build_weights_parallel(self, conn_df, input_synapses):
        def process_chunk(chunk):
            rows, cols, data = [], [], []
            for _, row in chunk.iterrows():
                pre_idx = self.id_to_idx[row['pre_root_id']]
                post_idx = self.id_to_idx[row['post_root_id']]
                sign = -1 if row['nt_type'] in {"GABA", "GLUT"} else 1
                post_input_syn = input_synapses.get(row['post_root_id'], 1)  # avoid KeyError
                weight = row['syn_count'] * sign / post_input_syn
                rows.append(pre_idx)
                cols.append(post_idx)
                data.append(weight)
            return (rows, cols, data)

        # NOTE:
        # np.array_split(DataFrame, ...) 会把 DataFrame 转成 ndarray（丢失列名），
        # 从而导致 chunk 没有 iterrows()。
        # 这里按索引分块，确保传给 process_chunk 的仍然是 DataFrame。
        chunk_indices = np.array_split(conn_df.index, self.n_jobs * 4)
        chunks = [conn_df.loc[idx] for idx in chunk_indices]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_chunk)(chunk) for chunk in tqdm(chunks)
        )
        
        # Merge results
        all_rows, all_cols, all_data = [], [], []
        for r, c, d in results:
            all_rows.extend(r)
            all_cols.extend(c)
            all_data.extend(d)
            
        return coo_matrix((all_data, (all_rows, all_cols)), 
                          shape=(len(self.neuron_ids), len(self.neuron_ids))).tocsr()

    def save_connectome_matrix_npy(self):
        """Save connectome matrix into three separate files: sparse_matrix.npz + neuron_ids.npy + text file"""
        if self.W is None:
            self.build_connectome_matrix()
            
        # 1. Save sparse matrix (.npz)
        sparse_filename = os.path.join(self.results_dir, f"connectome_matrix.npz")
        np.savez(sparse_filename, 
                 data=self.W.data, 
                 indices=self.W.indices, 
                 indptr=self.W.indptr,
                 shape=self.W.shape)
        
        # 2. Save neuron IDs (.npy)
        ids_filename = os.path.join(self.results_dir, f"neuron_ids.npy")
        np.save(ids_filename, self.neuron_ids)
        
        # 3. Save text file (.txt)
        txt_filename = os.path.join(self.results_dir, f"connectome_matrix.txt")
        self._save_as_txt(txt_filename)
        
        print(f"\nSparse connectome matrix saved to: {sparse_filename}")
        print(f"Neuron IDs saved to: {ids_filename}")
        print(f"Text format saved to: {txt_filename}")

    def _save_as_txt(self, filename, max_entries=None):
        W_coo = self.W.tocoo()
        total_entries = len(W_coo.data)
        
        if max_entries is not None and max_entries < total_entries:
            print(f"Only saving top {max_entries:,} strongest connections (of {total_entries:,})")
            idx = np.argsort(-np.abs(W_coo.data))[:max_entries]
            rows = W_coo.row[idx]
            cols = W_coo.col[idx]
            data = W_coo.data[idx]
        else:
            rows = W_coo.row
            cols = W_coo.col
            data = W_coo.data
        
        # Write into file
        with open(filename, 'w') as f:
            f.write(f"# Connectome Matrix (shape: {self.W.shape[0]}x{self.W.shape[1]})\n")
            f.write(f"# Total neurons: {len(self.neuron_ids)}\n")
            f.write(f"# Non-zero entries: {len(data):,}\n")
            f.write("# Format: pre_neuron_id post_neuron_id weight_value\n\n")
            
            for r, c, v in zip(rows, cols, data):
                pre_id = self.neuron_ids[r]
                post_id = self.neuron_ids[c]
                f.write(f"{pre_id} {post_id} {v:.6g}\n")