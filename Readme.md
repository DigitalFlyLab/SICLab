<p align="center">
  <img src="SIC_logo.jpeg" alt="SIC Logo" width="200"/>
</p>

<h1 align="center">Signal Infinite Cascade (SIC)</h1>

<p align="center">
  <em>A computational model for predicting neuron-level activity across the whole brain.</em>
</p>

<p align="center">
  <img src="Vision.gif" alt="SIC Vision Demo" width="600"/>
</p>


## 🧠 Overview

**SIC** is a model designed to predict neural activity across the entire brain.  
The SIC framework provides a unified approach for modeling neural dynamics in the **Drosophila visual system across diverse visual stimulus classes**.

Using the SIC model, researchers can achieve:

- **Improved prediction of neural activity**  
  The SIC model captures neural responses to visual stimuli with high stability and accuracy.

- **Generalization across visual stimulus conditions**  
  The model predicts neural dynamics in circuits responding to ON/OFF flashes, moving edges, looming objects, and real-world video while representing both inhibitory and excitatory interactions.

- **Whole-brain, neuron-level simulation**  
  By incorporating partially measured neural activity as constraints, SIC can simulate responses of individual neurons across the entire brain under diverse conditions. This provides a comprehensive framework linking sensory inputs to whole-brain neural activity and enables new possibilities for digital life modeling.
  
## ✨ SIC Process

The SIC model aims to simulate large-scale neural dynamics by:

- Receiving **direct sensory inputs from real-world environments**
- Modeling **signal propagation across neural networks**
- Predicting **responses of neurons across the whole brain**

## 📊 Model Comparison

| Aspect | DMN | LIF | SIC |
|---|---|---|---|
| Modelling paradigm | Deep-learning-based dynamical model | Dynamical neural simulation | Static functional inference |
| Inference strategy | Iterative parameter optimization | Explicit membrane dynamics simulation | Query-driven selective inference |
| Dependence on training | Required | Not required | Not required |
| Response resolution | Neuron-type level | Neuron level | Neuron level |
| Response representation | Detailed but population-averaged | Simplified dynamical firing states | Detailed and physiologically constrained |
| Generalization | Task-specific | Task-flexible | Task-flexible |
| Brain coverage | Visual circuits only | Whole-brain | Whole-brain |

---

## 🛠 Environment Setup

To set up the environment and install all necessary dependencies for this project, use the provided **requirements.in** file.

Run the following command to install the required packages directly with **pip**:

```bash
pip install -r requirements.in
```

Optional environment management with **pip-tools** can compile **requirements.in** into **requirements.txt** and synchronize the environment.

```bash
pip install pip-tools
pip-compile requirements.in
pip-sync
```

Several simulation notebooks explicitly select a computing device:

- **cuda:0** selects the first GPU.
- **cuda:1** selects the second GPU.
- **cpu** runs the simulation without CUDA.

Before running a notebook, set its **device** argument to match the available hardware.
The large whole-brain simulations are substantially faster on a GPU.

---

## 📦 Data Availability

The archived project is available from [Zenodo](https://doi.org/10.5281/zenodo.21373953).
Because of upload size limits, **SICLab_without_sk_and_PRS.tar.gz** is distributed as six
parts named **SICLab_without_sk_and_PRS.tar.gz.part-00** through
**SICLab_without_sk_and_PRS.tar.gz.part-05**.

Download all six parts into the same directory, then reconstruct and extract the archive:

```bash
cat SICLab_without_sk_and_PRS.tar.gz.part-* > SICLab_without_sk_and_PRS.tar.gz
tar -xzf SICLab_without_sk_and_PRS.tar.gz
```

To keep the archive size manageable, the Zenodo package does not include:

- **data/sk_lod1_783_healed/**: FlyWire neuron skeleton files used for 3D visualization.
- **results/PRS/**: large generated simulation results, which can be reproduced by running the notebooks in **script/**.

The omitted neuron skeletons can be obtained from the
[FlyWire Codex download portal](https://codex.flywire.ai/api/download?data_product=skeleton_swc_files&dataset=fafb).
After downloading and extracting them, place the SWC files under
**data/sk_lod1_783_healed/**.
  
## 🧬 3D Visualization of Neurons

To perform three-dimensional visualization of neurons, follow these steps:

1. **Download neuron skeletons** from the FlyWire dataset: [FlyWire Skeleton SWC Files](https://codex.flywire.ai/api/download?data_product=skeleton_swc_files&dataset=fafb).  
   Extract the downloaded skeletons into **data/sk_lod1_783_healed/**. These SWC files are required for reconstructing 3D neuronal structures and generating the corresponding visualizations.

2. **Ensure dependencies are installed:**  
   The **navis** Python package and its dependencies are required for 3D reconstruction. This is automatically handled by the **Environment Setup** instructions using the **requirements.in** file.

---

## 🚀 Reproduction Workflow

To reproduce the results and figures, start Jupyter from the **script/** directory because
the notebooks use paths relative to that directory:

```bash
cd script
jupyter lab
```

Then run the notebooks in the following order.

### Phase 1: Connectome Matrix Generation (Run First)

- **get_FM.ipynb**: Processes the raw synaptic connection data and writes the functional mapping matrices and visualization summaries to **script/preprocess/**. The Zenodo archive includes these precomputed files, so rerun this notebook only when regenerating the matrices from the raw data.
- **get_FM_block.ipynb**: Generates the blocked functional mapping matrices used by the pathway-blocking experiments.

### Phase 2: Neural Dynamics Simulation (Run in Any Order)

Once the matrices are generated, you can run the following simulation scripts independently:

- **neuron_FRI.ipynb**: Simulates ON/OFF stimuli to calculate the **ON/OFF contrast-selectivity index (FRI)**.
- **neuron_DSI.ipynb**: Simulates moving edge stimuli for Direction Selectivity Index (DSI) calculation.
- **neuron_looming.ipynb**: Simulates looming dark disk stimuli.
- **neuron_looming_block.ipynb**: Simulates looming responses after pathway blocking. The current notebook is configured for **blockLPLC2**; update **block**, **RESPONSE_ROOT**, and **output_dir** consistently to generate the **blockLC4** and **blockLC4_LPLC2** conditions used by Figure 4.
- **neuron_RealWorld.ipynb** *(optional)*: Simulates responses to complex real-world video stimuli; it is not required by the figure notebooks listed below.

The raw population-response files produced in this phase are written under **results/PRS/**.
They are excluded from the Zenodo archive because of their size and must be regenerated
when rerunning analyses that consume the raw responses.

### Phase 3: Results Analysis and Figure Reproduction (Run Last)

Once the simulation data have been generated, run the following plotting notebooks to reproduce the figures in the paper.
