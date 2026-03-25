<p align="left">
  <img src="SIC_logo.jpeg" alt="SIC Logo" width="200"/>
</p>

The **Signal Infinite Cascade (SIC)** model is a computational framework designed to predict neural activity across the whole brain. The model receives sensory inputs directly from real-world environments and propagates signals through large-scale neural circuits to estimate the responses of individual neurons.

<p align="center">
  <img src="vision.gif" alt="SIC Vision Demo" width="500"/>
</p>
## 🧠 Overview

**Signal Infinite Cascade (SIC)** is a model designed to predict neural activity across the entire brain.  
The model can directly receive inputs from the real environment and generate predictions for the responses of neurons throughout the brain.

## ✨ Key Idea

The SIC model aims to simulate large-scale neural dynamics by:

- Receiving **direct sensory inputs from real-world environments**
- Modeling **signal propagation across neural networks**
- Predicting **responses of neurons across the whole brain**

## 📊 Model Comparison

| Feature | DMN | LIF | SIC |
|------|------|------|------|
| Neural response | Type-level | Neuron-level | Neuron-level |
| Response representation | Detailed | Simplified | Detailed |
| Circuit-level function | Task-specific | Multiple tasks | Multiple tasks |
| Additional parameters | Parameter learning | Membrane potential | Measured activity |
| Inhibitory interactions | Explicit modeling | Partially nonfunctional | Explicit modeling |
| Brain coverage | Visual only | Whole brain | Whole brain |
| Sensory modalities | Visual | Multiple modalities | Multiple modalities |

---

---

## 📦 Data Availability

The data used in this project can be obtained from the following resources:

- Dataset repository: https://zenodo.org/records/19213473  
- Detailed connectomics and neural data: [https://flywire.ai/](https://codex.flywire.ai/api/download?dataset=fafb)

If three-dimensional visualization of neurons is required, the corresponding **Neuron Skeletons** should also be downloaded from the FlyWire dataset(https://codex.flywire.ai/api/download?dataset=fafb#collapseskeleton_swc_files).
