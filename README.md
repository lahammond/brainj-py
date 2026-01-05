# brainj-py

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![Status](https://img.shields.io/badge/Status-Experimental-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**brainj-py** is a Python-based extension of the original [BrainJ](https://github.com/lahammond/BrainJ) software. It introduces advanced deep learning capabilities for automated restoration and U-Net segmentation of brain tissue sections, followed by activity mapping and statistical analysis.
Active development on this repository was paused in 2023. While this code has been used successfully in specific research projects, it is **not currently production-ready** for general use. Users should expect to perform some manual configuration and debugging.

---

## Relationship to BrainJ (Fiji)

This tool is designed to work in tandem with the original BrainJ plugin (ImageJ/Fiji). It is not yet a complete standalone replacement.

* **Original Repo:** [lahammond/BrainJ](https://github.com/lahammond/BrainJ)
* **Recommended Workflow:** For the most stable results, we recommend a hybrid workflow:

| Step | Recommended Tool | Notes |
| :--- | :--- | :--- |
| **1. Section Ordering** | [BrainJ (Original)](https://github.com/lahammond/BrainJ) | Java UI is optimized for manual sorting/verification. |
| **2. Preprocessing** | [BrainJ (Original)](https://github.com/lahammond/BrainJ) | Cropping, background subtraction, and formatting. |
| **3. 3D Atlas Registration** | [BrainJ (Original)](https://github.com/lahammond/BrainJ) | Alignment to reference atlases (e.g., ABA). |
| **4. Restoration & Segmentation** | **brainj-py** | Uses U-Net (Deep Learning) for superior segmentation. |
| **5. Activity Mapping** | **brainj-py** | Automated mapping of cellular markers (e.g., c-Fos). |
| **6. Statistical Analysis** | MATLAB | MATLAB scripts for two-timepoint statistics. |

---

## Key Features

* **Deep Learning Segmentation:** Implements U-Net architectures for robust tissue segmentation (e.g., separating tissue from background/artifacts), offering improved accuracy over coventional thresholding methods.
* **Activity Mapping:** Automated mapping of cellular activity markers onto registered brain space.
* **Longitudinal Analysis:** Specialized support for **two-timepoint analysis** using MATLAB integration to compare changes over time.

---

## Known Limitations & Future Work

The following features are currently missing or require further development to be fully integrated into the Python pipeline:

* End-to-End Preprocessing: Initial image handling is currently best performed in the Fiji version.
* Section-to-Section Registration: Automatic alignment between consecutive slices requires further development.
* Template Registration* Direct registration to the Allen Brain Atlas (CCFv3) is not seamlessly implemented in this Python version.

---

## Installation & Requirements

### Python Dependencies
Development environment used CUDA 11.2 and cuDNN 8.1

