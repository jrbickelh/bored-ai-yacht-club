<img width="851" height="336" alt="image" src="https://github.com/user-attachments/assets/3ee63b3c-0303-4920-98bb-94285b41dbbd" />


# Generative AI: DCGANs & Neural Style Transfer

This repository contains a collection of Jupyter Notebooks exploring generative deep learning techniques. It includes implementations of **Deep Convolutional Generative Adversarial Networks (DCGAN)** using TensorFlow/Keras and **Neural Style Transfer** using PyTorch.

## 📂 Project Structure

The repository is divided into two primary categories of generative models:

### 1. DCGAN (Deep Convolutional Generative Adversarial Networks)
*Framework: TensorFlow / Keras*

These notebooks explore the creation of generative models to synthesize new images based on training datasets.

* **`bored-ape-yacht-club-gan.ipynb`**: A DCGAN implementation trained to generate new "Bored Ape" style avatars.
    * **Goal:** mitigate "mode collapse" and generate distinct, convince ape avatars.
    * **Architecture:** Uses a Generator (Conv2DTranspose) and Discriminator (Conv2D) model structure.
* **`dcgan360.ipynb` & `dcgan361.ipynb`**: Additional GAN experiments and implementations.
    * Includes exploration of dataset preprocessing (e.g., Exoplanet data from Kepler Mission) and latent space visualization.
* **`dcgan1.gif`**: A visualization of the training progression, showing the generator learning to create shapes over time.

### 2. Artistic Neural Style Transfer
*Framework: PyTorch*

* **`artistic-neural-style-transfer-using-pytorch.ipynb`**: An implementation of the style transfer method outlined in the paper *[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)* by Gatys et al.
    * **Mechanism:** Uses a pre-trained **VGG19** network to separate and recombine the "content" of one image with the "style" of another (e.g., applying Van Gogh's style to a photograph).
    * **Key Features:** Custom Gram Matrix calculation for style loss and VGG feature extraction.

---

## 🖼️ Visuals & Results

### DCGAN Training Progression
![DCGAN Training](dcgan1.gif)

### Stable Diffusion
View generated collections on OpenSea: [https://opensea.io/jrbickelhaupt](https://opensea.io/jrbickelhaupt)

---

## 🛠️ Installation & Dependencies

To run these notebooks locally, you will need the following dependencies. It is recommended to use a virtual environment or Conda environment.

```bash
# Core DL Frameworks
pip install tensorflow torch torchvision

# Data Manipulation & Visualization
pip install numpy pandas matplotlib seaborn

# Image Processing
pip install opencv-python pillow imageio
