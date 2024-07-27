#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 09:50:45 2024

@author: zaissmz
"""
import numpy as np
import matplotlib.pyplot as plt

Nx=50
Ny=Nx

dk=1/Nx

# Create linspace vectors for X and Y
x = np.linspace(0, Nx - 1, Nx) - Nx / 2
y = np.linspace(0, Ny - 1, Ny) - Ny / 2
r = np.array(np.meshgrid(x, y, indexing='ij'))

plt.figure(); plt.subplot(121), plt.imshow(r[0,:,:]), plt.title('rx')
plt.subplot(122), plt.imshow(r[1,:,:]), plt.title('ry')

k=np.array([-50,0])
# np.exp(-1j*k*r)

def kxy_wave(k,r):
     return np.exp(-1j * np.tensordot(k, r, axes=(0, 0)))
 
# img=kxy_wave(dk*np.array([0,1]),r)
# plt.figure();
# plt.imshow(np.real(img))

# Set up the figure for subplots
fig, axs = plt.subplots(9, 9, figsize=(9, 6))
idx= [0,1,2,3,5,10,20,30,50]
idx= [-50,-10,-20, -5,0,5,10,20,50]
for i in range(9):
    for j in range(9):
        k = dk * np.array([idx[i],idx[j]])
        img = kxy_wave(k, r)
        ax = axs[i, j]
        ax.imshow(np.real(img)); 
        ax.set_ylabel(f"({idx[i]},{idx[j]})")
        # ax.axis('off')  # Hide axes for clarity
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()

# %% 
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# Load Shepp-Logan phantom and resize to match your grid
phantom = shepp_logan_phantom()
phantom_resized = resize(phantom, (Nx, Ny), mode='reflect', anti_aliasing=True)

# Fourier Transform of the Phantom to get k-space representation
k_space = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phantom_resized)))

# Display the Phantom and its k-space magnitude
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(phantom, cmap='gray'), plt.title('Shepp-Logan Phantom')
plt.subplot(122), plt.imshow(np.log(np.abs(k_space)), cmap='gray'), plt.title('k-space Magnitude')
plt.show()

# %% 
plt.figure()
idx= [0,-1,1,-2,2,-3,3,-5,5,-6,6,-7,7,-8,8] 

# Create an empty k-space
recon_k_space = np.zeros_like(k_space)

recon_image_N = np.zeros([Nx,Ny,len(idx)*len(idx)])
N=0
# Reconstruction loop
for i in range(len(idx)):
    for j in range(len(idx)):
        recon_k_space[idx[i]+25, idx[j]+25] = k_space[idx[i]+25,idx[j]+25]
        
        recon_k_space_N = np.zeros_like(k_space)
        recon_k_space_N[idx[i]+25, idx[j]+25] = k_space[idx[i]+25,idx[j]+25]
        recon_image_N[:,:,N] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(recon_k_space_N)))
        N+=1       
# Inverse Fourier Transform to get back to spatial domain
recon_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(recon_k_space)))

plt.subplot(231), plt.imshow(phantom_resized, cmap='gray'), plt.title('Shepp-Logan Phantom')
plt.subplot(232), plt.imshow(np.abs(recon_image), cmap='gray')
plt.subplot(233), plt.imshow(np.log(np.abs(k_space)), cmap='gray')

IJ=100
plt.subplot(2,3,4), plt.imshow(np.abs(recon_image-recon_image_N[:,:,IJ]), cmap='gray',vmin=0,vmax=1)

plt.tight_layout()
plt.show()
