'''
Exercises - Homework:

1.1. Spin echo EPI  with zig-zag or non blipped trajectory and silent EPI   (evtl propeller EPI)
hint: exersice or any textbook for spin echo, http://mri.beckman.illinois.edu/interactive/topics/contents/fast_imaging/figures/zig.shtml
hint: Fig 2 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836719/, reuse radial reco
 reuse radial reco,   pp.make_extended_trapezoid

1.2. DREAM MRI
two readouts from one measurement: eg. FID, STE as two separate contrasts
hint: https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.24158


1.3. Diffusion weighted MRI
GRE DWI and spin echo DWI
hint: any textbook, https://blog.ismrm.org/2017/06/06/dwe-part-2/ 


1.4. B0 / T2* mapping  
hint: FLASH phase maps, any text book, both TE after one excitation


1.5. B1 mapping  
hint: Double angle https://onlinelibrary.wiley.com/doi/10.1002/mrm.20896 or (cosine fit)
reuse MP-FLASH


1.6a. T1 mapping  (invrec)
hint: any textbook, reuse MP-FLASH

1.6a. T1 mapping  (two flipangles)
hint: equation 3a and 3b of https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21969

1.7a. T2 mapping (Spin echo)
hint: any textbook, reuse SE or RARE

1.7a. T2 mapping (t2prep)
hint: https://www.kjronline.org/ViewImage.php?Type=F&aid=549810&id=F6&afn=68_KJR_18_1_113&fn=kjr-18-113-g006_0068KJR
reuse FLASH

1.8. FISP / PSIF 
hint: https://mriquestions.com/psif-vs-fisp.html
oder (fat suppression / water fat imaging)

1.9. HASTE sequence  + POCS recon
hint: textbook and https://mriquestions.com/partial-fourier.html

1.10. Spiral Trajectory -  EPI
any textbook and https://pulseq.github.io/, reuse radial reco
pp.make_arbitrary_grad.make_arbitrary_grad
pp.points_to_waveform
pp.traj_to_grad
grad_raster_time = 10e-6   % better!
seq = pp.Sequence(system=system)  % more acuarate
seq.calculate_kspace()
# Plot k-spaces
ktraj_adc, ktraj, t_excitation, t_refocusing, _ = seq.calculate_kspace()
plt.plot(ktraj.T)  # Plot the entire k-space trajectory
plt.figure()
plt.plot(ktraj[0], ktraj[1], 'b')  # 2D plot
plt.axis('equal')  # Enforce aspect ratio for the correct trajectory display
plt.plot(ktraj_adc[0], ktraj_adc[1], 'r.')
plt.show()

1.11  eight ball echo

1.12  full free FOV rotation translation

1.13   Segmentation  - segmented high res MPRAGE - segmented high res TSE

1.14   BURST MRI (advanced)

1.15 Ernst angle derivation - application to FID and FLASH

1.16 B0 distortion correction in EPI

1.17 Fastest Flash with trapezoidal gradients and ramp sampling

1.18 Slice selection - slice selection profile simulation and meas
https://github.com/pulseq/MR-Physics-with-Pulseq/tree/main/tutorials/02_rf_pulses
https://github.com/pulseq/MR-Physics-with-Pulseq/blob/main/slides/D1-1320-RF_Pulses_Tony_Stoecker.pdf

2.1. Code a small Bloch simulator an visualize spins




"""

Tuesday to Thursday 

17:00 Zoom meeting for questions

Excursion Henkestrasse 91 ALDI other side of the street


 
