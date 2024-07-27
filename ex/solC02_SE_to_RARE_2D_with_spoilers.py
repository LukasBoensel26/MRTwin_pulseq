# %% S0. SETUP env
import MRzeroCore as mr0
import pypulseq as pp

import numpy as np
import torch
from matplotlib import pyplot as plt

# makes the ex folder your working directory
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir(os.path.abspath(os.path.dirname(__file__)))

experiment_id = 'exB02_SE_to_RARE_2D'


# %% S1. SETUP sys

# choose the scanner limits
system = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6,
    adc_dead_time=20e-6, grad_raster_time=50 * 10e-6
)


# %% S2. DEFINE the sequence
seq = pp.Sequence()

# Define FOV and resolution
fov = 1000e-3
slice_thickness = 8e-3

Nread = 64    # frequency encoding steps/samples
Nphase = 64    # phase encoding steps/samples

# Define rf events
rf1, _, _ = pp.make_sinc_pulse(
    flip_angle=30 * np.pi / 180, phase_offset=90 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)
rf2, _, _ = pp.make_sinc_pulse(
    flip_angle=60 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)

# Define other gradients and ADC events
gx = pp.make_trapezoid(channel='x', flat_area=Nread,flat_time=2e-3, system=system)
adc = pp.make_adc(num_samples=Nread, duration=2e-3, phase_offset=90 * np.pi/180, delay=gx.rise_time, system=system)
gx_pre0 = pp.make_trapezoid(channel='x', area=+(1.0 + 3.0) * gx.area / 2, duration=1e-3, system=system)
gx_prewinder = pp.make_trapezoid(channel='x', area=+3.0 * gx.area / 2, duration=1e-3, system=system)

# calculate TE and delays
ct=pp.calc_rf_center(rf2)    # rf center time returns time and index of the center of the pulse
ct[0]                       # this is the rf center time

TE=  6e-3  # the echo time we want, defines the delays we need, min TE~=2ms
delayTE_1= pp.make_delay(TE/2 - pp.calc_duration(rf1))  # the rf pulses take time, which we need to subtract

delayTE_2= pp.make_delay(TE/2 - ct[0]- rf2.ringdown_time-pp.calc_duration(gx)/2) # half rf and half adc/gx time need to be subtracted, so echo is at adc center

#pp.make_delay(0.0041 - pp.calc_duration(rf1) - rf2.delay - rf2.t[-1] / 2 + rf2.ringdown_time / 2)

# ======
# CONSTRUCT SEQUENCE
# ======

# FLAIR
#rf_prep = pp.make_block_pulse(flip_angle=180 * np.pi / 180, duration=1e-3, system=system)
# seq.add_block(rf_prep)
# seq.add_block(make_delay(2.7))
# seq.add_block(gx_pre0)

seq.add_block(rf1)

seq.add_block(gx_pre0,delayTE_1) # only valid if delayTE_1 longer than gx_pre0
if pp.calc_duration(gx_pre0)>pp.calc_duration(delayTE_1): raise Exception("below minTE")

for ii in range(-Nphase // 2, Nphase // 2):  # e.g. -64:63
    seq.add_block(rf2)
    gp = pp.make_trapezoid(channel='y', area=ii, duration=1e-3, system=system)
    gp_ = pp.make_trapezoid(channel='y', area=-ii, duration=1e-3, system=system)
    seq.add_block(gx_prewinder, gp, delayTE_2)  
    seq.add_block(adc, gx)
    seq.add_block(gx_prewinder, gp_,delayTE_2)


# Trajectory calculation and plotting
ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
plt.figure()
plt.plot(time_axis, ktraj.T)  # Plot the entire k-space trajectory
plt.plot(t_adc, ktraj_adc[0], '.')  # Plot sampling points on the kx-axis
plt.figure()
plt.plot(ktraj[0], ktraj[1], 'b')  # 2D plot
plt.axis('equal')  # Enforce aspect ratio for the correct trajectory display
plt.plot(ktraj_adc[0], ktraj_adc[1], 'r.')  # Plot  sampling points
plt.show()


# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc, t_adc = mr0.util.pulseq_plot(seq, clear=False, figid=(11,12))

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id + '.seq')


# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
sz = [64, 64]

if 1:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)

# Manipulate loaded data
    obj_p.T2dash[:] = 30e-3
    obj_p.D *= 0 
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
    obj_p.B1 *= 1  # alter the B1 inhomogeneity
    # Store PD for comparison
    PD = obj_p.PD.squeeze()
    B0 = obj_p.B0.squeeze()
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = mr0.CustomVoxelPhantom(
        pos=[[-0.4, -0.4, 0], [-0.4, -0.2, 0], [-0.3, -0.2, 0], [-0.2, -0.2, 0], [-0.1, -0.2, 0]],
        PD=[1.0, 1.0, 0.5, 0.5, 0.5],
        T1=1.0,
        T2=0.1,
        T2dash=0.1,
        D=0.0,
        B0=0,
        voxel_size=0.1,
        voxel_shape="box"
    )
    # Store PD for comparison
    PD = obj_p.generate_PD_map()
    B0 = torch.zeros_like(PD)

obj_p.plot()
obj_p.size=torch.tensor([fov, fov, slice_thickness]) 
# Convert Phantom into simulation data
obj_p = obj_p.build()


# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot

# Read in the sequence 
seq0 = mr0.Sequence.import_file("out/external.seq")
 
#seq0.plot_kspace_trajectory()
# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, obj_p)

# PLOT sequence with signal in the ADC subplot
plt.close(11);plt.close(12)
sp_adc, t_adc = mr0.util.pulseq_plot(seq, clear=False, signal=signal.numpy())
 
 


# %% S6: MR IMAGE RECON of signal ::: #####################################
fig = plt.figure()  # fig.clf()
plt.subplot(411)
plt.title('ADC signal')
kspace = torch.reshape((signal), (Nphase, Nread)).clone().t()
plt.plot(torch.real(signal), label='real')
plt.plot(torch.imag(signal), label='imag')


# this adds ticks at the correct position szread
major_ticks = np.arange(0, Nphase * Nread, Nread)
ax = plt.gca()
ax.set_xticks(major_ticks)
ax.grid()

# fftshift
spectrum = torch.fft.fftshift(kspace)
# FFT
space = torch.fft.fft2(spectrum)
# fftshift
space = torch.fft.fftshift(space)


plt.subplot(345)
plt.title('k-space')
mr0.util.imshow(np.abs(kspace.numpy()))
plt.subplot(349)
plt.title('k-space_r')
mr0.util.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346)
plt.title('FFT-magnitude')
mr0.util.imshow(np.abs(space.numpy()))
plt.colorbar()
plt.subplot(3, 4, 10)
plt.title('FFT-phase')
mr0.util.imshow(np.angle(space.numpy()), vmin=-np.pi, vmax=np.pi)
plt.colorbar()

# % compare with original phantom obj_p.PD
plt.subplot(348)
plt.title('phantom PD')
mr0.util.imshow(PD)
plt.subplot(3, 4, 12)
plt.title('phantom B0')
mr0.util.imshow(B0)
