# %% S0. SETUP env
import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import torch
import matplotlib.pyplot as plt

# makes the ex folder your working directory
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir(os.path.abspath(os.path.dirname(__file__)))

experiment_id = 'exA06_gradient_echo_phase_encoding'


# %% S1. SETUP sys

# choose the scanner limits
system = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6,
    adc_dead_time=20e-6
)


# %% S2. DEFINE the sequence
seq = pp.Sequence(system) 

# Define FOV and resolution
fov = 1000e-3
Nread = 64
Nphase = 256

slice_thickness = 8e-3  # slice

# Define rf events
rf1, _, _ = pp.make_sinc_pulse(
    flip_angle=5 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)
# rf1 = pp.make_block_pulse(flip_angle=90 * np.pi / 180, duration=1e-3, system=system)

# Define other gradients and ADC events
adc = pp.make_adc(num_samples=Nread, duration=2e-3, phase_offset=0 * np.pi / 180, system=system)

# ======
# CONSTRUCT SEQUENCE
# ======

for ii in range(-Nphase // 2, Nphase // 2):
    seq.add_block(pp.make_delay(1))
    seq.add_block(rf1)
    gp = pp.make_trapezoid(channel='y', area=ii, duration=2e-3, system=system)
    seq.add_block(gp)
    seq.add_block(adc)


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

if 0:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)

# Manipulate loaded data
    obj_p.T2dash[:] = 30e-3
    obj_p.D *= 0 
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
    # Store PD for comparison
    PD = obj_p.PD.squeeze()
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = mr0.CustomVoxelPhantom(
        pos=[[-0.25, -0.25, 0],[-0.25, 0, 0]],
        PD=[1.0,0.3],
        T1=[1.0],
        T2=[0.1],
        T2dash=[0.1],
        D=[0.0],
        voxel_size=0.1,
        voxel_shape="box"
    )
    # Store PD for comparison
    PD = obj_p.generate_PD_map()

obj_p.plot()
obj_p.size=torch.tensor([fov, fov, slice_thickness]) 
# Convert Phantom into simulation data
obj_p = obj_p.build()


# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot

# Read in the sequence 
seq0 = mr0.Sequence.import_file("out/external.seq")
 
# #seq0.plot_kspace_trajectory()
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
spectrum = torch.reshape((signal), (Nphase, Nread)).clone().t()
kspace = spectrum
plt.plot(torch.real(signal), label='real')
plt.plot(torch.imag(signal), label='imag')


# this adds ticks at the correct position szread
major_ticks = np.arange(0, Nphase * Nread, Nread)
ax = plt.gca()
ax.set_xticks(major_ticks)
ax.grid()

space = torch.zeros_like(spectrum)

# fftshift
spectrum = torch.fft.ifftshift(spectrum)
for ii in range(0, Nread):
    space[ii, :] = torch.fft.fft(spectrum[ii, :])



# fft

# fftshift
space = torch.fft.fftshift(space)


plt.subplot(312)
plt.title('FFT')

plt.plot(torch.abs(torch.t(space).flatten(0)), label='abs'); plt.legend()
ax = plt.gca()
ax.set_xticks(major_ticks)
ax.grid()

# %%  compare with original phantom obj_p.PD
plt.subplot(313)
plt.title('phantom projection')

import torch.nn.functional as F
phantom_resampled = F.interpolate(PD.permute(2, 0, 1).unsqueeze(0) , size=(Nphase, Nphase), mode='nearest').squeeze(0)  # Squeeze to remove batch dimension if it was added
phantom_resampled = np.roll(phantom_resampled, -Nphase // sz[1] // 2 + 1, 0)  # this is needed due to the oversampling of the phantom, szread>sz
phantom_resampled=phantom_resampled.squeeze()

if gp.channel == 'x':
    plt.plot(np.sum(phantom_resampled, axis=1).flatten('F'), label='x-projection'); plt.legend()
else:
    plt.plot(np.sum(phantom_resampled, axis=0).flatten('F'), label='y-projection')

plt.show()

plt.figure()
