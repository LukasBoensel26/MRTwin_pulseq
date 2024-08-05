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

experiment_id = 'SpinEcho'


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
Nread = 128 # enlarging Nread makes also the gradients take longer --> thus delta k and FoV is fixed but resolution gets better
Nphase = 128
slice_thickness = 8e-3  # slice

# Define rf events
rf0, _, _ = pp.make_sinc_pulse(
    phase_offset=90 * np.pi / 180, # to make the spin echo positive and real
    flip_angle=90 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)

# inversion pulse
rf1, _, _ = pp.make_sinc_pulse(
    flip_angle=180 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)

# define timing parameters of the sequence
del_90_180 = 5e-3

# define further sequence parameters
num_echos = 10

# Define gradients and ADC events
g_phase = pp.make_trapezoid(channel='y', area=-Nphase/2, duration=1e-3, system=system)
g_phase_revert = pp.make_trapezoid(channel='y', area=-g_phase.area, duration=1e-3, system=system)

# del_90_180 + pp.calc_duration(rf0) = pp.calc_rf_center(rf0)[0] + rf0.delay + g_phase.flat_time + g_phase.rise_time + g_phase.fall_time + 0.5 * adc_duration
# --> solve for adc_duration
adc_duration = 2 * (del_90_180 + pp.calc_duration(rf0) - pp.calc_rf_center(rf0)[0] - rf0.delay - g_phase.flat_time - g_phase.rise_time - g_phase.fall_time)

g_read = pp.make_trapezoid(channel='x', flat_area=Nread, flat_time=adc_duration, system=system)
g_read_pre = pp.make_trapezoid(channel='x', area=g_read.area, duration=1e-3, system=system) 
# --> area needs to be positive as the phase will be shifted by pi due to inversion pulse

adc = pp.make_adc(num_samples=Nread, duration=adc_duration, phase_offset=0 * np.pi / 180, delay=g_read.rise_time, system=system)

# add spoiler gradients
g_read_spoil = pp.make_trapezoid(channel='x', area=g_read.area / 2, duration=1e-3, system=system)

# ======
# CONSTRUCT SEQUENCE
# ======

# TODO: better idea than killing magnetization with time --> maybe spoil it away in the end to get rid of transversal magnetization, not soo important
# --> also we need to wait for that time to regain longitudinal magnetization
# alternative: use a dummy 90 degree RARE sequence without signal acquisition and then only wait for 1s to regain longitudinal magnetization
# --> then all RARE sequences for all kspace-lines have the same signal
# ==> use 10s delay for beginning!!!

# TODO: adapt the FFT

for j in range(-Nphase // 2, Nphase // 2):
    # adapt phase gradients magnitude
    if j == 0:
        g_phase = pp.make_trapezoid(channel='y', area=j+1e-12, duration=1e-3, system=system)
        g_phase_revert = pp.make_trapezoid(channel='y', area=-g_phase.area, duration=1e-3, system=system)
    else:
        g_phase = pp.make_trapezoid(channel='y', area=j, duration=1e-3, system=system)
        g_phase_revert = pp.make_trapezoid(channel='y', area=-g_phase.area, duration=1e-3, system=system)

    seq.add_block(rf0)
    # introduce a certain delay between excitation and inversion pulse
    seq.add_block(pp.make_delay(del_90_180), g_read_pre)

    for i in range(num_echos):
        seq.add_block(rf1)
        seq.add_block(g_phase, g_read_spoil)
        seq.add_block(adc, g_read)
        seq.add_block(g_phase_revert, g_read_spoil)
        
    # let some time pass by to kill residual magnetization
    seq.add_block(pp.make_delay(10))


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
seq.write('../out/external.seq')
seq.write('../out/' + experiment_id + '.seq')


# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
sz = [64, 64]

if 0:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)

# Manipulate loaded data
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
    obj_p.D *= 0 
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = mr0.CustomVoxelPhantom(
        pos=[[-0.25, -0.25, 0]],
        PD=[1.0],
        T1=[3.0],
        T2=[0.5],
        T2dash=[10e-3],
        D=[0.0],
        B0=0,
        voxel_size=0.1,
        voxel_shape="box"
    )

obj_p.plot()
obj_p.size=torch.tensor([fov, fov, slice_thickness]) 
# Convert Phantom into simulation data
obj_p = obj_p.build()


# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot

# Read in the sequence 
seq0 = mr0.Sequence.import_file("../out/external.seq")

seq0.plot_kspace_trajectory()
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
spectrum = torch.reshape((signal), (Nread, num_echos, Nphase)).clone()

spaces = []

for ii in range(num_echos):
    kspace_signal = spectrum[:, ii, :]
    kspace = kspace_signal

    # plt.plot(torch.real(signal), label='real')
    # plt.plot(torch.imag(signal), label='imag')
    
    
    # # this adds ticks at the correct position szread
    # major_ticks = np.arange(0, Nphase * Nread, Nread)
    # ax = plt.gca()
    # ax.set_xticks(major_ticks)
    # ax.grid()
    
    space = torch.zeros_like(spectrum)
    
    # fftshift
    kspace_signal = torch.fft.fftshift(kspace_signal, 0)
    kspace_signal = torch.fft.fftshift(kspace_signal, 1)
    # FFT
    space = torch.fft.fft2(kspace_signal)
    # fftshift
    space = torch.fft.fftshift(space, 0)
    space = torch.fft.fftshift(space, 1)
    
    
    spaces.append(space.numpy())
    
    if ii == 9:    
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
        
        # # % compare with original phantom obj_p.PD
        # plt.subplot(348)
        # plt.title('phantom PD')
        # mr0.util.imshow(obj_p.recover().PD.squeeze())
        # plt.subplot(3, 4, 12)
        # plt.title('phantom B0')
        # mr0.util.imshow(obj_p.recover().B1.squeeze())
    
spaces = np.array(spaces)

# %% S7: T2 fitting

expo = np.log(np.abs(spaces[:, 30, 30])) # this is in the box where we have high signal
times = np.arange(num_echos) * (del_90_180 * 2)
plt.plot(times, expo, "o")
plt.show()

# get slope
slope = (expo[-1] - expo[0]) / times[-1]
T2 = -slope
print(T2)