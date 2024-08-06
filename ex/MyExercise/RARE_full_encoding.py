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

experiment_id = 'RARE'


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
Nread = 64 # enlarging Nread makes also the gradients take longer --> thus delta k and FoV is fixed but resolution gets better
Nphase = 64
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

if 1:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../../data/numerical_brain_cropped.mat')
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


# %% S6: MR IMAGE RECON of signal ::: ##############
fig = plt.figure()  # fig.clf()
plt.subplot(411)
plt.title('ADC signal')
plt.plot(torch.real(signal), label='real')
plt.plot(torch.imag(signal), label='imag')

# this adds ticks at the correct position szread
major_ticks = np.arange(0, Nphase * Nread * num_echos, Nread)
ax = plt.gca()
ax.set_xticks(major_ticks)
ax.grid()

spectrum = torch.reshape((signal), (Nread, num_echos, Nphase)).clone()

spaces = []

for ii in range(num_echos):
    kspace_signal = spectrum[:, ii, :].t()
    kspace = kspace_signal.clone() # deep copy
    
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
    
    if ii == 0:
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
        
    
spaces = np.array(spaces) # has real and imaginary parts

# %% S7: apply image mask

threshold = 200 # 450 # threshold experimentally defined having a look at the first spin echo signal

for ii in range(num_echos):
    space = np.abs(spaces[ii]) # absolute image space
    
    # iterate over the image and set signal down to yero if it is below a certain threshold
    for i in range(space.shape[0]):
        for j in range(space.shape[1]):
            if space[i,j] < threshold:
                space[i,j] = 1
                
    spaces[ii] = space        

# %% S8: plotting

# plot all reconstructed images
plt.figure()
for i in range(num_echos):
    plt.subplot(num_echos//5, 5, i+1)
    mr0.util.imshow(np.abs(spaces[i]))
    plt.colorbar()

# %% S9: T2 fitting

# def my_exp(x, a, b):
#     return a * np.exp(b * x)

# create T2 map
T2_map = np.zeros(shape=(Nread, Nphase))

time = (np.arange(num_echos) + 1) * ((del_90_180 + pp.calc_duration(rf0)) * 2)

# iterate over al pixels
for i in range(T2_map.shape[0]):
    for j in range(T2_map.shape[1]):
        
        signal = np.log(np.abs(spaces[:, i, j]))
        
        # perform a linear fit
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(time, signal)
        
        # perform an exponential fit
        # from scipy.optimize import curve_fit
        # popt, pcov = curve_fit(my_exp, time, signal)
        
        if i == 16 and j == 16: # this is in the box where we have high signal
            y_pred = slope * time + intercept
            
            plt.figure()
            plt.scatter(time, signal, color="blue", label="data points")
            plt.plot(time, y_pred, color="orange", label="fitted line")
            plt.title("T2 relaxation within a single pixel")
            plt.xlabel("time")
            plt.ylabel("signal")
            plt.legend()
            plt.show()
            
        # the slope is the negative relaxation rate R2
        if slope == 0:
            T2_measured = 0
        else:
            T2_measured = -1 / slope
            
        T2_map[i,j] = T2_measured
        
       
plt.figure()
mr0.util.imshow(T2_map, vmin=0, vmax=obj_p.T2[:].numpy().max())
plt.colorbar()
plt.title("T2 map")


# %% S10: compare T2 maps
plt.figure()
plt.subplot(1, 2, 1)
mr0.util.imshow(obj_p.recover().T2.squeeze())
plt.colorbar()
plt.title("True T2 map")

plt.subplot(1, 2, 2)
mr0.util.imshow(T2_map, vmin=0, vmax=obj_p.T2[:].numpy().max())
plt.colorbar()
plt.title("Measured T2 map")

# scatterplot
# remove outliers first
for i in range(T2_map.shape[0]):
    for j in range(T2_map.shape[1]):
        if T2_map[i,j] > (obj_p.T2[:].numpy().max() + 0.2) or (T2_map[i,j] < obj_p.T2[:].numpy().min() - 0.2):
            T2_map[i,j] = obj_p.recover().T2.squeeze()[i,j] # outliers are mapped onto regression line
            
plt.figure()
plt.scatter(obj_p.recover().T2.squeeze(), T2_map)
