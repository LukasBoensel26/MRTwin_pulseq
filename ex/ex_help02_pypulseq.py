# %% S0. SETUP env
import pypulseq as pp
import numpy as np
import MRzeroCore as mr0
# makes the ex folder your working directory
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir(os.path.abspath(os.path.dirname(__file__)))

# %% GENERATE and WRITE a sequence   .seq
# %% S1. SETUP sys

# choose the scanner limits
system = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6,
    grad_raster_time=50 * 10e-6
)
# grad_raster_time is optional

# %% S2. DEFINE the sequence
seq = pp.Sequence(system) 

# Define rf events
rf, _, _ = pp.make_sinc_pulse(
    flip_angle=90.0 * np.pi / 180, duration=2e-3, slice_thickness=8e-3,
    apodization=0.5, time_bw_product=4, system=system, return_gz=True
)
# apodization to bring the sinc slightly to zero
# time_bw_product = number of rings of the sinc

# rf1, _, _ = pp.make_block_pulse(
#     flip_angle=90 * np.pi / 180, duration=1e-3, system=system, return_gz=True
# )

# --> block pulse if no slice selection

# Define other gradients and ADC events
gx = pp.make_trapezoid(channel='y', area=80, duration=2e-3, system=system) # area = gradient moment
adc = pp.make_adc(num_samples=128, duration=10e-3, phase_offset=0 * np.pi / 180, system=system) # analog to digital converter
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=system)
del15 = pp.make_delay(0.0015)

seq.add_block(del15) # delay
seq.add_block(rf)
seq.add_block(gx_pre)
seq.add_block(adc, gx)

seq.plot()
# PLOT sequence
mr0.util.pulseq_plot(seq) # extension of seq.plot()


# %% S2. DEFINE the sequence --> with flat_area gradient
seq = pp.Sequence(system) 

# Define rf events
rf, _, _ = pp.make_sinc_pulse(
    flip_angle=90.0 * np.pi / 180, duration=2e-3, slice_thickness=8e-3,
    apodization=0.5, time_bw_product=4, system=system, return_gz=True
)
# apodization to bring the sinc slightly to zero
# time_bw_product = number of rings of the sinc

# rf1, _, _ = pp.make_block_pulse(
#     flip_angle=90 * np.pi / 180, duration=1e-3, system=system, return_gz=True
# )

# --> block pulse if no slice selection

# Define other gradients and ADC events
gx = pp.make_trapezoid(channel='y', flat_area=80, flat_time=2e-3, system=system) # area = gradient moment
gx_pre = pp.make_trapezoid(channel='x', flat_area=-gx.area / 2, flat_time=2e-3, system=system)
adc = pp.make_adc(num_samples=128, duration=2e-3, delay=gx.rise_time, phase_offset=0 * np.pi / 180, system=system) # analog to digital converter
del15 = pp.make_delay(0.0015)

seq.add_block(del15) # delay
seq.add_block(rf)
seq.add_block(gx_pre)
seq.add_block(adc, gx)

seq.plot()
# PLOT sequence
mr0.util.pulseq_plot(seq) # extension of seq.plot()


# %% S2.1 DEFINE the sequence of exercise 1
seq = pp.Sequence(system) 

# Define rf events
rf1, _, _ = pp.make_sinc_pulse(
    flip_angle=90.0 * np.pi / 180, duration=2e-3, slice_thickness=8e-3,
    apodization=0.5, time_bw_product=4, system=system, return_gz=True
)
rf2, _, _ = pp.make_sinc_pulse(
    flip_angle=120.0 * np.pi / 180, duration=2e-3, slice_thickness=8e-3,
    apodization=0.5, time_bw_product=4, system=system, return_gz=True
)
# apodization to bring the sinc slightly to zero
# time_bw_product = number of rings of the sinc

# rf1, _, _ = pp.make_block_pulse(
#     flip_angle=90 * np.pi / 180, duration=1e-3, system=system, return_gz=True
# )

# --> block pulse if no slice selection

# Define other gradients and ADC events
del15 = pp.make_delay(0.0015)
adc = pp.make_adc(num_samples=128, duration=10e-3, phase_offset=0 * np.pi / 180, system=system) # analog to digital converter

gx = pp.make_trapezoid(channel='x', area=-30, duration=2e-3, system=system)
gy = pp.make_trapezoid(channel='y', area=-gx.area, duration=2e-3, system=system)

seq.add_block(del15) # delay
seq.add_block(rf1)
seq.add_block(gx)
seq.add_block(gy)
seq.add_block(gy)
seq.add_block(gx)
seq.add_block(rf2)
seq.add_block(gx, gy, adc)

# seq.plot()
# PLOT sequence
mr0.util.pulseq_plot(seq) # extension of seq.plot()


# %% S2.2 DEFINE the sequence of exercise 2
seq = pp.Sequence(system) 

# Define rf events
rf1, _, _ = pp.make_sinc_pulse(
    flip_angle=90.0 * np.pi / 180, duration=2e-3, slice_thickness=8e-3,
    apodization=0.5, time_bw_product=4, system=system, return_gz=True
)
rf2, _, _ = pp.make_sinc_pulse(
    flip_angle=120.0 * np.pi / 180, duration=2e-3, slice_thickness=8e-3,
    apodization=0.5, time_bw_product=4, system=system, return_gz=True
)

delay1 = pp.make_delay(1e-1 - pp.calc_rf_center(rf1)[0] - rf.delay) # but which delay
# delay2 = pp.make_delay(2e-1 - pp.calc_rf_center(rf2)[0] - rf.delay - delay1.delay - pp.calc_duration(rf1)) # but which delay

delay2 = pp.make_delay(2e-1 - 1e-1 - (pp.calc_duration(rf1) - rf1.delay - rf1.ringdown_time) / 2)
# look at this again

seq.add_block(delay1)
seq.add_block(rf1)
seq.add_block(delay2)
seq.add_block(rf2)

mr0.util.pulseq_plot(seq)

# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

seq.plot()

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [1, 1, 8e-3])  # in m
seq.set_definition('Name', 'test')
seq.write('out/external.seq')


# %% LOAD and PLOT a sequence   .seq
seq = pp.Sequence(system) 

seq.read('out/external.seq')

seq.plot()
# PLOT sequence
mr0.util.pulseq_plot(seq)


# %% Exact pulse timing
#S1. SETUP sys and DEFINE the sequence

# choose the scanner limits
system = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6,
    adc_dead_time=20e-6
)

seq = pp.Sequence(system) 

# Define FOV and resolution
fov = 1000e-3
Nread = 128
Nphase = 1
slice_thickness = 8e-3  # slice

# Define rf events
rf, _, _ = pp.make_sinc_pulse(
    flip_angle=1 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    delay=0.0, system=system, return_gz=True
)

# what i want: an RF pulse at exactly 0.1

rf.delay                    # delay before the start of the pulse due to dead time or gradient rise time
ct=pp.calc_rf_center(rf)    # rf center time returns time and index of the center of the pulse
ct[0]                       # this is the rf center time
rf.ringdown_time            # this is the rf ringdown time after the rf pulse
pp.calc_duration(rf)        # this is the rf duration including delay and ringdown time


print("for a pulse with requested duration 1 ms the total duration is")
print("%0.5f" %(pp.calc_duration(rf) )  )     # this is the rf duration including delay and ringdown time
print("consiting of ")
print("%0.5f" %(1e-3), "s requested duration")                # requested duration
print("%0.5f" %(rf.delay), "s rf  delay"   )               # delay before the start of the pulse due to dead time or gradient rise time
print("%0.5f" %(rf.ringdown_time), "s ringdown_time")        # this is the rf ringdown time after the rf pulse

ct=pp.calc_rf_center(rf)    # rf center time returns time and index of the center of the pulse
print("%0.5f" %(ct[0]), "s after the delay, we have the center of the pulse, thus at"  )                # this is the rf center time
print("%0.5f" %(ct[0]+rf.delay), "s after the start of the block"  )                # this is the rf center time


delay1=pp.make_delay(.10)
# delay1=pp.make_delay(.10 - rf.delay)
# delay1=pp.make_delay(.10 - rf.delay - ct[0])

# ======
# CONSTRUCT SEQUENCE
# ======
seq.add_block(delay1)
seq.add_block(rf)

# # Bug: pypulseq 1.3.1post1 write() crashes when there is no gradient event
seq.add_block(pp.make_trapezoid('x', duration=20e-3, area=10))

# % S3. CHECK, PLOT and WRITE the sequence  as .seq
# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc, t_adc = mr0.util.pulseq_plot(seq)