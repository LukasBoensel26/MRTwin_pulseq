import numpy as np
import torch


# HOW we measure
class Scanner():
    
    def __init__(self,sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        self.NRep = NRep                              # number of repetitions
        self.T = T                       # number of "actions" with a readout
        self.NCoils = NCoils                # number of receive coil elements
        self.noise_std = noise_std              # additive Gaussian noise std
        
        self.adc_mask = None         # ADC signal acquisition event mask (T,)
        self.rampX = None        # spatial encoding linear gradient ramp (sz)
        self.rampY = None
        self.F = None                              # flip tensor (T,NRep,4,4)
        self.R = None                          # relaxation tensor (NVox,4,4)
        self.P = None                   # free precession tensor (NSpins,4,4)
        self.G = None            # gradient precession tensor (NRep,NVox,4,4)
        self.G_adj = None         # adjoint gradient operator (NRep,NVox,4,4)

        self.B0_grad_cos = None  # accum phase due to gradients (T,NRep,NVox)
        self.B0_grad_sin = None
        self.B0_grad_adj_cos = None  # adjoint grad phase accum (T,NRep,NVox)
        self.B0_grad_adj_sin = None
        
        self.B1 = None          # coil sensitivity profiles (NCoils,NVox,2,2)

        self.signal = None                # measured signal (NCoils,T,NRep,4)
        self.reco =  None                       # reconstructed image (NVox,) 
        
        self.use_gpu =  use_gpu
        
    # device setter
    def setdevice(self,x):
        if self.use_gpu:
            x = x.cuda(0)
            
        return x        
        
    def set_adc_mask(self):
        adc_mask = torch.from_numpy(np.ones((self.T,1))).float()
        adc_mask[:self.T-self.sz[0]] = 0
        
        self.adc_mask = self.setdevice(adc_mask)

    def get_ramps(self):
        
        use_nonlinear_grads = False                       # very experimental
        
        baserampX = np.linspace(-1,1,self.sz[0] + 1)
        baserampY = np.linspace(-1,1,self.sz[1] + 1)
        
        if use_nonlinear_grads:
            baserampX = np.abs(baserampX)**1.2 * np.sign(baserampX)
            baserampY = np.abs(baserampY)**1.2 * np.sign(baserampY)
        
        rampX = np.pi*baserampX
        rampX = -np.expand_dims(rampX[:-1],1)
        rampX = np.tile(rampX, (1, self.sz[1]))
        
        rampX = torch.from_numpy(rampX).float()
        rampX = rampX.view([1,1,self.NVox])    
        
        # set gradient spatial forms
        rampY = np.pi*baserampY
        rampY = -np.expand_dims(rampY[:-1],0)
        rampY = np.tile(rampY, (self.sz[0], 1))
        
        rampY = torch.from_numpy(rampY).float()
        rampY = rampY.view([1,1,self.NVox])    
        
        self.rampX = self.setdevice(rampX)
        self.rampY = self.setdevice(rampY)
        
    def init_coil_sensitivities(self):
        # handle complex mul as matrix mul
        B1 = torch.zeros((self.NCoils,1,self.NVox,2,2), dtype=torch.float32)
        B1[:,0,:,0,0] = 1
        B1[:,0,:,1,1] = 1
        
        self.B1 = self.setdevice(B1)
        
    def init_flip_tensor_holder(self):
        F = torch.zeros((self.T,self.NRep,1,4,4), dtype=torch.float32)
        
        F[:,:,0,3,3] = 1
        F[:,:,0,1,1] = 1
         
        self.F = self.setdevice(F)
         
    def set_flip_tensor(self,flips):
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        self.F[:,:,0,0,0] = flips_cos
        self.F[:,:,0,0,2] = flips_sin
        self.F[:,:,0,2,0] = -flips_sin
        self.F[:,:,0,2,2] = flips_cos 
         
    def set_relaxation_tensor(self,spins,dt):
        R = torch.zeros((self.NVox,4,4), dtype=torch.float32) 
        
        R = self.setdevice(R)
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,3,3] = 1
        
        R[:,0,0] = T2_r
        R[:,1,1] = T2_r
        R[:,2,2] = T1_r
        R[:,2,3] = 1 - T1_r
         
        R = R.view([1,self.NVox,4,4])
        
        self.R = R
        
    def set_freeprecession_tensor(self,spins,dt):
        P = torch.zeros((self.NSpins,1,1,4,4), dtype=torch.float32)
        
        P = self.setdevice(P)
        
        B0_nspins = spins.dB0.view([self.NSpins])
        
        B0_nspins_cos = torch.cos(B0_nspins*dt)
        B0_nspins_sin = torch.sin(B0_nspins*dt)
         
        P[:,0,0,0,0] = B0_nspins_cos
        P[:,0,0,0,1] = -B0_nspins_sin
        P[:,0,0,1,0] = B0_nspins_sin
        P[:,0,0,1,1] = B0_nspins_cos
         
        P[:,0,0,2,2] = 1
        P[:,0,0,3,3] = 1         
         
        self.P = P
         
    
    def init_gradient_tensor_holder(self):
        G = torch.zeros((self.NRep,self.NVox,4,4), dtype=torch.float32)
        G[:,:,2,2] = 1
        G[:,:,3,3] = 1
         
        G_adj = torch.zeros((self.NRep,self.NVox,4,4), dtype=torch.float32)
        G_adj[:,:,2,2] = 1
        G_adj[:,:,3,3] = 1
         
        self.G = self.setdevice(G)
        self.G_adj = self.setdevice(G_adj)
        
    def set_grad_op(self,t):
        
        self.G[:,:,0,0] = self.B0_grad_cos[t,:,:]
        self.G[:,:,0,1] = -self.B0_grad_sin[t,:,:]
        self.G[:,:,1,0] = self.B0_grad_sin[t,:,:]
        self.G[:,:,1,1] = self.B0_grad_cos[t,:,:]
        
    def set_grad_adj_op(self,t):
        
        self.G_adj[:,:,0,0] = self.B0_grad_adj_cos[t,:,:]
        self.G_adj[:,:,0,1] = self.B0_grad_adj_sin[t,:,:]
        self.G_adj[:,:,1,0] = -self.B0_grad_adj_sin[t,:,:]
        self.G_adj[:,:,1,1] = self.B0_grad_adj_cos[t,:,:]        
        
    def set_gradient_precession_tensor(self,grad_moms):
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        grads = temp[1:,:,:] - temp[:-1,:,:]        
        
        B0X = torch.unsqueeze(grads[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grads[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_cos = torch.cos(B0_grad)
        self.B0_grad_sin = torch.sin(B0_grad)
        
        # for backward pass
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_adj_cos = torch.cos(B0_grad)
        self.B0_grad_adj_sin = torch.sin(B0_grad)
        
    def flip(self,t,r,spins):
        spins.M = torch.matmul(self.F[t,r,:,:,:],spins.M)
        
    # apply flip at all repetition simultanetously (non TR transfer case)
    def flip_allRep(self,t,spins):
        spins.M = torch.matmul(self.F[t,:,:,:,:],spins.M)
        
    def relax(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        
    def relax_and_dephase(self,spins):
        
        spins.M = torch.matmul(self.R,spins.M)
        spins.M = torch.matmul(self.P,spins.M)
        
    def grad_precess(self,r,spins):
        spins.M = torch.matmul(self.G[r,:,:,:],spins.M)        
        
    def grad_precess_allRep(self,spins):
        spins.M = torch.matmul(self.G,spins.M)
        
    def init_signal(self):
        signal = torch.zeros((self.NCoils,self.T,self.NRep,4,1), dtype=torch.float32) 
        signal[:,:,:,2:,0] = 1                                 # aux dim zero
              
        self.signal = self.setdevice(signal)
        
    def init_reco(self):
        reco = torch.zeros((self.NVox,2), dtype = torch.float32)
        
        self.reco = self.setdevice(reco)
        
    def read_signal(self,t,r,spins):
        if self.adc_mask[t] > 0:
            sig = torch.sum(spins.M[:,0,:,:2,0],[0])
            sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(0).unsqueeze(4))
            
            self.signal[:,t,r,:2] = (torch.sum(sig,[2]) * self.adc_mask[t])
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,t,r,:,0].shape).float()
                noise[:,2:] = 0
                noise = self.setdevice(noise)
                self.signal[:,t,r,:,0] = self.signal[:,t,r,:,0] + noise        
        
    def read_signal_allRep(self,t,spins):
        
        if self.adc_mask[t] > 0:
            sig = torch.sum(spins.M[:,:,:,:2,0],[0])
            sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(4))
            self.signal[:,t,:,:2] = torch.sum(sig,[2]) * self.adc_mask[t]
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,t,:,:,0].shape).float()
                noise[:,:,2:] = 0
                noise = self.setdevice(noise)
                self.signal[:,t,:,:,0] = self.signal[:,t,:,:,0] + noise

    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t,spins):
        
        s = self.signal[:,t,:,:,:] * self.adc_mask[t]
        # for now we ignore parallel imaging options here (do naive sum sig over coil)
        s = torch.sum(s, 0)                                                  
        r = torch.matmul(self.G_adj.permute([1,0,2,3]), s)
        self.reco = self.reco + torch.sum(r[:,:,:2,0],1)
        
    ## extra func land        
    # aux flexible operators for sandboxing things
    def custom_flip(self,t,spins,flips):
        
        F = torch.zeros((self.T,self.NRep,1,4,4), dtype=torch.float32)
        
        F[:,:,0,3,3] = 1
        F[:,:,0,1,1] = 1
         
        F = self.setdevice(F)
        
        flips = self.setdevice(flips)
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        F[:,:,0,0,0] = flips_cos
        F[:,:,0,0,2] = flips_sin
        F[:,:,0,2,0] = -flips_sin
        F[:,:,0,2,2] = flips_cos         
        
        spins.M = torch.matmul(F[t,:,:,:],spins.M)
        
    def custom_relax(self,spins,dt=None):
        
        R = torch.zeros((self.NVox,4,4), dtype=torch.float32) 
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,3,3] = 1
        
        R[:,0,0] = T2_r
        R[:,1,1] = T2_r
        R[:,2,2] = T1_r
        R[:,2,3] = 1 - T1_r
         
        R = R.view([1,self.NVox,4,4])
        
        R = self.setdevice(R)
        
        spins.M = torch.matmul(R,spins.M)  
        
        
# variation for supervised learning
# TODO fix relax tensor -- is batch dependent
class Scanner_batched():
    
    def __init__(self,sz,NVox,NSpins,NRep,T,NCoils,noise_std,batch_size,use_gpu):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        self.NRep = NRep                              # number of repetitions
        self.T = T                       # number of "actions" with a readout
        self.NCoils = NCoils                # number of receive coil elements
        self.noise_std = noise_std              # additive Gaussian noise std
        
        self.adc_mask = None         # ADC signal acquisition event mask (T,)
        self.rampX = None        # spatial encoding linear gradient ramp (sz)
        self.rampY = None
        self.F = None                              # flip tensor (T,NRep,4,4)
        self.R = None                          # relaxation tensor (NVox,4,4)
        self.P = None                   # free precession tensor (NSpins,4,4)
        self.G = None            # gradient precession tensor (NRep,NVox,4,4)
        self.G_adj = None         # adjoint gradient operator (NRep,NVox,4,4)

        self.B0_grad_cos = None  # accum phase due to gradients (T,NRep,NVox)
        self.B0_grad_sin = None
        self.B0_grad_adj_cos = None  # adjoint grad phase accum (T,NRep,NVox)
        self.B0_grad_adj_sin = None
        
        self.B1 = None          # coil sensitivity profiles (NCoils,NVox,2,2)

        self.signal = None                # measured signal (NCoils,T,NRep,4)
        self.reco =  None                       # reconstructed image (NVox,) 
        
        self.batch_size = batch_size
        self.use_gpu =  use_gpu
        
    # device setter
    def setdevice(self,x):
        if self.use_gpu:
            x = x.cuda(0)
            
        return x        
        
    def set_adc_mask(self):
        adc_mask = torch.from_numpy(np.ones((self.T,1))).float()
        adc_mask[:self.T-self.sz[0]] = 0
        
        self.adc_mask = self.setdevice(adc_mask)

    def get_ramps(self):
        
        use_nonlinear_grads = False                       # very experimental
        
        baserampX = np.linspace(-1,1,self.sz[0] + 1)
        baserampY = np.linspace(-1,1,self.sz[1] + 1)
        
        if use_nonlinear_grads:
            baserampX = np.abs(baserampX)**1.2 * np.sign(baserampX)
            baserampY = np.abs(baserampY)**1.2 * np.sign(baserampY)
        
        rampX = np.pi*baserampX
        rampX = -np.expand_dims(rampX[:-1],1)
        rampX = np.tile(rampX, (1, self.sz[1]))
        
        rampX = torch.from_numpy(rampX).float()
        rampX = rampX.view([1,1,self.NVox])    
        
        # set gradient spatial forms
        rampY = np.pi*baserampY
        rampY = -np.expand_dims(rampY[:-1],0)
        rampY = np.tile(rampY, (self.sz[0], 1))
        
        rampY = torch.from_numpy(rampY).float()
        rampY = rampY.view([1,1,self.NVox])    
        
        self.rampX = self.setdevice(rampX)
        self.rampY = self.setdevice(rampY)
        
    def init_coil_sensitivities(self):
        # handle complex mul as matrix mul
        B1 = torch.zeros((self.NCoils,1,self.NVox,2,2), dtype=torch.float32)
        B1[:,0,:,0,0] = 1
        B1[:,0,:,1,1] = 1
        
        self.B1 = self.setdevice(B1)
        
    def init_flip_tensor_holder(self):
        F = torch.zeros((1,self.T,self.NRep,1,4,4), dtype=torch.float32)
        
        F[0,:,:,0,3,3] = 1
        F[0,:,:,0,1,1] = 1
         
        self.F = self.setdevice(F)
         
    def set_flip_tensor(self,flips):
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        self.F[0,:,:,0,0,0] = flips_cos
        self.F[0,:,:,0,0,2] = flips_sin
        self.F[0,:,:,0,2,0] = -flips_sin
        self.F[0,:,:,0,2,2] = flips_cos 
         
    def set_relaxation_tensor(self,spins,dt):
        R = torch.zeros((1,self.NVox,4,4), dtype=torch.float32) 
        
        R = self.setdevice(R)
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[0,:,3,3] = 1
        
        R[0,:,0,0] = T2_r
        R[0,:,1,1] = T2_r
        R[0,:,2,2] = T1_r
        R[0,:,2,3] = 1 - T1_r
         
        R = R.view([1,1,self.NVox,4,4])
        
        self.R = R
        
    def set_freeprecession_tensor(self,spins,dt):
        P = torch.zeros((1,self.NSpins,1,1,4,4), dtype=torch.float32)
        
        P = self.setdevice(P)
        
        B0_nspins = spins.dB0.view([self.NSpins])
        
        B0_nspins_cos = torch.cos(B0_nspins*dt)
        B0_nspins_sin = torch.sin(B0_nspins*dt)
         
        P[0,:,0,0,0,0] = B0_nspins_cos
        P[0,:,0,0,0,1] = -B0_nspins_sin
        P[0,:,0,0,1,0] = B0_nspins_sin
        P[0,:,0,0,1,1] = B0_nspins_cos
         
        P[0,:,0,0,2,2] = 1
        P[0,:,0,0,3,3] = 1         
         
        self.P = P
         
    
    def init_gradient_tensor_holder(self):
        G = torch.zeros((1,self.NRep,self.NVox,4,4), dtype=torch.float32)
        G[0,:,:,2,2] = 1
        G[0,:,:,3,3] = 1
         
        G_adj = torch.zeros((1,self.NRep,self.NVox,4,4), dtype=torch.float32)
        G_adj[0,:,:,2,2] = 1
        G_adj[0,:,:,3,3] = 1
         
        self.G = self.setdevice(G)
        self.G_adj = self.setdevice(G_adj)
        
    def set_grad_op(self,t):
        
        self.G[0,:,:,0,0] = self.B0_grad_cos[t,:,:]
        self.G[0,:,:,0,1] = -self.B0_grad_sin[t,:,:]
        self.G[0,:,:,1,0] = self.B0_grad_sin[t,:,:]
        self.G[0,:,:,1,1] = self.B0_grad_cos[t,:,:]
        
    def set_grad_adj_op(self,t):
        
        self.G_adj[0,:,:,0,0] = self.B0_grad_adj_cos[t,:,:]
        self.G_adj[0,:,:,0,1] = self.B0_grad_adj_sin[t,:,:]
        self.G_adj[0,:,:,1,0] = -self.B0_grad_adj_sin[t,:,:]
        self.G_adj[0,:,:,1,1] = self.B0_grad_adj_cos[t,:,:]        
        
    def set_gradient_precession_tensor(self,grad_moms):
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        grads = temp[1:,:,:] - temp[:-1,:,:]        
        
        B0X = torch.unsqueeze(grads[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grads[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_cos = torch.cos(B0_grad)
        self.B0_grad_sin = torch.sin(B0_grad)
        
        # for backward pass
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_adj_cos = torch.cos(B0_grad)
        self.B0_grad_adj_sin = torch.sin(B0_grad)
        
    def flip(self,t,r,spins):
        spins.M = torch.matmul(self.F[0,t,r,:,:,:],spins.M)
        
    # apply flip at all repetition simultanetously (non TR transfer case)
    def flip_allRep(self,t,spins):
        spins.M = torch.matmul(self.F[0,t,:,:,:,:],spins.M)
        
    def relax(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        
    def relax_and_dephase(self,spins):
        
        spins.M = torch.matmul(self.R,spins.M)
        spins.M = torch.matmul(self.P,spins.M)
        
    def grad_precess(self,r,spins):
        spins.M = torch.matmul(self.G[0,r,:,:,:],spins.M)        
        
    def grad_precess_allRep(self,spins):
        spins.M = torch.matmul(self.G,spins.M)
        
    def init_signal(self):
        signal = torch.zeros((self.batch_size,self.NCoils,self.T,self.NRep,4,1), dtype=torch.float32) 
        signal[:,:,:,:,2:,0] = 1                                 # aux dim zero
              
        self.signal = self.setdevice(signal)
        
    def init_reco(self):
        reco = torch.zeros((self.batch_size,self.NVox,2), dtype = torch.float32)
        
        self.reco = self.setdevice(reco)
        
    def read_signal(self,t,r,spins):
        if self.adc_mask[t] > 0:
            sig = torch.sum(spins.M[:,:,0,:,:2,0],[1])
            sig = torch.matmul(self.B1.unsqueeze(0),sig.unsqueeze(1).unsqueeze(1).unsqueeze(5))
            
            self.signal[:,:,t,r,:2] = (torch.sum(sig,[3]) * self.adc_mask[t])
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,:,t,r,:,0].shape).float()
                noise[:,:,2:] = 0
                noise = self.setdevice(noise)
                self.signal[:,:,t,r,:,0] = self.signal[:,:,t,r,:,0] + noise        
        
    def read_signal_allRep(self,t,spins):
        
        if self.adc_mask[t] > 0:
            sig = torch.sum(spins.M[:,:,:,:,:2,0],[1])
            sig = torch.matmul(self.B1.unsqueeze(0),sig.unsqueeze(1).unsqueeze(5))
            self.signal[:,:,t,:,:2] = torch.sum(sig,[3]) * self.adc_mask[t]
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,:,t,:,:,0].shape).float()
                noise[:,:,:,2:] = 0
                noise = self.setdevice(noise)
                self.signal[:,:,t,:,:,0] = self.signal[:,:,t,:,:,0] + noise

    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t,spins):
        s = self.signal[:,:,t,:,:,:] * self.adc_mask[t]
        # for now we ignore parallel imaging options here (do naive sum sig over coil)
        s = torch.sum(s, 1).unsqueeze(1)

        r = torch.matmul(self.G_adj.permute([0,2,1,3,4]), s)
        self.reco = self.reco + torch.sum(r[:,:,:,:2,0],2)
        
    ## extra func land        
    # aux flexible operators for sandboxing things
    def custom_flip(self,t,spins,flips):
        
        F = torch.zeros((self.T,self.NRep,1,4,4), dtype=torch.float32)
        
        F[:,:,0,3,3] = 1
        F[:,:,0,1,1] = 1
         
        F = self.setdevice(F)
        
        flips = self.setdevice(flips)
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        F[:,:,0,0,0] = flips_cos
        F[:,:,0,0,2] = flips_sin
        F[:,:,0,2,0] = -flips_sin
        F[:,:,0,2,2] = flips_cos         
        
        spins.M = torch.matmul(F[t,:,:,:],spins.M)
        
    def custom_relax(self,spins,dt=None):
        
        R = torch.zeros((self.NVox,4,4), dtype=torch.float32) 
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,3,3] = 1
        
        R[:,0,0] = T2_r
        R[:,1,1] = T2_r
        R[:,2,2] = T1_r
        R[:,2,3] = 1 - T1_r
         
        R = R.view([1,self.NVox,4,4])
        
        R = self.setdevice(R)
        
        spins.M = torch.matmul(R,spins.M)          