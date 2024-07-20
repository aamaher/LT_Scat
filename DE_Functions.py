import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

class pt_src:
    
    # PARAMETERS
    n = 1.33  # Medium (Brain)
    mus = 2
    mua = 0.01
    l_mfp = 1 / mus

    # Detector
    M = 2.5

    # Setting fluorophore and time axis
    
    a = 50
    b = 100
    df = 0.05

    def __init__(self):  
        self.v = 2.99792458e11 / self.n  # velocity in medium in mm/s
        self.D = 1 / (3.0 * (self.mua + self.mus))  # diffusion constant in mm
        self.SetTimeAxis()
        self.SetBoundary()
        self.SetSrc()
        self.SetDet()
        self.SetFluoro()
    
    def SetTimeAxis( self,dt = 0.2e-9, Length = 200):
        self.dt = dt
        self.Length = Length
        reduction = 0

        self.delay_f = 10 * dt
        self.Peak_index = self.delay_f/self.dt

        #Peak_index = np.max(delay_f) / dt
        Tmax = dt * self.Length / 2  # sec
        self.time = np.arange(-Tmax + dt, Tmax + dt, dt)
        self.t_positive = np.arange(dt, Tmax + dt, dt)
        self.time_long = np.arange(dt, Tmax - reduction * dt, dt)
        self.Length = len(self.t_positive)

    def SetBoundary(self,x_min = 0,x_max = 0,z_min = -15,z_max = -4,y_min = 0,  y_max = 0):
        ## Setting boundaries
        self.x_min, self.x_max = x_min, x_max  # mm
        self.z_min, self.z_max = z_min, z_max  # mm
        self.y_min, self.y_max = y_min, y_max  # mm    

    def SetSrc(self,x_d = 0, y_d = 0, z_d = -15):
        # Default is bot Illumination
        self.rs = np.array([x_d, y_d, z_d]).reshape(1, 3) 
        # Source
        self.So = 1  # Source energy (J)
        self.Beta = 1  # Modulation depth
    
    def SetDet(self,ND_x = 100,ND_y = 100,SNR_max = 40):
        self.Ae = 1  # Area (mm^2)
        self. Q = 1  # Efficiency
        self.ND_x = ND_x
        self.ND_y = ND_y
        self.alpha_fixed = 9.4 * 10**-6  # alpha defined by Milstein (2*Ye's alpha)
        self.rd = np.zeros((ND_x * ND_y, 3))
        self.SNR_max = SNR_max
        self.Det_size = 8 / 150  # 8mm is 150 pixels, # Double check that Det_size and the Voxel size make sense together
        self.V_vox = self.Det_size**3
        det_xaxis = np.zeros(ND_x)
        det_yaxis = np.zeros(ND_y)
        for i in range(ND_x):
            for j in range(ND_y):
                idx = i * self.Det_size - ND_x * self.Det_size / 2
                idy = j * self.Det_size - ND_y * self.Det_size / 2
                self.rd[j + (i - 1) * ND_x, :] = [idx, idy, 0]
                if i == 1:
                    det_yaxis[j] = idy
                det_xaxis[i] = idx

    def SetFluoro(self,Tau = 3.5e-9,Eta = 0.6, x = 0, y = 0, z = 0):
        self.rf = np.array([x,y,z]) 
        self.Eta = Eta
        self.Tau = Tau
    
    def GetImage(self):
        P = self.rd.shape[0] * self.rs.shape[0]
        Y = np.zeros((P, len(self.time_long)))
        # For each f_r, take data from dt to Tmax
        f_r_start = int(self.Length / 2) + 1
        f_r_end = int(self.Length)

        f_r_sfd = self.fun_f_r()
        Y_image = f_r_sfd[:, f_r_start:f_r_end]

        #Y_image = self.fun_add_noise(Y_image, self.time_long, self.alpha_fixed, self.SNR_max)

        
        Y_image = np.reshape(Y_image[:,int(self.Peak_index)],(self.ND_x,self.ND_y )) ### Note: there could be an issue here
        # Loop for setting variable parameter and calculating forward solution
        return Y_image
        
    def fun_f_r(self):

        Length_t = len(self.time)
        Length_Q = self.rs.shape[0]
        Length_M = self.rd.shape[0]
        Length_Y = 0
        f_r = np.zeros((Length_Q * Length_M, Length_t))

        for index_s in range(Length_Q):
            for index_d in range(Length_M):
                Tf_td = self.fun_td_T(self.rs[index_s], self.rf, self.rd[index_d])
                f_r[Length_Y, :] = np.sqrt(Tf_td[0, :]**2 + Tf_td[1, :]**2 + Tf_td[2, :]**2)  # vector norm (abs in equation 7 in 1989 paper)
                Length_Y += 1
        return f_r

    def fun_td_T(self,rs,rf,rd):
        # Brian Bentz
        # This function calculates the fluorescence transmittance in an infinite 3D 
        # homogenous highly scattering media. Assumes a point source at rs, point
        # fluorophore at rf, and a detector at rd.
        #
        # Transmittance: the number of photons reaching the surface per unit time 
        # per incident photon (J/s)

        # 5-18-2016

        # rs: vector to excitation source location (mm)
        # rf: vector to observation location (mm)
        # rd: column vector to observation location (mm)
        # 
        # mua: absorption (mm^-1)
        # mus: reduced scattering coefficients (mm^-1)
        # Eta: fluorescence yield (mm^-1)
        # Tau: fluorescence lifetime (sec)
        # 
        # t_positive:  positive time scale (s)
        # n:  refractive index
        # 
        # So: source energy (J)
        # Beta: modulation depth
        # Ae: detector area (mm^2)
        # Q: detector efficiency
        # V_vox: voxel volume

        Phi_td = self.fun_td_Psi(rf, rs)  # the propagation from the source to the point fluorophore
        J_td = self.fun_td_J(rd, rf)  # the fluorescence response
        Sf_td = self.fun_td_Sf()
        
        temp = scipy.signal.fftconvolve(Phi_td, Sf_td, mode='same')

        temp1 = self.Ae * self.Q * np.abs(self.So * self.Beta * self.V_vox * scipy.signal.fftconvolve(temp, J_td[0, :], mode='same'))
        temp2 = self.Ae * self.Q * np.abs(self.So * self.Beta * self.V_vox * scipy.signal.fftconvolve(temp, J_td[1, :], mode='same'))
        temp3 = self.Ae * self.Q * np.abs(self.So * self.Beta * self.V_vox * scipy.signal.fftconvolve(temp, J_td[2, :], mode='same'))

        #temp = self.myconv(Phi_td, Sf_td, self.dt)        #### Salem: Note - Maybe this can be replaced here

        #temp1 = self.So * self.Beta * self.V_vox * self.myconv(temp, J_td[0, :], self.dt)
        #temp2 = self.So * self.Beta * self.V_vox * self.myconv(temp, J_td[1, :], self.dt)
        #temp3 = self.So * self.Beta * self.V_vox * self.myconv(temp, J_td[2, :], self.dt)
        #temp1 = np.abs(temp1) * 
        #temp2 = np.abs(temp2) * 
        #temp3 = np.abs(temp3) * 

        Tf_td = np.array([temp1, temp2, temp3])

        return Tf_td

    def fun_td_Psi(self, rf, rs):
        # Brian Bentz
        # This function calculates the time domain Green's function solution of Psi
        # to the diffusion equation in an infinite homogenous highly scattering 
        # media.
        #
        # Psi: photon flux density (1/s/mm^2)
        # see: equations (3) in 1989 Patterson, Wilson, AppOpt 28

        # If taking fft: note that this is a real function so: FT(-w) = FT*(w)
        # where * is complex conjugate

        # 5-18-2016

        # rf: vector to observation location (mm)
        # rs: vector to excitation source location (mm)
        # mua: absorption (mm^-1)
        # mus: reduced scattering coefficients (mm^-1)
        # t_positive:  positive time scale (s)
        # n:  refractive index

        r = np.linalg.norm(rf - rs)

        A = self.v * (4 * np.pi * self.D * self.v)**(-3/2) * (self.t_positive**(-3/2))
        B = np.exp(-r**2 / (4 * self.D * self.v) / self.t_positive - self.mua * self.v * self.t_positive)
        Phi = A * B

        # zero pad
        Phi = np.concatenate((np.zeros(self.Length), Phi))

        return Phi

    def fun_td_J(self, rd, rf):
        # Brian Bentz
        # This function calculates the time domain Green's function solution of J
        # to the diffusion equation in an infinite homogenous highly scattering 
        # media.
        #
        # J: current density (1/s/mm^2)

        # If taking fft: note that this is a real function so: FT(-w) = FT*(w)
        # where * is complex conjugate

        # 5-18-2016

        # rd: column vector to observation location (mm)
        # rf: column vector to source location (mm)
        # mua: absorption (mm^-1)
        # mus: reduced scattering coefficients (mm^-1)
        # t_positive: row vector of the positive time scale (s)
        # n: refractive index

        rv = rd - rf  # vector
        r = np.linalg.norm(rv)  # magnitude

        A = 1/16 * ((np.pi * self.D * self.v)**(-3/2)) * (self.t_positive**(-5/2))     ##  Salem: Should this 1/16 be 1/8??
        B = np.exp(-r**2 / (4 * self.D * self.v) / self.t_positive - self.mua * self.v * self.t_positive)
        J = A * B

        # zero pad

        Length = len(self.t_positive)
        temp1 = rv[0] * np.concatenate((np.zeros(Length), J))
        temp2 = rv[1] * np.concatenate((np.zeros(Length), J))
        temp3 = rv[2] * np.concatenate((np.zeros(Length), J))
        J = np.array([temp1, temp2, temp3])

        return J

    def fun_td_Sf(self):
        # Brian Bentz
        # This function calculates the time domain fluorescence source term Sf
        # used in the diffusion equation
        #
        # Sf: current density (1/s/mm^3)

        # If taking fft: note that this is a real function so: FT(-w) = FT*(w)
        # where * is complex conjugate

        # 5-18-2016

        # Eta: fluorescence yield (mm^-1)
        # Tau: fluorescence lifetime (sec)
        # t_positive:  positive time scale (sec)


        Sf = self.Eta / self.Tau * np.exp(-self.t_positive / self.Tau)

        # zero pad
        Length = len(self.t_positive)
        Sf = np.concatenate((np.zeros(Length), Sf))

        return Sf

  # def myconv(self, A, B, delta):
  #     N = len(A)
  #     return self.ift(self.ft(A, delta) * self.ft(B, delta), 1 / (N * delta))

  # def ft(self, g, delta):
  #     return np.fft.fftshift(np.fft.fft(np.fft.fftshift(g))) * delta

  # def ift(self, G, delta_f):
  #     return np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G)) * len(G) * delta_f)

    def fun_add_noise(self, f_r, time_long, alpha_fixed, SNR_max):
        # Add noise to the analytical data in a homogenous medium
        # Salem: Edited from the original code written by Brian for simplicity, needs to be checked again
        # add noise (Ye 1999)

        delta = 1  # modify to not plot all the points for clarity

        row, col = f_r.shape
        snr = np.zeros_like(f_r)
        f_r_noisy = np.zeros_like(f_r)

        for index in range(row):
            for index2 in range(col):
                snr[index, index2] = SNR_max
                f_r_noisy[index, index2] = f_r[index, index2] + abs(f_r[index, index2]) / (10**(SNR_max / 10)) * np.random.randn()
            # make positive because negative values don't make sense
            f_r_noisy[index, :] = np.abs(f_r_noisy[index, :])

        return f_r_noisy

# Main script
Pt = pt_src()
Pt.SetFluoro(3.5e-9,0.6,0  ,0,-8)



start_time = time.time()      
Im = Pt.GetImage()
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.6f} seconds".format(elapsed_time))

plt.figure()
plt.imshow(Im, cmap='gray')
plt.show()
plt.colorbar()
plt.title('Forward Data')