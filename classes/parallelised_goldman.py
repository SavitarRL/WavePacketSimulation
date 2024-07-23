import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from classes.lattice_classes import *
from classes.goldman_classes import *
import os
from numba import cuda, complex128
import cupy as cp

class ParallelisedGoldmanHamiltonian(GoldmanHamiltonian):
    def __init__(self, size, phi, theta, J):
        
        super().__init__(size, phi, theta, J)
        
        # converting to CUDA device arrays
        self.expsigmaY_cuda = cp.asarray(self.expsigmaY)
        self.conexpsigmaY_cuda = cp.asarray(self.conexpsigmaY)
        self.expsigmaX_cuda = cp.asarray(self.expsigmaX)
        self.conexpsigmaX_cuda = cp.asarray(self.conexpsigmaX)
        self.Identity_cuda = cp.asarray(self.Identity)

    def T_matrix(self):
        T = cp.kron(self.Identity,self.expsigmaY_cuda)
        return T
    
    def Tdagger_matrix(self):
        TDagger = cp.kron(self.Identity,self.conexpsigmaY_cuda)
        return TDagger

    def hopping_right_matrix(self):
        HoppingMatrixRight= cp.zeros((self.size,self.size),dtype=cp.float64)
        for i in range(self.size - 1):
            HoppingMatrixRight[i][i+1] =1 
        return HoppingMatrixRight

    def hopping_left_matrix(self):
        HoppingMatrixLeft= cp.zeros((self.size,self.size),dtype=cp.float64)
        for i in range(self.size - 1):
            HoppingMatrixLeft[i+1][i] =1 
        return HoppingMatrixLeft
    
    def U_matrix(self):
        hop_right = cp.kron(self.hopping_right_matrix(), self.conexpsigmaX_cuda) 
        hop_left = cp.kron(self.hopping_left_matrix(), self.expsigmaX_cuda)
        U = hop_right + hop_left
        return U
    
    def hamiltonian(self):
        H_x = cp.kron(self.Identity_cuda,self.U_matrix())
        H_left = cp.kron(self.hopping_left_matrix(), self.T_matrix())
        H_right = cp.kron(self.hopping_right_matrix(), self.Tdagger_matrix())
        H_y = H_left + H_right
        return H_x + H_y

class ParallelisedSimulate(Simulate):
    def __init__(self, size, phi, theta, J, L_x, L_y, delta_t, frame_num):
        
        super().__init__(size, phi, theta, J, L_x, L_y, delta_t, frame_num)

        # hamiltonian from ParallelisedGoldmanHamiltonian class
        self.p_hamiltonian = ParallelisedGoldmanHamiltonian(size, phi, theta, J).hamiltonian()
        
        # lattice information from parallelised Lattice from lattice_classes
        self.lattice_instance = ParallelisedLattice(L_x, L_y)
        self.lattice = ParallelisedLattice(L_x, L_y).lattice
        self.L_x = L_x
        self.L_y = L_y
        
        # for time evolution animation

        self.time_evo = self.set_time_evol()


    def set_time_evol(self):
        p_hamiltonian = self.p_hamiltonian
        time_evo = cp.identity(2 * self.L_x * self.L_y) - 1j * self.delta_t * p_hamiltonian
        return time_evo
    
    def setup_statevector(self):
        
        start_lattice_size_1 = self.L_x * self.L_y + self.L_x
        start_lattice_size_2 = self.L_x * self.L_y + self.L_x + 1

        psi_0=cp.zeros((2*self.L_x*self.L_y,1))/cp.sqrt(2)
        psi_0[start_lattice_size_1]=1/cp.sqrt(2)
        psi_0[start_lattice_size_2]=1/cp.sqrt(2)
        return psi_0

    # function for storing time evolution of wavefunction in frame_num many timesteps
    def time_evol_wavefunction(self):
        psi_0 = self.setup_statevector()
        a = cp.zeros((2*self.L_x*self.L_y,self.frame_num))

        for i in range(1,self.frame_num):
            #Time evolution operator multiplied by wave function at each step
            psi_0=cp.matmul(self.time_evo, psi_0)    #Collect all wave functions in together
            a[:,[i]]=psi_0
        return a


    #Calculate the square magnitude/probability of each timely-framed wavefunction and stored it a new matrix
    def time_evol_probabilities(self):
        
        a = self.time_evol_wavefunction()
        d = cp.zeros((self.L_x*self.L_y,self.frame_num))

        for ii in range(0, self.frame_num-1):
            b=a[:,ii]
            for k in range(0, 2*self.L_x*self.L_y,2):
                d[k//2][ii] = math.sqrt((abs(b[k]))**2 +(abs(b[k+1]))**2)        
        return d
    
    # void function to remove the older to new plot
    def update_plot(self, frame_number, xv, yv, d, plot, ax):
    
        plot[0].remove()    
        #Plot density of wave packet versus x and y corrdinates
        plot[0] = ax.plot_surface(xv, yv, cp.reshape((cp.abs(d[:,[frame_number]]))**2, (self.L_x,self.L_y)), cmap="magma")#Create 3D plot
    

    def make_simulation_gif(self, filename, fps = 20, color='0.75', rstride=1, cstride=1):
        
        # mesh grid
        x_co = self.lattice_instance.x_co
        y_co = self.lattice_instance.y_co
        d = self.time_evol_probabilities()

        X = cp.asnumpy(x_co)
        Y = cp.asnumpy(y_co)
        xv, yv = np.meshgrid(X, Y)#Define function to updating plot
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') #set view position as top
        ax.view_init(45,0)
        
        #Plot initial wave function 
        plot_wavefunction = [ax.plot_surface(xv, yv, np.reshape((np.abs(d[:,[0]]))**2, (self.L_x,self.L_y)), color=color, rstride=rstride, cstride=cstride)]
        
        ax.set_zlim(0,1.1) #Create animation
        
        ani = FuncAnimation(fig, self.update_plot, self.frame_num, fargs=(xv, yv, d, plot_wavefunction, ax), interval=1000/fps) #Save your animation in.gif format
        
        path = "GIF"
        name = filename + ".gif"
        gif_dir = os.path.join(path, name)
        ani.save(gif_dir, writer='pillow', fps = fps)

        # print("gif successfully saved at {}".format(gif_dir))

