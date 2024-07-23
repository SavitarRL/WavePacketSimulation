import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from classes.lattice_classes import *
import os

class GoldmanHamiltonian:
    def __init__(self, size, phi, theta, J):
        # parameters
        self.size = size
        self.phi = phi
        self.theta = theta
        self.J = J

        #exp terms
        self.expsigmaY = np.array([[J*math.cos(phi), J*math.sin(phi)], [-J*math.sin(phi), J*math.cos(phi)]])
        self.conexpsigmaY = self.expsigmaY.conj().T
        self.expsigmaX = np.array([[J*math.cos(theta), 1j*J*math.sin(theta)], [1j*J*math.sin(theta), J*math.cos(theta)]])
        self.conexpsigmaX = self.expsigmaX.conj().T

        # identity
        self.Identity = np.eye(self.size)

    def T_matrix(self):
        T = np.kron(self.Identity,self.expsigmaY)
        return T
    
    def Tdagger_matrix(self):
        TDagger = np.kron(self.Identity,self.conexpsigmaY)
        return TDagger

    def hopping_right_matrix(self):
        HoppingMatrixRight= np.zeros((self.size,self.size))
        for i in range(self.size - 1):
            HoppingMatrixRight[i][i+1] =1 
        return HoppingMatrixRight

    def hopping_left_matrix(self):
        HoppingMatrixLeft= np.zeros((self.size,self.size))
        for i in range(self.size - 1):
            HoppingMatrixLeft[i+1][i] =1 
        return HoppingMatrixLeft
    
    def U_matrix(self):
        hop_right = np.kron(self.hopping_right_matrix(), self.conexpsigmaX) 
        hop_left = np.kron(self.hopping_left_matrix(), self.expsigmaX)
        U = hop_right + hop_left
        return U
    
    def hamiltonian(self):
        H_x = np.kron(self.Identity,self.U_matrix())
        H_left = np.kron(self.hopping_left_matrix(), self.T_matrix())
        H_right = np.kron(self.hopping_right_matrix(), self.Tdagger_matrix())
        H_y = H_left + H_right
        return H_x + H_y

class Simulate:
    def __init__(self, size, phi, theta, J, L_x, L_y, delta_t, frame_num):
        
        # hamiltonian from GoldmanHamiltonian class
        self.hamiltonian = GoldmanHamiltonian(size, phi, theta, J).hamiltonian()
        
        # lattice information from lattice_classes
        self.lattice_instance = Lattice(L_x, L_y)
        self.lattice = Lattice(L_x, L_y).lattice
        self.L_x = L_x
        self.L_y = L_y
        
        # for time evolution animation
        self.delta_t = delta_t
        self.time_evo = self.set_time_evol()
        self.frame_num = frame_num


    def setup_statevector(self):
        
        start_lattice_size_1 = self.L_x * self.L_y + self.L_x
        start_lattice_size_2 = self.L_x * self.L_y + self.L_x + 1

        psi_0=np.zeros((2*self.L_x*self.L_y,1))/np.sqrt(2)
        psi_0[start_lattice_size_1]=1/np.sqrt(2)
        psi_0[start_lattice_size_2]=1/np.sqrt(2)
        return psi_0


    def set_time_evol(self):
        hamiltonian = self.hamiltonian
        time_evo = np.identity(2*self.L_x*self.L_y)-1.j*self.delta_t*hamiltonian
        return time_evo
    
    # function for storing time evolution of wavefunction in frame_num many timesteps
    def time_evol_wavefunction(self):
        psi_0 = self.setup_statevector()
        a = np.zeros((2*self.L_x*self.L_y,self.frame_num))

        for i in range(1,self.frame_num):
            #Time evolution operator multiplied by wave function at each step
            psi_0=np.matmul(self.time_evo, psi_0)    #Collect all wave functions in together
            a[:,[i]]=psi_0
        return a


    #Calculate the square magnitude/probability of each timely-framed wavefunction and stored it a new matrix
    def time_evol_probabilities(self):
        
        a = self.time_evol_wavefunction()
        d = np.zeros((self.L_x*self.L_y,self.frame_num))

        for ii in range(0, self.frame_num-1):
            b=a[:,ii]
            for k in range(0, 2*self.L_x*self.L_y,2):
                d[k//2][ii] = math.sqrt((abs(b[k]))**2 +(abs(b[k+1]))**2)        
        return d
    
    # void function to remove the older to new plot
    def update_plot(self, frame_number, xv, yv, d, plot, ax):
    
        plot[0].remove()    
        #Plot density of wave packet versus x and y corrdinates
        plot[0] = ax.plot_surface(xv, yv, np.reshape((np.abs(d[:,[frame_number]]))**2, (self.L_x,self.L_y)), cmap="magma")#Create 3D plot
    

    def make_simulation_gif(self, filename, fps = 20, color='0.75', rstride=1, cstride=1):
        
        # mesh grid
        x_co = self.lattice_instance.x_co
        y_co = self.lattice_instance.y_co
        d = self.time_evol_probabilities()

        X = x_co
        Y = y_co
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

