import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
#from tight_binding_approximation import *
%matplotlib inline

#Lattice Size
L_x=20
L_y=20

#Create 2-D Lattice. Numbers Represents Each Lattice Sites
lat = np.arange(L_x*L_y).reshape(L_x,L_y)
array = np.append(lat, lat, axis=0)
lattice = array[:-L_x, :]

#Respectively Site Coordinates on Lattice
x_co = np.arange(L_x)
y_co = np.arange(L_y)
arr = []
for j in range(len(x_co)):
    for i in range(len(y_co)):
        arr=np.append(arr, [x_co[i], y_co[j]])
xy = arr.reshape((L_x*L_y,2))

#Find Neighbors Each Sites with Tight-Binding Approximation (But for Hard-Wall Boundary Conditions) 
def HardBC(arr):
    neighbors = {}
    for i in range(len(arr)):
        for j, value in enumerate(arr[i]):
            if i == 0 or i == len(arr) - 1 or j == 0 or j == len(arr[i]) - 1:
                new_neighbors = []
                if i != 0:
                    new_neighbors.append(arr[i - 1][j])  
                if j != len(arr[i]) - 1:
                    new_neighbors.append(arr[i][j + 1]) 
                if i != len(arr) - 1:
                    new_neighbors.append(arr[i + 1][j])  
                if j != 0:
                    new_neighbors.append(arr[i][j - 1])
            else:
                new_neighbors = [
                    arr[i - 1][j],  
                    arr[i][j + 1],  
                    arr[i + 1][j],  
                    arr[i][j - 1]   
                ]
            neighbors[value] = new_neighbors
    return neighbors

#Find Neighbors Each Sites with Tight-Binding Approximation (But for Periodic Boundary Conditions) 
def PerBC(arr):
    neighbors = {}
    for i in range(len(arr)):
        for j, value in enumerate(arr[i]):
            new_neighbors = [
                arr[(i - 1)%L_x][j%L_y],  
                arr[i%L_x][(j + 1)%L_y],  
                arr[(i + 1)%L_x][j%L_y],  
                arr[i%L_x][(j - 1)%L_y]   
            ]
            neighbors[value] = new_neighbors
    return neighbors

#Definiton For Easy Operation Which Show Nearest Neighbors (Tigh-Binding Approximation) Each Sites
HardBCLat = HardBC(lattice)
PerBCLat = PerBC(lattice)


#Lattice Visualization 
fig1, ax1 = plt.subplots()
ax2 = ax1.twiny()
for i in range(L_x):
    for j in range(L_y):
        plt.plot(i, j, 'ro', markersize=7)
ax1.axes.get_xaxis().set_visible(False)        
plt.xticks(x_co)
plt.yticks(y_co)
ax1.invert_yaxis()  
ax1.set_ylabel('Y Coordinates')  
ax2.set_xlabel('X Coordinates')
plt.title('Lattice Structure (Size of Lattice: '+str(L_x)+'x'+str(L_y)+')')
ax1.grid()
ax2.grid()   

#Find Tight-Binding Neighbors (for Periodic Boundary Conditions) with Lattice Visualization
def PlotPerBC(choose_x, choose_y):
    plt.plot(choose_x, choose_y, 'bo', markersize=7, label='Your Choice')
    plt.plot((choose_x+1)%L_x, choose_y%L_y, 'co', markersize=7, label='Periodic Neighbors')
    plt.plot((choose_x-1)%L_x, choose_y%L_y, 'co', markersize=7)
    plt.plot(choose_x%L_x, (choose_y+1)%L_y, 'co', markersize=7)
    plt.plot(choose_x%L_x, (choose_y-1)%L_y, 'co', markersize=7)
    plt.legend(loc="upper left")
    plt.show()
    
#PlotPerBC(15, 15)
#Find Tight-Binding Neighbors (for Hard-Wall Boundary Condition) with Lattice Visualization    
def PlotHardBC(choose_x, choose_y):
    plt.plot(choose_x, choose_y, 'bo', markersize=7, label='Your Choice')
    if choose_x==0 or choose_x==L_x-1 or choose_y==0 or choose_y==L_y-1:
        if choose_x!=0:
            plt.plot(choose_x-1, choose_y, 'co', markersize=7)
        if choose_y!=L_y-1:
            plt.plot(choose_x, choose_y+1, 'co', markersize=7)
        if choose_x!=L_x-1:
            plt.plot(choose_x+1, choose_y, 'co', markersize=7)
        if choose_y!=0:
            plt.plot(choose_x, choose_y-1, 'co', markersize=7)
    else:
        plt.plot(choose_x+1, choose_y, 'co', markersize=7)
        plt.plot(choose_x-1, choose_y, 'co', markersize=7)
        plt.plot(choose_x, choose_y+1, 'co', markersize=7)
        plt.plot(choose_x, choose_y-1, 'co', markersize=7)

#PlotHardBC(0, 0)


#######################################################################################

####Simulation
#Position of inital wave packet
def HMat(alpha):    #Createa  zero matrix with size equal to lattice size
    H = np.zeros((L_x*L_y, L_x*L_y), dtype=complex)    #Iterate in all lattice sites
    for m in range(L_x*L_y):
        for n in range(L_x*L_y):            #Use tight-binding approximation
            if m in HardBCLat[n]:                #Gain positive phase
                if xy[m][0] > xy[n][0]:
                    H[m][n] = -np.exp(1j*2*np.pi*alpha*xy[m][1])                #Gain negative phase
                elif xy[m][0] < xy[n][0]:
                    H[m][n] = -np.exp(-1j*2*np.pi*alpha*xy[m][1])                #No phase
                else:
                    H[m][n]=-1    
    return H



start_lattice_size_1=5
start_lattice_size_2=6
#Normalization of state vector 
psi_0=np.zeros((L_x*L_y,1))/np.sqrt(2)
psi_0[start_lattice_size_1]=1
psi_0[start_lattice_size_2]=1


#Infinitisimal time
delta_t = 0.01#The Value of alpha must be a rational number. For example:
alpha = 1 / 5#Series expansion of the time evolution operator
time_evo=np.identity(L_x*L_y)-1.j*delta_t*HMat(alpha)

#Frame number of animation
frn = 1500 #Create time evolution array
a = np.zeros((L_x*L_y,frn))
for i in range(1,frn):
    #Time evolution operator multiplied by wave function at each step
    psi_0=np.matmul(time_evo, psi_0)    #Collect all wave functions in together
    a[:,[i]]=psi_0

#Create a mesh grid of lattice
X = x_co
Y = y_co
xv, yv = np.meshgrid(X, Y)#Define function to updating plot
def update_plot(frame_number, a, plot):
    #remove the older to new plot 
    plot[0].remove()    #Plot density of wave packet versus x and y corrdinates
    plot[0] = ax.plot_surface(xv, yv, np.reshape((np.abs(a[:,[frame_number]]))**2, (L_x,L_y)), cmap="magma")#Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')#set view position as top
ax.view_init(90,0)#Plot initial wave function
plot = [ax.plot_surface(xv, yv, np.reshape((np.abs(a[:,[0]]))**2, (L_x,L_y)), color='0.75', rstride=1, cstride=1)]
ax.set_zlim(0,1.1)#Create animation
fps=200
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(a, plot), interval=1000/fps)#Save your animation in.gif format
ani.save('animation.gif', writer='pillow', fps=200)