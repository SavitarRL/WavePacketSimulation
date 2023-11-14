import numpy as np
import matplotlib.pyplot as plt

#from tight_binding_approximation import *

#Create 2-D Lattice. Numbers Represents Each Lattice Sites
#Lattice Size

class Lattice:
    
    def create_lattice(self, L_x, L_y):
        lat = np.arange(L_x*L_y).reshape(L_x,L_y)
        array = np.append(lat, lat, axis=0)
        lattice = array[:-L_x, :]
        return lattice
    
    
    #Respectively Site Coordinates on Lattice
    # def site_coordinates(self):
    #     arr = []
    #     x_co = np.arange(self.L_x)
    #     y_co = np.arange(self.L_y)
    #     for j in range(len(x_co)):
    #         for i in range(len(y_co)):
    #             arr=np.append(arr, [x_co[i], y_co[j]])
    #     xy = arr.reshape((self.L_x*self.L_y,2))
    #     return arr, xy
    
    def __init__(self, L_x, L_y):
        self.L_x = L_x
        self.L_y = L_y

        self.arr = []
        self.x_co = np.arange(self.L_x)
        self.y_co = np.arange(self.L_y)

        for j in range(len(self.x_co)):
            for i in range(len(self.y_co)):
                self.arr=np.append(self.arr, [self.x_co[i], self.y_co[j]])
        self.xy = self.arr.reshape((self.L_x*self.L_y,2))
        
        # self.arr = self.site_coordinates()[0]
        # self.xy = self.site_coordinates()[1]
       
        self.lattice = self.create_lattice(L_x,L_y)
    

#Find neighbours Each Sites with Tight-Binding Approximation (But for Hard-Wall Boundary Conditions) 
class BoundaryConditions:
    
    # eats in arr of type lattice
    def hard_new_neighbours(self,lat,i,j):
        if i == 0 or i == len(lat) - 1 or j == 0 or j == len(lat[i]) - 1:
            new_neighbours = []
            if i != 0:
                new_neighbours.append(lat[i - 1][j])  
            if j != len(lat[i]) - 1:
                new_neighbours.append(lat[i][j + 1]) 
            if i != len(lat) - 1:
                new_neighbours.append(lat[i + 1][j])  
            if j != 0:
                new_neighbours.append(lat[i][j - 1])
        else:
            new_neighbours = [
                lat[i - 1][j],  
                lat[i][j + 1],  
                lat[i + 1][j],  
                lat[i][j - 1]   
            ]
        return new_neighbours

    
    def HardBC(self,lat):
        neighbours = {}
        for i in range(len(lat)):
            for j, value in enumerate(lat[i]):
                new_neighbours = self.hard_new_neighbours(lat,i,j)
                neighbours[value] = new_neighbours
        return neighbours

    #Find neighbours Each Sites with Tight-Binding Approximation (But for Periodic Boundary Conditions) 
    def periodic_new_neighbours(self,lat,i,j,L_x,L_y):
        new_neighbours = [
                    lat[(i - 1)%L_x][j%L_y],  
                    lat[i%L_x][(j + 1)%L_y],  
                    lat[(i + 1)%L_x][j%L_y],  
                    lat[i%L_x][(j - 1)%L_y]   
                ]
        return new_neighbours
    
    def PerBC(self,lat, L_x, L_y):
        neighbours = {}
        for i in range(len(lat)):
            for j, value in enumerate(lat[i]):
                new_neighbours = self.periodic_new_neighbours(lat,i,j,L_x,L_y)
                neighbours[value] = new_neighbours
        return neighbours

class Visualise(Lattice):
    
    def __init__(self, L_x, L_y):
        super().__init__(L_x, L_y)

    def plot_lattice_points(self, points = "ro", markersize = 7):
        for i in range(self.L_x):
            for j in range(self.L_y):
                plt.plot(i, j, points , markersize = markersize)
        
    def plot_lattice_structure(self):
        
        fig1, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        
        self.plot_lattice_points(self, points = "ro", markersize = 7)
        
        ax1.axes.get_xaxis().set_visible(False)        
        plt.xticks(self.x_co)
        plt.yticks(self.y_co)
        ax1.invert_yaxis()  
        ax1.set_ylabel('Y Coordinates')  
        ax2.set_xlabel('X Coordinates')
        plt.title('Lattice Structure (Size of Lattice: '+str(self.L_x)+'x'+str(self.L_y)+')')
        ax1.grid()
        ax2.grid()  

    def PlotPerBC(self, choose_x, choose_y):

        L_x = self.L_x
        L_y = self.L_y

        plt.plot(choose_x, choose_y, 'bo', markersize=7, label='Your Choice')
        plt.plot((choose_x+1)%L_x, choose_y%L_y, 'co', markersize=7, label='Periodic Neighbors')
        plt.plot((choose_x-1)%L_x, choose_y%L_y, 'co', markersize=7)
        plt.plot(choose_x%L_x, (choose_y+1)%L_y, 'co', markersize=7)
        plt.plot(choose_x%L_x, (choose_y-1)%L_y, 'co', markersize=7)
        plt.legend(loc="upper left")
        plt.show()

    def PlotHardBC(self, choose_x, choose_y):

        L_x = self.L_x
        L_y = self.L_y
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