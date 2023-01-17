"""
This litle program uses logistic regression to split the unit square 
into two regions A and B based on the user's choice of points of type 
A and B.
Just click on the plot to add a point (right click for type A, left 
click for type B) and observe how the separation line changes.
"""
import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib


def main():
    fig, ax_1 = plt.subplots()
    ClickDraw(ax_1).show()


def sigmoid(x):
    return 1/(1+np.exp(-x))


class ClickDraw(object):
    def __init__(self, ax):
        """
        Initialise
        """
        self.types = [0, 1, 2]
        self.colors = ["g", "r", "y"]
        self.shapes = ["o", "x", "^"]
        self.ax = ax
        self.fig = ax.figure
        #self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.xy = {}
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
    
    def on_click(self, event):
        """
        Draws a dot of the type specified by the mouse button used and 
        calls seperation to determine the best seperation line between 
        the two types.
        """
        self.ax.set_xlim(0,1)
        self.ax.set_ylim(0,1)
        if not event.inaxes:
            return
        if event.button is MouseButton.LEFT:
            kind = self.types[0]
        elif event.button is MouseButton.RIGHT:
            kind = self.types[1]
        elif event.button is MouseButton.MIDDLE:
            kind = self.types[2]
        
        if kind not in self.xy:
            self.xy[kind] = np.array([[event.xdata, event.ydata]])
        else:
            self.xy[kind] = np.insert(self.xy[kind], -1, 
                                      [[event.xdata, event.ydata]],
                                      axis=0)
        
        try:
            self.fig.canvas.restore_region(self.background)
        except AttributeError:
            pass
        
        if kind in [0, 1]:
            self.ax.draw_artist(self.ax.scatter(event.xdata,event.ydata,
                            color=self.colors[kind], s=200, 
                            marker=self.shapes[kind]))
            self.fig.canvas.blit(self.ax.bbox)
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            try:
                self.separation()
            except ValueError:
                pass

    def cost(self, w, b):
        """
        cost function of the logistic regression model.
        """
        m = len(self.xy[0]) + len(self.xy[1])
        z_0 = np.matmul(self.xy[0], w) + b  
        z_1 = np.matmul(self.xy[1], w) + b
        c = 1/m*(sum(-np.log(1-sigmoid(z_0)))-sum(np.log(sigmoid(z_1))))
        dc_dw = 1/m*(np.matmul(sigmoid(z_0),self.xy[0])+\
                     np.matmul(sigmoid(z_1)-1,self.xy[1]))
        dc_db = 1/m*(sum(sigmoid(z_0))+sum(sigmoid(z_1)-1))
        return c, dc_dw, dc_db
            
    def separation(self, datapoint = None):
        """
        Here we do the logistic regression and plot the separation 
        line.
        """
        maxit = 10000
        alpha = 10
        w = np.array([1, 1])
        b = 1
        if 0 not in self.xy or 1 not in self.xy:
            raise ValueError("There are only data points of one kind.")
        
        dc_dw = [1]
        ite = 0
        while np.linalg.norm(dc_dw)> 1e-3 and ite < maxit:
            ite += 1
            c, dc_dw, dc_db = self.cost(w, b)
            w = w - alpha*dc_dw
            b = b - alpha*dc_db
        self.sepline = list(self.ax.errorbar([0,1], [-b/w[1],
                                             -w[0]/w[1]-b/w[1]], 
                                             ls="--", color="k"))[0]
        self.ax.draw_artist(self.sepline)
        self.fig.canvas.blit(self.ax.bbox)
        
        
    def show(self):
        plt.show()
        
        
        
if __name__ == "__main__":
    print(__doc__)
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
