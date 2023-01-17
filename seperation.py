import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib
"""
t = np.arange(0, 1, 0.01)
s = np.sin(2*np.pi*t)
fig, ax_1 = plt.subplots()
ax_1.plot(s,t)

def on_move(event):
    if event.inaxes:
        print(f"data coords {event.xdata} {event.ydata}",
              f"pixel coords {event.x} {event.y}")

def on_click(event, ax):
    if event.button is MouseButton.LEFT:
        print("disconnecting callback")
        plt.disconnect(binding_id)
        
"""

def main():
    fig, ax_1 = plt.subplots()

    ClickDraw(ax_1).show()


def sigmoid(x):
    return 1/(1+np.exp(-x))


class ClickDraw(object):
    def __init__(self, ax):
        self.types = [0, 1, 2]
        self.colors = ["g", "r", "y"]
        self.shapes = ["o", "x", "^"]
        self.ax = ax
        self.fig = ax.figure
        self.xy = {}
        # ~ self.points = {i : ax.scatter([], [], c=self.colors[i], 
                       # ~ marker=self.shapes[i], picker=20, s=200, zorder =100*i) \
                       # ~ for i in self.types}
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
    
    def on_click(self, event):
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
        w = np.array([1,2])
        
        
            
        if kind in [0, 1]:
            self.ax.draw_artist(self.ax.scatter(event.xdata,event.ydata,
                            color=self.colors[kind], s=200, 
                            marker=self.shapes[kind]))
            self.fig.canvas.blit(self.ax.bbox)
            if 0 in self.xy and 1 in self.xy:
                self.separation()
        else:
            if 0 in self.xy and 1 in self.xy:
                self.separation([event.xdata, event.ydata])

    def cost(self, w, b):
        m = len(self.xy[0]) + len(self.xy[1])
        z_0 = np.matmul(self.xy[0], w) + b  # wx^i + b for all i
        z_1 = np.matmul(self.xy[1], w) + b
        c = 1/m*sum(-np.log(1-sigmoid(z_0)))-sum(np.log(sigmoid(z_1)))
        dc_dw = 1/m*(np.matmul(sigmoid(z_0),self.xy[0])+\
                     np.matmul(sigmoid(z_1)-1,self.xy[1]))
        dc_db = 1/m*(sum(sigmoid(z_0))+sum(sigmoid(z_1)-1))
        print(dc_dw)
        return c, dc_dw, dc_db
            
    def separation(self, datapoint = None):
        """
        Here we do the logarithmic regression and plot the separation 
        line.
        """
        maxit = 1000
        alpha = 10
        w = np.array([0, 0])
        b = 1
        if 0 not in self.xy or 1 not in self.xy:
            raise ValueError("There are only data points of one kind.")
        
        c = 100
        while c>0.01:
            c, dc_dw, dc_db = self.cost(w, b)
            print(c)
            w = w - alpha*dc_dw
            b = b - alpha*dc_db
        self.ax.plot([0,1], [-b/w[1], -w[0]/w[1]-b/w[1]], "-")
        print(b/w[1],w[0]/w[1]+b/w[1] )
        """
        if datapoint == None:
            return 0
        else:
            return 0,0
        """
        
        
        
    def show(self):
        plt.show()
        
        
        
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
