import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.backend_bases.__file__)
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



class ClickDraw(object):
    def __init__(self, ax):
        self.types = [0, 1, 2]
        self.colors = ["g", "r", "y"]
        self.shapes = ["o", "x", "^"]
        self.ax = ax
        self.fig = ax.figure
        self.xy = {i : [] for i in self.types}
        self.points = {i : ax.scatter([], [], c=self.colors[i], 
                       marker=self.shapes[i], picker=20, s=200, zorder =100*i) \
                       for i in self.types}
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
    
    def on_click(self, event):
        if not event.inaxes:
            return
        if event.button is MouseButton.LEFT:
            kind = self.types[0]
        elif event.button is MouseButton.RIGHT:
            kind = self.types[1]
        elif event.button is MouseButton.MIDDLE:
            kind = self.types[2]
        self.xy[kind].append([event.xdata, event.ydata])
        self.points[kind].set_offsets(self.xy[kind]) # ?
        self.ax.draw_artist(self.points[kind])
        self.fig.canvas.blit(self.ax.bbox)
        # ~ self.points[kind].set_zorder(-100*kind + 30)
        # ~ self.ax.plot([0,1],[0,1], color="k")
            
            
    def separation(self):
        """
        Here we do the logarithmic regression and plot the separation 
        line.
        """
        
        
    def show(self):
        plt.show()
        
        
        
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
