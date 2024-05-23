import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import Bloch
import matplotlib as mpl
from matplotlib import cm

from .main_fun import pulse_x, pulse_z, pulse_x_with_noise
import numpy as np

def plot_excharges(self, out, index, ry_tf = False, j = False):

    labels_axis = ["X","Y","Z","Does not apply"]
    # Grafica del valor esperado:
    if ry_tf == False:
        plt.figure(figsize=(6, 2))
        # Subplot para el valor esperado
        plt.subplot(1, 1, 1)  # 2 filas, 1 columna, primer subplot
        plt.plot(self.tlist*1e9, out.expect[index])
        if j == False:
            plt.title(f'Expected value \n Axis {labels_axis[index]}')
        else:
            plt.title(f'Expected value')
            plt.xlabel('Time [ns]')
            plt.ylabel('Expected value')
            plt.ylim(-1.1, 1.1)
            plt.grid(True)
    else:
        time = list(self.tlist*1e9) + list(self.tlist*1e9) + list(self.tlist*1e9)
        Ntime = len(time)
        tt = np.linspace(0,3*self.tlist[-1]*1e9, Ntime)
        y = list(out[0].expect[index]) + list(out[1].expect[index]) + list(out[2].expect[index])
        plt.figure(figsize=(6, 2))
        # Subplot para el valor esperado
        plt.subplot(1, 1, 1)  # 2 filas, 1 columna, primer subplot
        plt.plot(tt, y)
        plt.title(f'Expected value \n Axis {labels_axis[index]}')
        plt.xlabel('Time [ns]')
        plt.ylabel('Expected value')
        plt.ylim(-1.1, 1.1)
        plt.grid(True)
    plt.show()
    print("\n")
    # Grafica del pulso:
    if self.tf_noise == False:
        if self.Dt == -7:
            # Subplot para el pulso
            plt.figure(figsize=(4, 3))
            plt.subplot(2, 1, 1)  # 2 filas, 1 columna, segundo subplot
            plt.plot(self.tlist*1e9, self.B0 * pulse_z(self.tlist, self.args))
            plt.title('Pulse B0')
            plt.xlabel('Time [ns]')
            plt.ylabel('Amplitude [T]')
            plt.grid(True)
            # Subplot para el pulso
            plt.subplot(2, 1, 2)  # 2 filas, 1 columna, segundo subplot
            plt.plot(self.tlist*1e9, self.B1 * pulse_x(self.tlist, self.args))
            plt.title('Pulse B1')
            plt.xlabel('Time [ns]')
            plt.ylabel('Amplitude [T]')
            plt.grid(True)
        else:
            plt.figure(figsize=(4, 3))
            plt.subplot(2, 1, 1)  # 2 filas, 1 columna, segundo subplot
            plt.plot(self.tlist*1e6, self.J * pulse_z(self.tlist, self.args))
            plt.title('Pulse J')
            plt.xlabel('Time [µs]')
            plt.ylabel('Amplitude [Hz]')
            plt.grid(True)

    else:
        if self.Dt == -7:
            plt.figure(figsize=(4, 3))
            plt.subplot(2, 1, 1)  # 2 filas, 1 columna, segundo subplot
            plt.plot(self.tlist*1e9, self.B0 * pulse_z(self.tlist, self.args))
            plt.title('Pulse B0')
            plt.xlabel('Time [ns]')
            plt.ylabel('Amplitude [T]')
            plt.grid(True)
            # Subplot para el pulso
            plt.subplot(2, 1, 2)  # 2 filas, 1 columna, segundo subplot
            plt.plot(self.tlist*1e9, self.B1 * pulse_x_with_noise(self.tlist, self.args))
            plt.title('Pulso B1')
            plt.xlabel('Time [ns]')
            plt.ylabel('Amplitude [T]')
            plt.grid(True)
        else:
            plt.figure(figsize=(4, 3))
            plt.subplot(2, 1, 1)  # 2 filas, 1 columna, segundo subplot
            plt.plot(self.tlist*1e6, self.J * pulse_z(self.tlist, self.args))
            plt.title('Pulse J')
            plt.xlabel('Time [µs]')
            plt.ylabel('Amplitude [Hz]')
            plt.grid(True)
    # Ajustar el espacio entre los subgráficos para evitar solapamiento
    plt.tight_layout()
    # Mostrar el gráfico
    plt.show()
    
def plot_expect(self, out, ry_tf = False):
    ## create Bloch sphere instance ##
    if ry_tf == False:
      fig = plt.figure(constrained_layout=True)
      ax1 = fig.add_subplot(1, 2, 1, projection='3d')
      b=Bloch()
      b.axes = ax1
      b.fig = fig
      ## normalize colors to times in tlist ##
      nrm = mpl.colors.Normalize(0, self.delt_t*1e9)
      colors = cm.jet(nrm(self.tlist*1e9))
      ## add data points from expectation values ##
      b.add_points([out.expect[0],out.expect[1],out.expect[2]],'m')
      ## customize sphere properties ##
      b.point_color=list(colors)
      b.point_marker=['o']
      b.point_size=[8]
      b.view=[-9,11]
      b.zlpos=[1.1,-1.2]
      b.zlabel=['$\left|0\\right>$','$\left|1\\right>$']
      ## plot sphere ##
      b.render()
      ## Add color bar ##
      sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=nrm)
      sm.set_array([])  # You need to set a dummy array for the right scaling
      cbar = plt.colorbar(sm, ax = ax1, orientation='vertical', shrink=0.5)
      cbar.set_label('Time [ns]')
      plt.show()
    else:
      fig = plt.figure(constrained_layout=True)
      ax1 = fig.add_subplot(1, 2, 1, projection='3d')
      b=Bloch()
      b.axes = ax1
      b.fig = fig
      ## normalize colors to times in tlist ##
      nrm = mpl.colors.Normalize(0, self.delt_t*1e9)
      colors = cm.jet(nrm(self.tlist*1e9))
      ## add data points from expectation values ##
      for i in range(len(out)):
        b.add_points([out[i].expect[0], out[i].expect[1], out[i].expect[2]],'m')
      ## customize sphere properties ##
      b.point_color=list(colors)
      b.point_marker=['o']
      b.point_size=[8]
      b.view=[-9,11]
      b.zlpos=[1.1,-1.2]
      b.zlabel=['$\left|0\\right>$','$\left|1\\right>$']
      ## plot sphere ##
      b.render()
      ## Add color bar ##
      sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=nrm)
      sm.set_array([])  # You need to set a dummy array for the right scaling
      cbar = plt.colorbar(sm, ax = ax1, orientation='vertical', shrink=0.5)
      cbar.set_label('Time [ns]')
      plt.show()