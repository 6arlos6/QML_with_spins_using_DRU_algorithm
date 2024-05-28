import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import Bloch
import matplotlib as mpl
from matplotlib import cm
import os
import numpy as np

from .main_fun import pulse_x, pulse_z, pulse_x_with_noise



# =======================================================================

def plot_expect(self, out, ry_tf = False,
                path_to_save_img = os.path.join("results_of_test","test_2"),
                index = 0):
    
    label_axis = ["X", "Y", "Z"]
    
    fig = plt.figure(constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    b = Bloch()
    b.axes = ax1
    b.fig = fig
    ## normalize colors to times in tlist ##
    nrm = mpl.colors.Normalize(0, self.delt_t*1e9)
    colors = cm.jet(nrm(self.tlist*1e9))
    ## add data points from expectation values ##
    if ry_tf == True:
        for i in range(len(out)):
            b.add_points([out[i].expect[0], out[i].expect[1], out[i].expect[2]],'m')
    else:
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
    path_to_save_img_file = os.path.join(path_to_save_img,f"R{label_axis[index]}_withRWA_expectation_in_Bloch.pdf")
    plt.savefig(path_to_save_img_file , format='pdf', bbox_inches = 'tight')
    plt.show()

# ==============================================================

label_axis = [r"$\langle \sigma_x \rangle$",
              r"$\langle \sigma_y \rangle$",
            r"$\langle \sigma_z \rangle$"]

label_axis_file_name = ["x","y","z"]

def plot_excharges(self, out, ry_tf = False,
                   index = 0,
                   path_to_save_img = os.path.join("results_of_test","test_2"),
                   name_file = None, nick_name="Rx"):
    
    # Sino se ingresa nombre poner uno:
    if name_file is None:
        name_file = f"Expectation_val_{label_axis_file_name[index]}.pdf"
    name_file = nick_name + "_" + name_file
    
    # Grafica del valor esperado:
    if ry_tf == False:
        # caso distinto a Ry:
        plt.figure(figsize=(6, 2))
        plt.plot(self.tlist*1e9, out.expect[index])
        plt.xlabel('Time [ns]')
        plt.ylabel(label_axis[index])
        plt.ylim(-1.1, 1.1)
        plt.grid(True)
    else:
        # caso para Ry:
        time = list(self.tlist*1e9) + list(self.tlist*1e9) + list(self.tlist*1e9)
        Ntime = len(time)
        tt = np.linspace(0,3*self.tlist[-1]*1e9, Ntime)
        y = list(out[0].expect[index]) + list(out[1].expect[index]) + list(out[2].expect[index])
        plt.figure(figsize=(6, 2))
        plt.plot(tt, y)
        plt.xlabel('Time [ns]')
        plt.ylabel(label_axis[index])
        plt.ylim(-1.1, 1.1)
        plt.grid(True)
    # Export and show:
    path_to_save_img_file = os.path.join(path_to_save_img, name_file)
    plt.savefig(path_to_save_img_file  , format='pdf', bbox_inches = 'tight')
    plt.show()
    
    print("\n")
    
    # Grafica del pulso:
    # ========================================
    # Detectar si es pulso de interaccion J...
    if self.Dt == -7:
        # Sino es pulso de interaccion J,
        # detectar si viene con ruido coherente...
        if self.tf_noise == False:
            # Sin ruido coherente:
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
            # Con ruido coherente:
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
        # Si es pulso cuadrado:
        plt.figure(figsize=(4, 3))
        plt.subplot(2, 1, 1)  # 2 filas, 1 columna, segundo subplot
        plt.plot(self.tlist*1e9, self.J * pulse_z(self.tlist, self.args))
        plt.title('Pulse J')
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [Hz]')
        plt.grid(True)
        
    plt.tight_layout()
    # Mostrar el gráfico
    path_to_save_img_file = os.path.join(path_to_save_img, "pulse_" + name_file)
    plt.savefig(path_to_save_img_file  , format='pdf', bbox_inches = 'tight')
    plt.show()

