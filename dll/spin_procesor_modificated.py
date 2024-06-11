from .main_fun import traducir_a_positivo
from pennylane import numpy as np
from collections import defaultdict
from qutip import sigmax, sigmay, sigmaz, tensor, mesolve, Qobj, qeye, destroy, Options, QobjEvo
from .main_fun import pulse_x,  pulse_x_with_noise


class Quantum_Spin_Proces:

  def __init__(self, h = 1, gir = 1.760e11, B0 = 10e-3, nf = 4, N_qubits = 1, J = 1e12, tf_noise = False,
                 noise_std = 0.01, B1_offset = 0, n_points_pulse_Ri = 2,
                 n_points_pulse_2Qbits = 2, n_swap = 1, T1 = 1e3, T2 = 1e3,
                 tf_quantum_noise = False, f_rage = 0, save_time_values = False,
                 n_points_pulse_Ri_spl = 1000, save_pulses = False, nstepsolver = 1_000,
                 version_qutip = "5", free_time = 0):
        self.gir = gir
        self.B0 = B0
        self.B1 = B1_offset
        self.Dt = -7
        self.h = h
        self.nf = nf
        self.N_qubits = N_qubits
        self.J = J
        self.tf_noise = tf_noise
        self.noise_std = noise_std
        self.B1_offset = B1_offset
        self.n_points_pulse_Ri = n_points_pulse_Ri
        self.n_points_pulse_2Qbits = n_points_pulse_2Qbits
        self.n_swap = n_swap
        self.global_time = 0
        self.dict_states = {}
        self.pulse_type = defaultdict(list)
        self.T1 = T1
        self.T2 = T2
        self.tf_quantum_noise = tf_quantum_noise
        # OJO
        self.f_rage = f_rage
        # States save
        self.save_tv = save_time_values
        self.states_in_time = []
        self.omegas_x = []
        # spline 
        self.n_points_pulse_Ri_spl = n_points_pulse_Ri_spl
        self.save_pulse = save_pulses
        # solver
        self.nstepsolver = nstepsolver
        # vesion:
        self.version_qutip = version_qutip
        
        self.free_time = free_time

  def Rz(self, alpha, ket_0, q_obj = 0, tf_expect = False):
      # Estados iniciales y qubit objetivo:
      self.q_obj = q_obj
      self.ket_0 = ket_0
      # parametros de compuerta:
      self.ω_x = 0
      self.ω_z = self.gir * self.B0
      alpha  = traducir_a_positivo(alpha)
      self.delt_t = (alpha)/self.ω_z
      self.B1 = 0
      self.O_x = self.gir*(self.B1/2)
      # solucion:
      out = self.Hamiltonian_solve(tf_expect)
      if self.save_tv == True:
        self.states_in_time.append(out.states)
      return out

  def Rx(self, alpha, ket_0, q_obj = 0, tf_expect = False):
      # Estados iniciales y qubit objetivo:
      self.q_obj = q_obj
      self.ket_0 = ket_0
      # parametros de compuerta:
      self.ω_x = self.gir * self.B0 + self.f_rage # OJO
      self.ω_z = self.gir * self.B0 
      self.delt_t = (np.abs(alpha)*self.nf)/self.ω_x
      self.B1 = (alpha * 2)/(self.gir * self.delt_t)
      self.O_x = self.gir*(self.B1/2)
      # solucion:
      out = self.Hamiltonian_solve(tf_expect)
      if self.save_tv == True:
        self.states_in_time.append(out.states)
        self.omegas_x.append(self.ω_x)
      return out

  def Ry(self, alpha, ket_0, q_obj = 0, tf_expect = False):
      out_1 = self.Rx(np.pi/2, ket_0, q_obj=q_obj, tf_expect = False)
      end_state_1 = out_1.states[-1]
      #self.states_in_time.append(out_1.states)
      out_2 = self.Rz(alpha, end_state_1,q_obj=q_obj, tf_expect = False)
      end_state_2 = out_2.states[-1]
      #self.states_in_time.append(out_2.states)
      out_3 = self.Rx(-np.pi/2, end_state_2,q_obj=q_obj, tf_expect = False)
      #self.states_in_time.append(out_3.states)
      if tf_expect == True:
        out_1_exp = self.Rx(np.pi/2, ket_0, q_obj=q_obj, tf_expect = True)
        out_2_exp = self.Rz(alpha, end_state_1,q_obj=q_obj, tf_expect = True)
        out_3_exp = self.Rx(-np.pi/2, end_state_2,q_obj=q_obj, tf_expect = True)
        out = [out_1_exp, out_2_exp, out_3_exp]
      else:
        out = out_3
      return out

  def H(self, ket_0, alpha = np.pi/2, q_obj = 0, tf_expect = False):
      out_1 = self.Rz(np.pi/2, ket_0, q_obj=q_obj, tf_expect = False)
      end_state_1 = out_1.states[-1]
      out_2 = self.Rx(alpha, end_state_1,q_obj=q_obj, tf_expect = False)
      end_state_2 = out_2.states[-1]
      out_3 = self.Rz(np.pi/2, end_state_2,q_obj=q_obj, tf_expect = False)
      if tf_expect == True:
        out_1_exp = self.Rz(np.pi/2, ket_0, q_obj=q_obj, tf_expect = True)
        out_2_exp = self.Rx(alpha, end_state_1,q_obj=q_obj, tf_expect = True)
        out_3_exp = self.Rz(np.pi/2, end_state_2,q_obj=q_obj, tf_expect = True)
        out = [out_1_exp, out_2_exp, out_3_exp]
      else:
        out = out_3
      return out

  def SWAP(self, ket_0, measure_op, q_obj = [0,1], tf_expectt = False):
      self.q_obj = q_obj
      self.ket_0 = ket_0
      self.Dt = np.pi/(self.J*self.n_swap)
      #self.measure = measure_op
      #self.qobj = Qobj(self.measure, dims=[[2,2],[2,2]])
      self.out = self.Hamiltonian_solve_excharge(tf_expectt)
      if self.save_tv == True:
        self.states_in_time.append(self.out.states)
      return self.out

  def sqrt_SWAP(self, ket_0, measure_op, q_obj = [0,1], tf_expectt = False):
      self.q_obj = q_obj
      self.ket_0 = ket_0
      self.Dt = (np.pi/(2*self.J*self.n_swap))
      #self.measure = measure_op
      #self.qobj = Qobj(self.measure, dims=[[2,2],[2,2]])
      self.out = self.Hamiltonian_solve_excharge(tf_expectt)
      if self.save_tv == True:
        self.states_in_time.append(self.out.states) 
      return self.out

  def CNOT(self, ket_0, measure_op, q_obj = [0,1], tf_expectt = False):
      self.q_obj = q_obj
      q_control , q_target = q_obj
      state_1 = self.Ry(np.pi/2, ket_0, q_obj = q_target, tf_expect = False).states[-1]
      state_2 = self.sqrt_SWAP(state_1, [], q_obj = q_obj, tf_expectt = False).states[-1]
      state_3 = self.Rz(np.pi, state_2, q_obj = q_control, tf_expect = False).states[-1]
      state_4 = self.sqrt_SWAP(state_3, [], q_obj= q_obj, tf_expectt = False).states[-1]
      state_5 = self.Rz(-np.pi/2, state_4, q_obj = q_control, tf_expect = False).states[-1]
      state_6 = self.Rz(-np.pi/2, state_5, q_obj = q_target, tf_expect = False).states[-1]
      state_7 = self.Ry(-np.pi/2, state_6, q_obj = q_target, tf_expect = False).states[-1]
      return state_7

  def CZ(self, ket_0, measure_op, q_obj = [0,1], tf_expectt = False):
      self.q_obj = q_obj
      q_control , q_target = q_obj
      state_2 = self.sqrt_SWAP(ket_0, [], q_obj = q_obj, tf_expectt = False).states[-1]
      state_3 = self.Rz(-np.pi, state_2, q_obj = q_control, tf_expect = False).states[-1]
      state_4 = self.sqrt_SWAP(state_3, [], q_obj = q_obj, tf_expectt = False).states[-1]
      state_5 = self.Rz(np.pi/2, state_4, q_obj = q_control, tf_expect = False).states[-1]
      state_6 = self.Rz(-np.pi/2, state_5, q_obj = q_target, tf_expect = False).states[-1]
      return state_6
  
  def init_H_1q_and_opt(self, tf_expect = False):
      # Constantes del Hamiltoniano:
      h0_constant = - (self.h/2) * (self.ω_x)
      h1_constant =   (self.h/2) * (self.ω_z)
      h2_constant =   (self.h/2) * (self.O_x)
      
      # Apli rotations in individual qubits:
      apply_qbit_z = []
      apply_qbit_x = []
      for i in range(self.N_qubits):
        if i == self.q_obj:
          apply_qbit_z.append(sigmaz())
          apply_qbit_x.append(sigmax())
        else:
          apply_qbit_z.append(qeye(2))
          apply_qbit_x.append(qeye(2))
      
      # Terminos constantes del hamiltoniano:
      self.H0 = h0_constant * tensor(*apply_qbit_z)
      self.H1 = h1_constant * tensor(*apply_qbit_z)
      self.H2 = h2_constant * tensor(*apply_qbit_x)

      # Expectation values:
      if tf_expect:
          if self.N_qubits > 1:
            apply_qbit_e_ops = []
            for i in range(self.N_qubits):
              if i == self.q_obj:
                apply_qbit_e_ops.append(sigmaz())
              else:
                apply_qbit_e_ops.append(qeye(2))
            e_ops = [tensor(*apply_qbit_e_ops)]
          else:
            e_ops = [sigmax(), sigmay(), sigmaz()]
      else:
          e_ops = []
      # Expectations ops:
      self.e_ops = e_ops

      # Colpase operators:
      if self.tf_quantum_noise:
          apply_qbit_c_ops_1 = []
          apply_qbit_c_ops_2 = []
          for i in range(self.N_qubits):
            if i == self.q_obj:
              a = destroy(2)
              T2_star = 1/((1/self.T2) - (1/(2*self.T1)))
              c1 = a/(np.sqrt(self.T1))
              c2 = a.dag()*a*np.sqrt(2/T2_star)
              print(T2_star)
              apply_qbit_c_ops_1.append(c1)
              apply_qbit_c_ops_2.append(c2)
            else:
              apply_qbit_c_ops_1.append(qeye(2))
              apply_qbit_c_ops_2.append(qeye(2))
          c_ops = [tensor(*apply_qbit_c_ops_1), tensor(*apply_qbit_c_ops_2)]
      else:
        c_ops = []
      self.c_ops = c_ops

  def Hamiltonian_solve(self, tf_expect = False):
      # Parameters of hamiltonian:
      self.init_H_1q_and_opt(tf_expect)
      # Correccion de desviacion estandar:
      if self.B1 != 0:
        dv = self.noise_std/abs(self.B1)
      else:
        dv = 0
      self.args = { "t_init": 0, "t_final": self.delt_t, "std_noise": dv}
      # Aca controlamos el tiempo de simulacion independiente del tiempo spline:
      self.tlist  = np.linspace(0, self.delt_t + self.free_time, self.n_points_pulse_Ri)
      # Hamiltonian
      if self.tf_noise == False:
        H = [self.H1, [self.H2 + self.H0, pulse_x]]
        Noise_x = ""
      else:
        # Ruido coherente
        H, Noise_x = self.spline_from_version()
      # Guardar pulso:
      if self.save_pulse == True:
        t_actual = self.global_time
        t_final = self.global_time + self.delt_t
        self.pulse_type[self.q_obj].append({
            "Type_pulse": "Unitary",
            "B0": self.B0,
            "B1": self.B1,
            "Delt_t": self.delt_t,
            "t_i": t_actual,
            "t_f": t_final,
            "Noise": Noise_x
        })
      # Actualizar tiempo:
      self.global_time += self.delt_t
      # opciones:
      options = Options(nsteps=self.nstepsolver)
      # solve ME:
      output_dm = mesolve(H, self.ket_0, self.tlist, self.c_ops, self.e_ops, self.args, options = options)
      return output_dm
  
  def spline_from_version(self):
      if self.version_qutip == "4":
        # Version <= 4
        self.tlist_spline  = np.linspace(0, self.delt_t, self.n_points_pulse_Ri_spl)
        noise_x = pulse_x_with_noise(self.tlist_spline, self.args)
        S_x = Cubic_Spline(self.tlist_spline[0], self.tlist_spline[-1], noise_x)
        H = [self.H0 + self.H1, [self.H2, S_x]]
      else:
        # Version >= 5
        times_spl = np.linspace(0, self.delt_t, self.n_points_pulse_Ri_spl)
        noise_x = pulse_x_with_noise(times_spl, self.args).flatten()
        H = QobjEvo([self.H0 + self.H1, [self.H2, noise_x]], tlist=times_spl)
      return H, noise_x
     

  def Hamiltonian_solve_excharge(self, tf_expect = False):
      # parametros del hamiltoniano:
      delt_t = self.Dt
      Si, Sj = self.q_obj
      apply_qbit_z = []
      apply_qbit_x = []
      apply_qbit_y = []
      for i in range(self.N_qubits):
        if i == Si or i == Sj:
          apply_qbit_z.append(sigmaz())
          apply_qbit_x.append(sigmax())
          apply_qbit_y.append(sigmay())
        else:
          apply_qbit_z.append(qeye(2))
          apply_qbit_x.append(qeye(2))
          apply_qbit_y.append(qeye(2))

      # Hamiltonian:
      H = ((self.J * self.h**2)/4)*(tensor(*apply_qbit_x) + tensor(*apply_qbit_y) + tensor(*apply_qbit_z))
      
      # Hamiltonian pulse:
      h_t = [H, pulse_x]

      # Correccion de desviacion estandar:
      if self.B1 != 0:
        dv = self.noise_std/abs(self.B1)
      else:
        dv = 0
      
      # Argumentos del pulso:
      self.args = { "t_init": 0, "t_final": delt_t, "std_noise": dv}

      # Expectation values:
      if tf_expect:
        qobj = Qobj(self.measure, dims=[[2,2],[2,2]])
        e_ops = [qobj]
      else:
        e_ops = []
      
      # Colpase operators:
      c_ops = []

      # times:
      self.tlist  = np.linspace(0, delt_t, self.n_points_pulse_2Qbits)

      if self.save_pulse == True:
        # Save pulses
        t_actual = self.global_time
        t_final = self.global_time + delt_t
        self.pulse_type[f'I_{self.q_obj[0]}-{self.q_obj[1]}'].append(
                          {
                              "Type_pulse": "Two_Qubits",
                              "Q_bits_target":self.q_obj,
                              "J": self.J,
                              "Delt_t": self.Dt,
                              "t_i": t_actual,
                              "t_f": t_final,
                              "Noise": ""
                            })
      self.global_time += self.Dt
      self.output = mesolve(h_t, self.ket_0, self.tlist, c_ops, e_ops, self.args)
      return self.output

