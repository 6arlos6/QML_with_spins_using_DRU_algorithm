from pennylane import numpy as np
from qutip import basis
import copy
from .spin_procesor_modificated import Quantum_Spin_Proces

      
class CompositeGateProcessor(Quantum_Spin_Proces):
  # se esta bajando con esto
  def __init__(self, nf=4, h=1, gir =1.760e11, B0=10e-3, N_qubits=1, J=1e12, tf_noise=False,
                 noise_std=0.01, B1_offset=0, n_points_pulse_Ri=100, n_points_pulse_2Qbits=100,
                n_swap=1, T1=1e3, T2 = 1e3, tf_quantum_noise=False, f_rage = 0,
                save_time_values = False, n_points_pulse_Ri_spl = 1000, save_pulses = False, nstepsolver = 1_000,
                version_qutip = "5", free_time = 0,  ket_dru_0 = basis(2, 0),  save_intra_states = False):
                # Inicializador del procesdaor:
                super().__init__(h, gir, B0, nf, N_qubits, J, tf_noise,
                 noise_std, B1_offset, n_points_pulse_Ri, n_points_pulse_2Qbits,
                 n_swap, T1, T2, tf_quantum_noise, f_rage, save_time_values, n_points_pulse_Ri_spl,
                 save_pulses, nstepsolver, version_qutip, free_time)
                # Estado inicial:
                self.ket_dru_0 = ket_dru_0
                self.ket_dru_0_to_reset = ket_dru_0
                self.save_intra_states = save_intra_states
                
  def reset_ket_0(self):
    self.ket_dru_0 = self.ket_dru_0_to_reset
    self.global_time = 0
    
  def qcircuit_DRU_1_Qubit(self, params, x, bias=None, entanglement=False, tf_expect = False):
    for i,p in enumerate(params):
        arg = np.multiply(p,x) + bias[i]
        arg1, arg2, arg3 = arg
        self.ket_dru_0 = self.RzRyRz(float(arg3), float(arg1), float(arg2), self.ket_dru_0, tf_expect = tf_expect)
        if self.save_intra_states == True:
          self.dict_states[f"ket_1_qubits_{i}"] = self.ket_dru_0.full()
    return self.ket_dru_0

  def qcircuit_DRU_2_Qubit(self, params, x, bias=None, entanglement=False):
    # def layers in function of parameters:
    n_layer = len(params) // 2
    for i in range(n_layer):
      # parameters:
      arg_1 = np.multiply(params[i],x) + bias[i]
      arg_1_1, arg_1_2, arg_1_3 = arg_1
      arg_2 = np.multiply(params[i + n_layer],x) + bias[i + n_layer]
      arg_2_1, arg_2_2, arg_2_3 = arg_2

      tg1 = copy.copy(self.global_time)
      self.ket_dru_0 = self.RzRyRz(float(arg_1_1), float(arg_1_2), float(arg_1_3), self.ket_dru_0, q_obj=0, tf_expect=False)
      tg2 = copy.copy(self.global_time)
      self.global_time = copy.copy(tg1) # Reset global time a tg1
      self.ket_dru_0 = self.RzRyRz(float(arg_2_1), float(arg_2_2), float(arg_2_3), self.ket_dru_0, q_obj=1, tf_expect=False)
      tg3 = copy.copy(self.global_time)

      # Avanzar con el global time mas grande:
      list_tg = [tg2, tg3]
      self.global_time = max(list_tg)

      if self.save_intra_states == True:
        self.dict_states[f"ket_2_qubits_{i}"] = self.ket_dru_0.full()
      if entanglement == True:
        self.ket_dru_0 = self.CZ(self.ket_dru_0, [], q_obj = [0,1])
        if self.save_intra_states == True:
          self.dict_states[f"ket_2_qubits_entanglement_{i}"] = self.ket_dru_0.full()
    return self.ket_dru_0

  def qcircuit_DRU_4_Qubit(self, params, x, bias=None, entanglement=False):
    # Numero de layers:
    n_layer = len(params) // 4

    for i in range(n_layer):
      # Parameters:
      arg_1 = np.multiply(params[i],x) + bias[i]
      arg_2 = np.multiply(params[i + n_layer],x) + bias[i + n_layer]
      arg_3 = np.multiply(params[i + 2*n_layer],x) + bias[i + 2*n_layer]
      arg_4 = np.multiply(params[i + 3*n_layer],x) + bias[i + 3*n_layer]
      arg_1_1, arg_1_2, arg_1_3 = arg_1
      arg_2_1, arg_2_2, arg_2_3 = arg_2
      arg_3_1, arg_3_2, arg_3_3 = arg_3
      arg_4_1, arg_4_2, arg_4_3 = arg_4

      tg1 = copy.copy(self.global_time)
      self.ket_dru_0 = self.RzRyRz(float(arg_1_1), float(arg_1_2), float(arg_1_3), self.ket_dru_0, q_obj=0, tf_expect=False)
      tg2 = copy.copy(self.global_time)
      self.global_time = copy.copy(tg1)
      self.ket_dru_0 = self.RzRyRz(float(arg_2_1), float(arg_2_2), float(arg_2_3), self.ket_dru_0, q_obj=1, tf_expect=False)
      tg3 = copy.copy(self.global_time)
      self.global_time = copy.copy(tg1)
      self.ket_dru_0 = self.RzRyRz(float(arg_3_1), float(arg_3_2), float(arg_3_3), self.ket_dru_0, q_obj=2, tf_expect=False)
      tg4 = copy.copy(self.global_time)
      self.global_time = copy.copy(tg1)
      self.ket_dru_0 = self.RzRyRz(float(arg_4_1), float(arg_4_2), float(arg_4_3), self.ket_dru_0, q_obj=3, tf_expect=False)
      tg5 = copy.copy(self.global_time)

      # Avanzar con el global time mas grande:
      list_tg = [tg2, tg3, tg4, tg5]
      self.global_time = max(list_tg)
      if self.save_intra_states == True:
        self.dict_states[f"ket_4_qubits_{i}"] = self.ket_dru_0.full()
      if entanglement == True:
        if i % 2 == 0 and i < n_layer-1:
          # par:
          self.ket_dru_0 = self.CZ(self.ket_dru_0, [], q_obj = [0,1])
          self.ket_dru_0 = self.CZ(self.ket_dru_0, [], q_obj = [2,3])
          if self.save_intra_states == True:
            self.dict_states[f"ket_4_qubits_entanglement_par{i}"] = self.ket_dru_0.full()
        elif i < n_layer-1:
          # impar:
          # def CZ(self, ket_0, measure_op, q_obj = [0,1], tf_expectt = False):
          # 👁 OJITOOOOO
          self.ket_dru_0 = self.CZ(self.ket_dru_0, [], q_obj = [1,2], tf_expectt =False)
          self.ket_dru_0 = self.CZ_4(self.ket_dru_0, q_obj=[0,3], tf_expect=False)
          if self.save_intra_states == True:
            self.dict_states[f"ket_4_qubits_entanglement_impar{i}"] = self.ket_dru_0.full()
    return self.ket_dru_0

  def RzRyRz(self, alpha, beta, gamma, ket_0, q_obj=0, tf_expect=False):
        
        rz_result = self.Rz(alpha, ket_0, q_obj=q_obj, tf_expect=False)
        end_state_rz = rz_result.states[-1]
        rx_result = self.Ry(beta, end_state_rz, q_obj=q_obj, tf_expect=False)
        end_state_rx = rx_result.states[-1]
        ry_result = self.Rz(gamma, end_state_rx, q_obj=q_obj, tf_expect=False)
        end_state_ry = ry_result.states[-1]

        if tf_expect:
            rz_result_exp = self.Rz(alpha, ket_0, q_obj=q_obj, tf_expect=True)
            rx_result_exp = self.Ry(beta, end_state_rz, q_obj=q_obj, tf_expect=True)
            ry_result_exp = self.Rz(gamma, end_state_rx, q_obj=q_obj, tf_expect=True)
            return [rz_result_exp, rx_result_exp, ry_result_exp]
        else:
            return end_state_ry

  def CZ_4(self, ket_0, q_obj=[0,3], tf_expect=True):
    q_objetivo, q_target = q_obj
    state_1 = self.H(ket_0, q_obj = q_target, tf_expect = tf_expect).states[-1]
    state_2 = self.CNOT_4(state_1, tf_expect = tf_expect)
    state_3 = self.H(state_2, q_obj = q_target, tf_expect = tf_expect).states[-1]
    return state_3

  def CNOT_4(self, ket_0, q_obj=[0,3], tf_expect=True):
    q_objetivo, q_target = q_obj
    state_1 = self.SWAP(ket_0, [], q_obj =   [2,3], tf_expectt = tf_expect).states[-1]
    state_2 = self.SWAP(state_1, [], q_obj = [0,1], tf_expectt = tf_expect).states[-1]
    state_3 = self.CNOT(state_2, [], q_obj = [1,2], tf_expectt = tf_expect)
    state_4 = self.SWAP(state_3, [], q_obj = [0,1], tf_expectt = tf_expect).states[-1]
    state_5 = self.SWAP(state_4, [], q_obj = [2,3], tf_expectt = tf_expect).states[-1]
    return state_5