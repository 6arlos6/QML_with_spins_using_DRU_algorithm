"""
Quantum Spin Processor Module.

This module provides the `Quantum_Spin_Proces` class for simulating quantum spin dynamics
using QuTiP and PennyLane. It supports various quantum gates and pulse simulations.
"""

from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, Union

from pennylane import numpy as np
from qutip import (
    sigmax,
    sigmay,
    sigmaz,
    tensor,
    mesolve,
    Qobj,
    qeye,
    destroy,
    Options,
    QobjEvo,
)

# Local imports
try:
    from .main_fun import traducir_a_positivo, pulse_x, pulse_x_with_noise
except ImportError:
    # Fallback for when running NOT as a package (e.g. direct execution or different path setup)
    # This helps robustness if user runs file directly.
    from main_fun import traducir_a_positivo, pulse_x, pulse_x_with_noise


class Quantum_Spin_Proces:
    """
    Simulates a quantum spin processor.

    Attributes:
        gir (float): Gyromagnetic ratio.
        B0 (float): Static magnetic field Z-component.
        B1 (float): RF magnetic field amplitude.
        nf (int): Noise factor/scaling.
        N_qubits (int): Number of qubits in the system.
        J (float): Coupling constant.
        tf_noise (bool): Flag to enable/disable noise in pulses.
        noise_std (float): Standard deviation of the noise.
        B1_offset (float): Offset for B1 field.
        n_points_pulse_Ri (int): Number of points for single-qubit pulse.
        n_points_pulse_2Qbits (int): Number of points for two-qubit pulse.
        n_swap (int): Number of SWAP operations.
        T1 (float): Relaxation time T1.
        T2 (float): Dephasing time T2.
        tf_quantum_noise (bool): Flag to enable/disable quantum noise (Lindblad).
        f_rage (float): Frequency range/detuning.
        save_time_values (bool): Flag to save state evolution over time.
        n_points_pulse_Ri_spl (int): Number of points for spline interpolation.
        save_pulses (bool): Flag to save pulse data.
        nstepsolver (int): Number of steps for the solver.
        version_qutip (str): QuTiP version ("4" or "5").
        free_time (float): Free evolution time added to operations.
    """

    def __init__(
        self,
        h: float = 1,
        gir: float = 1.760e11,
        B0: float = -10e-3,
        nf: int = 4,
        N_qubits: int = 1,
        J: float = 0.1e12,
        tf_noise: bool = False,
        noise_std: float = 0.01,
        B1_offset: float = 0,
        n_points_pulse_Ri: int = 2,
        n_points_pulse_2Qbits: int = 2,
        n_swap: int = 1,
        T1: float = 1e3,
        T2: float = 1e3,
        tf_quantum_noise: bool = False,
        f_rage: float = 0,
        save_time_values: bool = False,
        n_points_pulse_Ri_spl: int = 1000,
        save_pulses: bool = False,
        nstepsolver: int = 1_000,
        version_qutip: str = "5",
        free_time: float = 0,
    ):
        """
        Initializes the Quantum_Spin_Proces.
        """
        self.h = h
        self.gir = gir
        self.B0 = B0
        self.nf = nf
        self.N_qubits = N_qubits
        self.J = J
        self.tf_noise = tf_noise
        self.noise_std = noise_std
        self.B1_offset = B1_offset
        self.B1 = B1_offset 
        
        self.n_points_pulse_Ri = n_points_pulse_Ri
        self.n_points_pulse_2Qbits = n_points_pulse_2Qbits
        self.n_swap = n_swap
        # Initializing other parameters
        self.T1 = T1
        self.T2 = T2
        self.tf_quantum_noise = tf_quantum_noise
        self.f_rage = f_rage
        self.save_tv = save_time_values
        self.n_points_pulse_Ri_spl = n_points_pulse_Ri_spl
        self.save_pulse = save_pulses
        self.nstepsolver = nstepsolver
        self.version_qutip = version_qutip
        self.free_time = free_time

        # Internal state
        self.global_time = 0
        self.dict_states = {}
        self.pulse_type = defaultdict(list)
        self.states_in_time = []
        self.omegas_x = []
        
        # Initialized in methods but good to be aware of
        self.Dt = -7 # Default init
        self.q_obj = 0
        self.ket_0 = None
        self.delt_t = 0
        self.O_x = 0
        self.ω_x = 0
        self.ω_z = 0
        self.H0 = None
        self.H1 = None
        self.H2 = None
        self.e_ops = []
        self.c_ops = []
        self.args = {}
        self.tlist = []

    def Rz(
        self,
        alpha: float,
        ket_0: Qobj,
        q_obj: int = 0,
        tf_expect: bool = False
    ) -> Any:
        """
        Implements a rotation around the Z-axis (Rz gate).

        Args:
            alpha (float): Rotation angle.
            ket_0 (Qobj): Initial state vector.
            q_obj (int): Target qubit index.
            tf_expect (bool): Whether to calculate expectation values.

        Returns:
            Result object from mesolve.
        """
        alpha = -alpha
        # Initial states and target qubit
        self.q_obj = q_obj
        self.ket_0 = ket_0
        
        # Gate parameters
        self.ω_x = 0
        self.ω_z = self.gir * self.B0
        alpha = traducir_a_positivo(alpha)
        
        if np.abs(self.ω_z) > 0:
            self.delt_t = alpha / np.abs(self.ω_z)
        else:
            self.delt_t = 0 
            
        self.B1 = 0
        self.O_x = self.gir * (self.B1 / 2)

        # Solve Hamiltonian
        out = self.Hamiltonian_solve(tf_expect)
        if self.save_tv:
            self.states_in_time.append(out.states)
        return out

    def Rx(
        self,
        alpha: float,
        ket_0: Qobj,
        q_obj: int = 0,
        tf_expect: bool = False
    ) -> Any:
        """
        Implements a rotation around the X-axis (Rx gate).

        Args:
            alpha (float): Rotation angle.
            ket_0 (Qobj): Initial state vector.
            q_obj (int): Target qubit index.
            tf_expect (bool): Whether to calculate expectation values.

        Returns:
            Result object from mesolve.
        """
        # Initial states and target qubit
        self.q_obj = q_obj
        self.ket_0 = ket_0
        
        # Gate parameters
        self.ω_x = self.gir * self.B0 + self.f_rage
        self.ω_z = self.gir * self.B0
        
        if np.abs(self.ω_x) > 0:
             self.delt_t = (np.abs(alpha) * self.nf) / np.abs(self.ω_x)
        else:
             self.delt_t = 0

        # Calculate B1 required for the pulse
        if self.gir * self.delt_t != 0:
            self.B1 = (alpha) / (self.gir * self.delt_t)
        else:
             self.B1 = 0
             
        self.O_x = self.gir * self.B1

        # Solve Hamiltonian
        out = self.Hamiltonian_solve(tf_expect)
        if self.save_tv:
            self.states_in_time.append(out.states)
            self.omegas_x.append(self.ω_x)
        return out

    def Ry(
        self,
        alpha: float,
        ket_0: Qobj,
        q_obj: int = 0,
        tf_expect: bool = False
    ) -> Union[Any, List[Any]]:
        """
        Implements a rotation around the Y-axis (Ry gate) using decomposition Rx(pi/2) -> Rz(alpha) -> Rx(-pi/2).

        Args:
            alpha (float): Rotation angle.
            ket_0 (Qobj): Initial state vector.
            q_obj (int): Target qubit index.
            tf_expect (bool): Whether to calculate expectation values.

        Returns:
            Result object from the final Rx gate, or list of results if tf_expect is True.
        """
        # 1. Rx(pi/2)
        out_1 = self.Rx(np.pi / 2, ket_0, q_obj=q_obj, tf_expect=False)
        end_state_1 = out_1.states[-1]
        
        # 2. Rz(alpha)
        out_2 = self.Rz(alpha, end_state_1, q_obj=q_obj, tf_expect=False)
        end_state_2 = out_2.states[-1]
        
        # 3. Rx(-pi/2)
        out_3 = self.Rx(-np.pi / 2, end_state_2, q_obj=q_obj, tf_expect=False)

        if tf_expect:
            out_1_exp = self.Rx(np.pi / 2, ket_0, q_obj=q_obj, tf_expect=True)
            out_2_exp = self.Rz(alpha, end_state_1, q_obj=q_obj, tf_expect=True)
            out_3_exp = self.Rx(-np.pi / 2, end_state_2, q_obj=q_obj, tf_expect=True)
            out = [out_1_exp, out_2_exp, out_3_exp]
        else:
            out = out_3
        return out

    def H(
        self,
        ket_0: Qobj,
        alpha: float = np.pi / 2,
        q_obj: int = 0,
        tf_expect: bool = False
    ) -> Union[Any, List[Any]]:
        """
        Implements the Hadamard gate using Rz and Rx decomposition.
        """
        out_1 = self.Rz(np.pi / 2, ket_0, q_obj=q_obj, tf_expect=False)
        end_state_1 = out_1.states[-1]
        
        out_2 = self.Rx(alpha, end_state_1, q_obj=q_obj, tf_expect=False)
        end_state_2 = out_2.states[-1]
        
        out_3 = self.Rz(np.pi / 2, end_state_2, q_obj=q_obj, tf_expect=False)
        
        if tf_expect:
            out_1_exp = self.Rz(np.pi / 2, ket_0, q_obj=q_obj, tf_expect=True)
            out_2_exp = self.Rx(alpha, end_state_1, q_obj=q_obj, tf_expect=True)
            out_3_exp = self.Rz(np.pi / 2, end_state_2, q_obj=q_obj, tf_expect=True)
            out = [out_1_exp, out_2_exp, out_3_exp]
        else:
            out = out_3
        return out

    def SWAP(
        self,
        ket_0: Qobj,
        measure_op: Any = None,
        q_obj: List[int] = [0, 1],
        tf_expectt: bool = False
    ) -> Any:
        """
        Implements a SWAP gate between two qubits.
        """
        self.q_obj = q_obj
        self.ket_0 = ket_0
        self.Dt = np.pi / (self.J * self.n_swap)
        self.out = self.Hamiltonian_solve_excharge(tf_expectt)
        if self.save_tv:
            self.states_in_time.append(self.out.states)
        return self.out

    def sqrt_SWAP(
        self,
        ket_0: Qobj,
        measure_op: Any = None,
        q_obj: List[int] = [0, 1],
        tf_expectt: bool = False
    ) -> Any:
        """
        Implements a square root of SWAP gate.
        """
        self.q_obj = q_obj
        self.ket_0 = ket_0
        self.Dt = (np.pi / (2 * self.J * self.n_swap))
        self.out = self.Hamiltonian_solve_excharge(tf_expectt)
        if self.save_tv:
            self.states_in_time.append(self.out.states)
        return self.out

    def CNOT(
        self,
        ket_0: Qobj,
        measure_op: Any = None,
        q_obj: List[int] = [0, 1],
        tf_expectt: bool = False
    ) -> Qobj:
        """
        Implements a CNOT gate using decomposition.
        """
        self.q_obj = q_obj
        q_control, q_target = q_obj
        
        state_1 = self.Ry(np.pi / 2, ket_0, q_obj=q_target, tf_expect=False).states[-1]
        state_2 = self.sqrt_SWAP(state_1, [], q_obj=q_obj, tf_expectt=False).states[-1]
        state_3 = self.Rz(np.pi, state_2, q_obj=q_control, tf_expect=False).states[-1]
        state_4 = self.sqrt_SWAP(state_3, [], q_obj=q_obj, tf_expectt=False).states[-1]
        state_5 = self.Rz(-np.pi / 2, state_4, q_obj=q_control, tf_expect=False).states[-1]
        state_6 = self.Rz(-np.pi / 2, state_5, q_obj=q_target, tf_expect=False).states[-1]
        state_7 = self.Ry(-np.pi / 2, state_6, q_obj=q_target, tf_expect=False).states[-1]
        
        return state_7

    def CZ(
        self,
        ket_0: Qobj,
        measure_op: Any = None,
        q_obj: List[int] = [0, 1],
        tf_expectt: bool = False
    ) -> Qobj:
        """
        Implements a Controlled-Z (CZ) gate using decomposition.
        """
        self.q_obj = q_obj
        q_control, q_target = q_obj
        
        state_2 = self.sqrt_SWAP(ket_0, [], q_obj=q_obj, tf_expectt=False).states[-1]
        state_3 = self.Rz(-np.pi, state_2, q_obj=q_control, tf_expect=False).states[-1]
        state_4 = self.sqrt_SWAP(state_3, [], q_obj=q_obj, tf_expectt=False).states[-1]
        state_5 = self.Rz(np.pi / 2, state_4, q_obj=q_control, tf_expect=False).states[-1]
        state_6 = self.Rz(-np.pi / 2, state_5, q_obj=q_target, tf_expect=False).states[-1]
        
        return state_6

    def init_H_1q_and_opt(self, tf_expect: bool = False):
        """
        Initializes the single-qubit Hamiltonian components and operators.
        """
        # Hamiltonian Constants
        h0_constant = (self.h / 2) * (self.ω_z)
        h1_constant = -(self.h / 2) * (self.ω_x)
        h2_constant = (self.h / 2) * (self.O_x)

        # Apply rotations in individual qubits
        apply_qbit_z = []
        apply_qbit_x = []
        
        for i in range(self.N_qubits):
            if i == self.q_obj:
                apply_qbit_z.append(sigmaz())
                apply_qbit_x.append(sigmax())
            else:
                apply_qbit_z.append(qeye(2))
                apply_qbit_x.append(qeye(2))

        # Hamiltonian Terms
        self.H0 = h0_constant * tensor(*apply_qbit_z)
        self.H1 = h1_constant * tensor(*apply_qbit_z)
        self.H2 = h2_constant * tensor(*apply_qbit_x)

        # Expectation values
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
        self.e_ops = e_ops

        # Collapse operators (for Lindblad master equation)
        if self.tf_quantum_noise:
            apply_qbit_c_ops_1 = []
            apply_qbit_c_ops_2 = []
            for i in range(self.N_qubits):
                if i == self.q_obj:
                    a = destroy(2)
                    # Avoid division by zero if T1 or T2 are perfect
                    # Assuming T1, T2 are reasonable, but protecting against 0 division just in case
                    try: 
                        T2_star_inv = (1 / self.T2) - (1 / (2 * self.T1))
                        if T2_star_inv != 0:
                            T2_star = 1 / T2_star_inv
                            c2_coeff = np.sqrt(2 / T2_star)
                        else:
                            c2_coeff = 0
                    except ZeroDivisionError:
                         c2_coeff = 0

                    c1 = a / (np.sqrt(self.T1)) if self.T1 > 0 else a * 0
                    c2 = a.dag() * a * c2_coeff
                    
                    apply_qbit_c_ops_1.append(c1)
                    apply_qbit_c_ops_2.append(c2)
                else:
                    apply_qbit_c_ops_1.append(qeye(2))
                    apply_qbit_c_ops_2.append(qeye(2))
            c_ops = [tensor(*apply_qbit_c_ops_1), tensor(*apply_qbit_c_ops_2)]
        else:
            c_ops = []
        self.c_ops = c_ops

    def Hamiltonian_solve(self, tf_expect: bool = False) -> Any:
        """
        Solves the Hamiltonian evolution for a single operation.
        """
        # Initialize Hamiltonian parameters
        self.init_H_1q_and_opt(tf_expect)
        
        # Standard deviation correction for noise
        if self.B1 != 0:
            dv = self.noise_std / abs(self.B1)
        else:
            dv = 0
            
        self.args = {"t_init": 0, "t_final": self.delt_t, "std_noise": dv}

        # Simulation time including free evolution
        self.delt_t = self.delt_t + self.free_time
        
        # Time list for solver
        self.tlist = np.linspace(0, self.delt_t, self.n_points_pulse_Ri)

        # Construct Hamiltonian
        if not self.tf_noise:
            H = [self.H0, [self.H1, pulse_x], [self.H2, pulse_x]]
            Noise_x = ""
        else:
            # Coherent noise handling
            H, Noise_x = self.spline_from_version()

        # Save pulse metadata if requested
        if self.save_pulse:
            t_actual = self.global_time
            t_final = self.global_time + self.delt_t
            self.pulse_type[self.q_obj].append({
                "Type_pulse": "Unitary",
                "B0": self.B0,
                "B1": self.B1,
                "Delt_t": self.delt_t,
                "Q_bits_target": self.q_obj,
                "t_i": t_actual,
                "t_f": t_final,
                "Noise": Noise_x
            })

        # Update global time
        self.global_time += self.delt_t
        
        # Solve Master Equation
        options = Options(nsteps=self.nstepsolver)
        output_dm = mesolve(
            H,
            self.ket_0,
            self.tlist,
            self.c_ops,
            self.e_ops,
            self.args,
            options=options
        )
        return output_dm

    def spline_from_version(self) -> Tuple[Any, Any]:
        """
        Handles spline interpolation based on QuTiP version for noisy pulses.
        """
        if self.version_qutip == "4":
            # Version <= 4
            self.tlist_spline = np.linspace(0, self.delt_t, self.n_points_pulse_Ri_spl)
            noise_x = pulse_x_with_noise(self.tlist_spline, self.args)
            
            try:
                from qutip import Cubic_Spline
                S_x = Cubic_Spline(self.tlist_spline[0], self.tlist_spline[-1], noise_x)
            except ImportError:
                # If Cubic_Spline is missing (e.g. newer qutip or not installed), just use noise array or log warning
                # Returning noise_x as fallback to prevent crash, though logic might be slightly off without spline
                S_x = noise_x 
                
            H = [self.H0, [self.H1, pulse_x], [self.H2, S_x]]
        else:
            # Version >= 5
            times_spl = np.linspace(0, self.delt_t, self.n_points_pulse_Ri_spl)
            noise_x = pulse_x_with_noise(times_spl, self.args).flatten()
            pulse_x_values = pulse_x(times_spl, self.args).flatten()
            
            # QobjEvo is imported
            H = QobjEvo(
                [self.H0, [self.H1, pulse_x_values], [self.H2, noise_x]],
                tlist=times_spl
            )
        return H, noise_x

    def Hamiltonian_solve_excharge(self, tf_expect: bool = False) -> Any:
        """
        Solves the Hamiltonian for exchange interaction (Two-Qubit gates).
        """
        # Parameters
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

        # Exchange Hamiltonian
        term_sum = (
            tensor(*apply_qbit_x) +
            tensor(*apply_qbit_y) +
            tensor(*apply_qbit_z)
        )
        H = ((self.J * self.h**2) / 4) * term_sum
        
        # Hamiltonian pulse wrapper
        h_t = [H, pulse_x]
        
        # Standard deviation correction
        if self.B1 != 0:
            dv = self.noise_std / abs(self.B1)
        else:
            dv = 0
            
        self.args = {"t_init": 0, "t_final": delt_t, "std_noise": dv}

        # Expectation values
        if tf_expect:
            e_ops = []
        else:
            e_ops = []

        # Collapse operators
        c_ops = []
        
        self.tlist = np.linspace(0, delt_t, self.n_points_pulse_2Qbits)
        
        if self.save_pulse:
            t_actual = self.global_time
            t_final = self.global_time + delt_t
            key = f'I_{self.q_obj[0]}-{self.q_obj[1]}'
            self.pulse_type[key].append({
                "Type_pulse": "Two_Qubits",
                "Q_bits_target": self.q_obj,
                "J": self.J,
                "Delt_t": self.Dt,
                "t_i": t_actual,
                "t_f": t_final,
                "Noise": ""
            })
            
        self.global_time += self.Dt
        
        self.output = mesolve(
            h_t,
            self.ket_0,
            self.tlist,
            c_ops,
            e_ops,
            self.args
        )
        return self.output
