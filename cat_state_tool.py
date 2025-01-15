import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from tqdm import tqdm
import qutip as qt
from qutip import Qobj
from moviepy import ImageSequenceClip, concatenate_videoclips
import os
import re
import gc


class TimeDependentCatStateEvolution:
    def __init__(
        self, hilbert_dimension, time_total, timesteps, kappa, omega_r, gamma_t
    ):
        self.hilbert_dimension = hilbert_dimension
        self.time_total = time_total
        self.timesteps = timesteps
        self.dt = time_total / timesteps
        self.kappa = kappa
        self.omega_r = omega_r
        self.gamma_t = gamma_t
        self.a = self.annihilation(self.hilbert_dimension)
        self.adag = self.a.conj().T
        self.state_plus = self.basis(2, 0)
        self.state_down = self.basis(2, 1)

    def annihilation(self, hilbert_dimension):
        a = np.zeros((hilbert_dimension, hilbert_dimension), dtype=complex)
        for i in range(1, hilbert_dimension):
            a[i - 1, i] = np.sqrt(i)
            pass
        return a

    def basis(self, space_size, selected_dimension):
        if selected_dimension > space_size:
            raise ValueError("The Selected Dimension is Greater than the Space Size")
        basis = np.zeros((space_size, 1), dtype=complex)
        basis[selected_dimension] = 1.0
        return basis

    def alpha(self, t):
        return np.complex128(0.0 + 0j)

    def coherent(self, t, orientation):
        state = np.zeros((self.hilbert_dimension, 1), dtype=complex)
        match orientation:
            case 0:
                for n in range(self.hilbert_dimension):
                    state[n] = (self.alpha(t) ** n / np.sqrt(factorial(n))) * np.exp(
                        -(abs(self.alpha(t)) ** 2) / 2
                    )
                return state / np.linalg.norm(state)

            case 1:
                for n in range(self.hilbert_dimension):
                    state[n] = ((-self.alpha(t)) ** n / np.sqrt(factorial(n))) * np.exp(
                        -(abs(-self.alpha(t)) ** 2) / 2
                    )
                return state / np.linalg.norm(state)
            case _:
                raise ValueError("The Orientation Must be 0 or 1.")

    def time_dependent_cat_state(self, t):
        coherent_plus = np.kron(self.basis(2, 0), self.coherent(t, 0))
        coherent_minus = np.kron(self.basis(2, 1), self.coherent(t, 1))
        cat_state = coherent_plus + coherent_minus
        return cat_state / np.linalg.norm(cat_state)

    def prepare_initial_state(self):
        spin_down = self.basis(2, 1)
        vacuum = self.coherent(0, 0)
        initial_state = np.kron(spin_down, vacuum)
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        combined_gate = Z @ H
        combined_total = np.kron(combined_gate, np.eye(self.hilbert_dimension))
        final_state = combined_total @ initial_state
        return final_state / np.linalg.norm(final_state)

    def density_matrix_initial(self):
        cat_state = self.prepare_initial_state()
        density_matrix = np.outer(cat_state, cat_state.conj())
        trace = np.abs(np.trace(density_matrix))
        if not np.isclose(trace, 1.0, rtol=1e-10):
            print(f"Warning: Initial density matrix trace = {trace}")
            pass
        return density_matrix

    def lindblad_dissipator(self, t, density_matrix):
        identity = np.eye(2, dtype=complex)
        a = np.kron(identity, self.a)
        adag = a.conj().T
        phase = np.exp(-1j * t * self.omega_r)
        a_tilde = a * phase
        adag_tilde = adag * phase.conj()
        term1 = a_tilde @ density_matrix @ adag_tilde
        term2 = 0.5 * adag_tilde @ a_tilde @ density_matrix
        term3 = 0.5 * density_matrix @ adag_tilde @ a_tilde
        return term1 - term2 - term3

    def hamiltonian_density_matrix_commutator(self, t, density_matrix):
        phase = np.exp(-1j * t * self.omega_r)
        a_tilde = self.a * phase
        adag_tilde = self.adag * phase.conj()
        hamiltonian_tilde = self.gamma_t(t) * np.kron(
            np.array([[1, 0], [0, -1]]), (a_tilde + adag_tilde)
        )
        return hamiltonian_tilde @ density_matrix - density_matrix @ hamiltonian_tilde

    def simulation_evolution(self):
        times = np.linspace(0, self.time_total, self.timesteps)
        density_matrix = self.density_matrix_initial()
        dt = self.time_total / self.timesteps
        density_matrix_history = [density_matrix.copy()]

        def check_density_matrix(rho, step):
            if np.any(np.isnan(rho)):
                raise ValueError(f"NaN detected at step {step}")

            trace = np.abs(np.trace(rho))
            if not 0.99 < trace < 1.01:
                raise ValueError(f"Trace not preserved at step {step}: {trace}")
            if not np.allclose(rho, rho.conj().T):
                raise ValueError(f"Non-hermitian matrix at step {step}")

        for i in tqdm(range(self.timesteps)):
            t = times[i]
            try:
                k1 = dt * (
                    -1j * self.hamiltonian_density_matrix_commutator(t, density_matrix)
                    + self.kappa * self.lindblad_dissipator(t, density_matrix)
                )
                k2 = dt * (
                    -1j
                    * self.hamiltonian_density_matrix_commutator(
                        t + 0.5 * dt, density_matrix + 0.5 * k1
                    )
                    + self.kappa
                    * self.lindblad_dissipator(t + 0.5 * dt, density_matrix + 0.5 * k1)
                )
                k3 = dt * (
                    -1j
                    * self.hamiltonian_density_matrix_commutator(
                        t + 0.5 * dt, density_matrix + 0.5 * k2
                    )
                    + self.kappa
                    * self.lindblad_dissipator(t + 0.5 * dt, density_matrix + 0.5 * k2)
                )
                k4 = dt * (
                    -1j
                    * self.hamiltonian_density_matrix_commutator(
                        t + dt, density_matrix + k3
                    )
                    + self.kappa * self.lindblad_dissipator(t + dt, density_matrix + k3)
                )
                density_matrix += (k1 + 2 * k2 + 2 * k3 + k4) / 6
                check_density_matrix(density_matrix, i)
                density_matrix_history.append(density_matrix.copy())
            except ValueError as e:
                print(f"Simulation failed at step {i}, time {t}")
                print(f"Error: {str(e)}")
                print(f"Last valid trace: {np.trace(density_matrix_history[-1])}")
                break
        return density_matrix_history, times

    def wigner_block(self, density_matrix, xvec, pvec):
        return qt.wigner(density_matrix, xvec, pvec)

    def wigner_spin_blocks_sum(self, rho_total, xvec, pvec):
        size = rho_total.shape[0]
        if size % 2 != 0:
            raise ValueError("rho_total's dimension must be 2N x 2N.")
        N = size // 2

        rho_uu = Qobj(rho_total[0:N, 0:N])
        rho_ud = Qobj(rho_total[0:N, N : 2 * N])
        rho_du = Qobj(rho_total[N : 2 * N, 0:N])
        rho_dd = Qobj(rho_total[N : 2 * N, N : 2 * N])
        W_uu = self.wigner_block(rho_uu, xvec, pvec)
        W_ud = self.wigner_block(rho_ud, xvec, pvec)
        W_du = self.wigner_block(rho_du, xvec, pvec)
        W_dd = self.wigner_block(rho_dd, xvec, pvec)
        W_sum = W_uu + W_dd + np.real(W_ud) + np.real(W_du)  # type: ignore
        return W_sum

    def plot_wigner_2d(
        self, density_matrix_history, xvec, pvec, frame_index, filename="wigner_2d"
    ):
        if frame_index >= len(density_matrix_history):
            raise ValueError("Frame index exceeds the number of density matrices.")
        time = frame_index * self.dt
        fig, ax = plt.subplots(figsize=(8, 5))
        wigner = self.wigner_spin_blocks_sum(
            density_matrix_history[frame_index], xvec, pvec
        )
        x, y = np.meshgrid(xvec, pvec)
        wlim = max(abs(wigner.min()), abs(wigner.max()))
        c = ax.contourf(
            x, y, wigner, levels=np.linspace(-wlim, wlim, 100), cmap="seismic"
        )
        plt.colorbar(c)
        ax.set_xlabel(r"$\mathrm{Re}\left( \alpha \right)$")
        ax.set_ylabel(r"$\mathrm{Im}\left( \alpha \right)$")
        ax.set_title(f"Wigner function of time {time: 2f}")
        filename = filename + "_time_{:.2f}.png".format(time)
        fig.savefig(filename, dpi=1000, bbox_inches="tight")
        return fig

    def plot_wigner_2d_no_return(
        self, density_matrix_history, xvec, pvec, frame_index, filename="wigner_2d"
    ):
        if frame_index >= len(density_matrix_history):
            raise ValueError("Frame index exceeds the number of density matrices.")
        time = frame_index * self.dt
        fig, ax = plt.subplots(figsize=(8, 5))
        wigner = self.wigner_spin_blocks_sum(
            density_matrix_history[frame_index], xvec, pvec
        )
        x, y = np.meshgrid(xvec, pvec)
        wlim = max(abs(wigner.min()), abs(wigner.max()))
        c = ax.contourf(
            x, y, wigner, levels=np.linspace(-wlim, wlim, 100), cmap="seismic"
        )
        plt.colorbar(c)
        ax.set_xlabel(r"$\mathrm{Re}\left( \alpha \right)$")
        ax.set_ylabel(r"$\mathrm{Im}\left( \alpha \right)$")
        ax.set_title(f"Wigner function of time {time: 2f}")
        filename = filename + "_time_{:.2f}.png".format(time)
        fig.savefig(filename, dpi=1000, bbox_inches="tight")
        plt.close(fig)

    def plot_wigner_2d_multi(
        self, density_matrix_history, xvec, pvec, frame_number, filename="wigner_2d"
    ):
        filename_backup = filename
        if frame_number > len(density_matrix_history):
            raise ValueError("Frame number exceeds the number of density matrices.")
        indices = np.linspace(
            0, len(density_matrix_history) - 1, frame_number, dtype=int
        )

        selected_density_matrices = [density_matrix_history[int(i)] for i in indices]
        times = [i * self.dt for i in indices]
        for i in tqdm(range(0, len(selected_density_matrices))):
            plt.clf()
            plt.close("all")
            gc.collect()

            fig, ax = plt.subplots(figsize=(10, 8))
            wigner = self.wigner_spin_blocks_sum(
                selected_density_matrices[i], xvec, pvec
            )
            x, y = np.meshgrid(xvec, pvec)
            # wlim = max(abs(wigner.min()), abs(wigner.max()))
            wlim = 0.68
            levels = np.linspace(-wlim, wlim, 100)
            c = ax.contourf(x, y, wigner, levels=levels, cmap="seismic")
            plt.colorbar(c)
            ax.set_xlabel(r"$\mathrm{Re}\left( \alpha \right)$")
            ax.set_ylabel(r"$\mathrm{Im}\left( \alpha \right)$")
            ax.set_title(f"Wigner function of time {times[i]: 2f}")
            filename = filename + "_time_{:.2f}.png".format(times[i])
            fig.savefig(filename, dpi=500)
            plt.close(fig)
            filename = filename_backup


def animate_wigner_2d(foldername, filename="wigner_2d", fps=5, batch_size=10):
    png_files = [f for f in os.listdir(foldername) if f.endswith(".png")]

    def extract_time(name):
        match = re.search(r"wigner_2d_time_(\d+(\.\d+)?).png", name)
        return float(match.group(1)) if match else float("inf")

    png_files.sort(key=extract_time)

    temp_clips = []
    for i in tqdm(range(0, len(png_files), batch_size)):
        batch = png_files[i : i + batch_size]
        clip = ImageSequenceClip([os.path.join(foldername, f) for f in batch], fps=fps)
        temp_clips.append(clip)

    final_clip = concatenate_videoclips(temp_clips)
    final_clip.write_videofile(filename + ".mp4", codec="libx264", fps=fps)
