# -*- coding: utf-8 -*-
"""
ESTUDIO SISTEMÁTICO DEL MOVIMIENTO - RESORTE EN ARO VERTICAL
Mecánica y Ondas I — Grado en Física (Universidad de Córdoba)
Tema 3: Formulación Lagrangiana de la Mecánica

Genera 2 láminas científicas esenciales y limpias para el Dossier:
- Lámina 1: Dinámica del Sistema y Conservación de la Energía
- Lámina 2: Verificación Física: Casos Límite y Fuerza de Ligadura
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolver_sistema(m=1.0, k=50.0, R=1.0, l0=0.5, g=9.81,
                     theta0=np.pi/4, vtheta0=0.0, r0_extra=0.4, vr0=0.0,
                     t_max=10.0, n_points=1000):
    """Integra las ecuaciones de Euler-Lagrange con DOP853."""
    r0 = R + l0 + r0_extra
    state0 = [r0, vr0, theta0, vtheta0]

    def odes(t, s):
        r, vr, th, vth = s
        r_safe = max(r, 0.02)
        drdt = vr
        dvrdt = r_safe * (vth**2) + g * np.cos(th) - (k / m) * (r_safe - R - l0)
        dthdt = vth
        dvthdt = -2.0 * (vr * vth) / r_safe - (g / r_safe) * np.sin(th)
        return [drdt, dvrdt, dthdt, dvthdt]

    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(odes, (0, t_max), state0, t_eval=t_eval, method='DOP853',
                    rtol=1e-9, atol=1e-9)

    r, vr, th, vth = sol.y
    T = 0.5 * m * (vr**2 + (r * vth)**2)
    V = -m * g * r * np.cos(th) + 0.5 * k * (r - R - l0)**2
    E = T + V
    lam = -k * (r - R - l0)  # Multiplicador de Lagrange: Reacción normal del aro
    ptheta = m * (r**2) * vth  # Momento canónico conjugado

    return {
        't': sol.t, 'r': r, 'vr': vr, 'theta': th, 'vtheta': vth,
        'T': T, 'V': V, 'E': E, 'lam': lam, 'ptheta': ptheta,
        'x': r * np.sin(th), 'y': -r * np.cos(th),
        'm': m, 'k': k, 'R': R, 'l0': l0, 'g': g
    }

def generar_lamina_1():
    """Lámina 1: Dinámica Fundamental y Conservación de la Energía."""
    print("[1/2] Generando Lámina 1: Dinámica y Conservación de la Energía...")
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))

    sol = resolver_sistema(m=1.0, k=50.0, R=1.0, l0=0.5, g=9.81,
                           theta0=np.pi/3, r0_extra=0.4, t_max=10.0)

    # 1. Trayectoria real en el espacio (x, y)
    theta_aro = np.linspace(0, 2*np.pi, 200)
    axs[0, 0].plot(sol['R']*np.sin(theta_aro), -sol['R']*np.cos(theta_aro),
                   'k--', lw=1.5, label=rf'Aro rígido ($R = {sol["R"]:.1f}$ m)')
    axs[0, 0].plot(sol['x'], sol['y'], color='#1f77b4', lw=1.5, alpha=0.85, label='Trayectoria masa $m$')
    axs[0, 0].plot(0, 0, 'r+', ms=10, mew=2, label='Centro $O(0,0)$')
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_title('1. Trayectoria Real en el Plano $(x, y)$')
    axs[0, 0].set_xlabel('$x$ (m)')
    axs[0, 0].set_ylabel('$y$ (m)')
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid(True)

    # 2. Evolución de coordenadas generalizadas (r, theta)
    ax_th = axs[0, 1]
    ax_r = ax_th.twinx()
    l1 = ax_th.plot(sol['t'], np.degrees(sol['theta']), color='#2ca02c', lw=1.8, label=r'$\theta(t)$ (grados)')
    l2 = ax_r.plot(sol['t'], sol['r'], color='#d62728', ls='--', lw=1.8, label=r'$r(t)$ (m)')
    ax_th.set_title(r'2. Evolución Temporal de Coordenadas $(r, \theta)$')
    ax_th.set_xlabel('Tiempo $t$ (s)')
    ax_th.set_ylabel(r'$\theta$ (grados)', color='#2ca02c')
    ax_r.set_ylabel(r'$r$ (m)', color='#d62728')
    ax_th.grid(True)
    lines = l1 + l2
    ax_th.legend(lines, [l.get_label() for l in lines], loc='upper right')

    # 3. Balance de energía mecánica
    axs[1, 0].plot(sol['t'], sol['T'], color='#ff7f0e', lw=1.8, label='Cinética $T(t)$')
    axs[1, 0].plot(sol['t'], sol['V'], color='#17becf', lw=1.8, label='Potencial $V(t)$')
    axs[1, 0].plot(sol['t'], sol['E'], color='black',   lw=2.4, label='Total $E = T + V$ (cte)')
    axs[1, 0].set_title(r'3. Conservación de la Energía Mecánica')
    axs[1, 0].set_xlabel('Tiempo $t$ (s)')
    axs[1, 0].set_ylabel('Energía (J)')
    axs[1, 0].legend(loc='upper right')
    axs[1, 0].grid(True)

    # 4. Influencia de la rigidez k del resorte
    k_vals = [20.0, 50.0, 150.0]
    k_labels = ['$k = 20$ N/m (blando)', '$k = 50$ N/m (medio)', '$k = 150$ N/m (rígido)']
    colores_k = ['#9467bd', '#1f77b4', '#d62728']
    for k_i, lbl, col in zip(k_vals, k_labels, colores_k):
        sol_k = resolver_sistema(k=k_i, theta0=np.pi/4, r0_extra=0.3, t_max=6.0)
        elong = sol_k['r'] - (sol_k['R'] + sol_k['l0'])
        axs[1, 1].plot(sol_k['t'], elong, label=lbl, color=col)
    axs[1, 1].set_title(r'4. Influencia de la Rigidez $k$ en la Elongación Radial')
    axs[1, 1].set_xlabel('Tiempo $t$ (s)')
    axs[1, 1].set_ylabel(r'Elongación $\Delta l = r - R - l_0$ (m)')
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].grid(True)

    fig.suptitle('LÁMINA 1: Dinámica del Sistema y Conservación de la Energía', fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'dossier_lamina_1.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Guardado: {path}")

def generar_lamina_2():
    """Lámina 2: Comprobación Física: Casos Límite y Fuerza de Ligadura."""
    print("[2/2] Generando Lámina 2: Casos Límite y Fuerza de Ligadura...")
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))

    # 1. Variación de amplitud inicial theta_0
    amplitudes = [np.radians(15), np.radians(45), np.radians(90)]
    labels_amp = [r'$\theta_0 = 15^\circ$ (pequeña oscilación)',
                  r'$\theta_0 = 45^\circ$ (amplitud media)',
                  r'$\theta_0 = 90^\circ$ (gran amplitud)']
    colores_amp = ['#2ca02c', '#ff7f0e', '#d62728']
    for th0, lbl, col in zip(amplitudes, labels_amp, colores_amp):
        sol = resolver_sistema(theta0=th0, r0_extra=0.2, t_max=8.0)
        axs[0, 0].plot(sol['t'], np.degrees(sol['theta']), label=lbl, color=col)
    axs[0, 0].set_title(r'1. Influencia de la Amplitud Inicial $\theta_0$')
    axs[0, 0].set_xlabel('Tiempo $t$ (s)')
    axs[0, 0].set_ylabel(r'$\theta$ (grados)')
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid(True)

    # 2. Caso límite k -> inf vs. Péndulo simple
    L_pend = 1.0 + 0.5  # R + l0 = 1.5 m
    sol_rigido = resolver_sistema(k=10000.0, r0_extra=0.0, theta0=np.pi/3, t_max=6.0)

    def ode_pendulo(t, y):
        th, vth = y
        return [vth, -(9.81 / L_pend) * np.sin(th)]

    t_eval = sol_rigido['t']
    sol_p = solve_ivp(ode_pendulo, (0, 6.0), [np.pi/3, 0.0], t_eval=t_eval, method='DOP853')

    axs[0, 1].plot(sol_rigido['t'], np.degrees(sol_rigido['theta']),
                   'b-', lw=2.4, label=r'Simulación ($k = 10^4$ N/m)')
    axs[0, 1].plot(sol_p.t, np.degrees(sol_p.y[0]),
                   'r--', lw=2.0, label=rf'Péndulo simple ($L = R + l_0 = {L_pend:.1f}$ m)')
    axs[0, 1].set_title(r'2. Caso Límite $k \to \infty$: Péndulo Simple Rígido')
    axs[0, 1].set_xlabel('Tiempo $t$ (s)')
    axs[0, 1].set_ylabel(r'$\theta$ (grados)')
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].grid(True)

    # 3. Caso límite g = 0: Coordenada cíclica y momento angular
    sol_g0 = resolver_sistema(g=0.0, theta0=0.0, vtheta0=2.0, r0_extra=0.3, t_max=8.0)
    axs[1, 0].plot(sol_g0['t'], sol_g0['ptheta'], color='#8c564b', lw=2.2,
                   label=r'$p_\theta(t) = m r^2 \dot{\theta}$')
    axs[1, 0].set_title(r'3. Caso Límite $g = 0$: Conservación de $p_\theta$')
    axs[1, 0].set_xlabel('Tiempo $t$ (s)')
    axs[1, 0].set_ylabel(r'Momento Canónico $p_\theta$ (kg$\cdot$m$^2$/s)')
    axs[1, 0].set_ylim(sol_g0['ptheta'][0] * 0.95, sol_g0['ptheta'][0] * 1.05)
    axs[1, 0].legend(loc='center right')
    axs[1, 0].grid(True)

    # 4. Fuerza de ligadura normal del aro lambda(t) (Apartado d)
    sol_norm = resolver_sistema(theta0=np.pi/4, r0_extra=0.4, t_max=8.0)
    axs[1, 1].plot(sol_norm['t'], sol_norm['lam'], color='#9467bd', lw=1.8,
                   label=r'$\lambda(t) = -k(r - R - l_0)$')
    axs[1, 1].axhline(0, color='gray', ls=':', lw=1.0)
    axs[1, 1].set_title(r'4. Fuerza de Ligadura Normal del Aro $\lambda(t)$')
    axs[1, 1].set_xlabel('Tiempo $t$ (s)')
    axs[1, 1].set_ylabel(r'Fuerza Normal $\lambda$ (N)')
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].grid(True)

    fig.suptitle('LÁMINA 2: Casos Límite y Comprobación Física de Ligaduras', fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'dossier_lamina_2.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Guardado: {path}")

def ejecutar_todo():
    print("=== GENERANDO LÁMINAS CIENTÍFICAS DEL DOSSIER SIMPLIFICADO ===")
    generar_lamina_1()
    generar_lamina_2()
    print("¡Láminas generadas con éxito!")

if __name__ == '__main__':
    ejecutar_todo()
