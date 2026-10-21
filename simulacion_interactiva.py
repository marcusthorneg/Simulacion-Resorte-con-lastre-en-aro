# -*- coding: utf-8 -*-
"""
SIMULACIÓN INTERACTIVA DEL RESORTE EN UN ARO VERTICAL
Mecánica y Ondas I — Grado en Física (Universidad de Córdoba)
Tema 3: Formulación Lagrangiana de la Mecánica

Características:
- Integración numérica de orden superior (DOP853) para las ecuaciones de Euler-Lagrange.
- Representación física 2D del aro, enganche sin masa, resorte elástico deformable y masa m.
- Monitoreo en tiempo real de la conservación de la energía mecánica total E = T + V.
- Vector interactivo de la fuerza de ligadura normal radial ejercida por el aro (lambda).
- Panel interactivo con explicaciones físicas de los casos límite de interés.
- Controles en tiempo real para todos los parámetros físicos y condiciones iniciales.
"""

import os
import sys
import numpy as np

try:
    from scipy.integrate import solve_ivp
except ModuleNotFoundError as exc:
    raise ImportError("Ejecute `pip install scipy` y vuelva a lanzar.") from exc

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass  # Si TkAgg no está disponible, Matplotlib seleccionará el backend interactivo predeterminado del sistema
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrowPatch

# ==============================================================================
# TEMA VISUAL OSCURO DE ALTO CONTRASTE
# ==============================================================================
plt.rcParams.update({
    "figure.facecolor": "#0a0b16",
    "axes.facecolor":   "#101226",
    "axes.edgecolor":   "#2d325a",
    "axes.labelcolor":  "#c5d1eb",
    "xtick.color":      "#7f8ba8",
    "ytick.color":      "#7f8ba8",
    "text.color":       "#c5d1eb",
    "grid.color":       "#1c2042",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
})

# ==============================================================================
# 1. PARÁMETROS Y RESOLUCIÓN NUMÉRICA
# ==============================================================================
P_DEFAULT = dict(
    m=1.0, k=50.0, R=1.0, l0=0.5, g=9.81,
    theta0_deg=45.0, vtheta0=0.0,
    dr0=0.5, vr0=0.0,
    t_max=15.0,
    n_frames=900,
)

N_ESPIRAS = 11

def resolver(p):
    """Integra las ecuaciones de Euler-Lagrange del Tema 3 con DOP853."""
    m, k, R, l0, g = p["m"], p["k"], p["R"], p["l0"], p["g"]
    th0 = np.radians(p["theta0_deg"])
    r0 = R + l0 + p["dr0"]
    vth0 = p["vtheta0"]
    vr0 = p["vr0"]

    def odes(t, s):
        r, vr, th, vth = s
        r_eff = max(r, 0.02)
        drdt = vr
        dvrdt = r_eff * (vth**2) + g * np.cos(th) - (k / m) * (r_eff - R - l0)
        dthdt = vth
        dvthdt = -2.0 * (vr * vth) / r_eff - (g / r_eff) * np.sin(th)
        return [drdt, dvrdt, dthdt, dvthdt]

    t_eval = np.linspace(0, p["t_max"], p["n_frames"])
    sol = solve_ivp(odes, (0, p["t_max"]),
                    [r0, vr0, th0, vth0],
                    t_eval=t_eval, method="DOP853",
                    rtol=1e-8, atol=1e-8)

    r, vr, th, vth = sol.y
    T = 0.5 * m * (vr**2 + (r * vth)**2)
    V = -m * g * r * np.cos(th) + 0.5 * k * (r - R - l0)**2
    E = T + V
    # Fuerza de ligadura que el aro ejerce sobre el enganche del resorte (Apartado d)
    lam = -k * (r - R - l0)
    ptheta = m * (r**2) * vth
    elong = r - R - l0
    err_E = np.abs((E - E[0]) / (E[0] if abs(E[0]) > 1e-6 else 1.0))

    return dict(
        t=sol.t, r=r, vr=vr, theta=th, vtheta=vth,
        T=T, V=V, E=E, lam=lam, ptheta=ptheta, elong=elong, err_E=err_E,
        xm=r * np.sin(th), ym=-r * np.cos(th),
        xh=R * np.sin(th), yh=-R * np.cos(th),
    )

def puntos_muelle(x0, y0, x1, y1, n=N_ESPIRAS, amp=0.07):
    """Genera coordenadas zigzag realistas del resorte."""
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    if L < 1e-6:
        return np.array([x0, x1]), np.array([y0, y1])
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx
    n_pts = n * 4 + 2
    s = np.linspace(0, 1, n_pts)
    zigzag = amp * np.sin(np.linspace(0, 2 * np.pi * n, n_pts))
    zigzag[0] = 0.0
    zigzag[-1] = 0.0
    return x0 + s * dx + zigzag * nx, y0 + s * dy + zigzag * ny

# ==============================================================================
# DICCIONARIO DE EXPLICACIONES FÍSICAS (TEMAS 3 Y 3' DE LA ASIGNATURA)
# ==============================================================================
INFO_CASOS = {
    "base": (
        "1. Configuración General del Examen (Apartados b y c)",
        "• Sistema con N = 2 grados de libertad descritos por (r, θ).\n"
        "• El enganche se mantiene colineal con la masa por equilibrio de fuerzas normales (m_e = 0).\n"
        "• Dinámica acoplada: la rotación genera fuerza centrífuga m·r·θ̇² hacia afuera,\n"
        "  y la vibración radial introduce aceleración de Coriolis -2·ṙ·θ̇/r en la velocidad angular.\n"
        "• La energía mecánica total E = T + V permanece estrictamente constante."
    ),
    "rigido": (
        "2. Límite de Rigidez k → ∞ (Péndulo Simple)",
        "• El resorte se vuelve prácticamente indeformable frente a las demás fuerzas.\n"
        "• La coordenada radial se bloquea en su longitud natural: r(t) ≈ R + l₀ = 1.5 m.\n"
        "• El sistema pierde efectivamente 1 grado de libertad y se comporta como un PÉNDULO SIMPLE.\n"
        "• La ecuación angular converge exactamente a:  θ̈ = -(g / L)·sin(θ)  con L = R + l₀.\n"
        "• La fuerza normal del aro λ equilibra en todo instante la tensión del resorte."
    ),
    "singravedad": (
        "3. Ausencia de Gravedad g = 0 (Coordenada Cíclica)",
        "• Al anular el campo gravitatorio (g = 0), la Lagrangiana no depende de θ (∂L/∂θ = 0).\n"
        "• Por el Teorema de Noether, el momento canónico conjugado p_θ = m·r²·θ̇ es\n"
        "  una CONSTANTE DEL MOVIMIENTO (conservación del momento angular).\n"
        "• El movimiento angular es monótono y la masa barre áreas iguales en tiempos iguales.\n"
        "• La fuerza radial es central y el movimiento describe oscilaciones elásticas puras con giro."
    ),
    "elastico_puro": (
        "4. Límite de Radio del Aro R → 0 (Péndulo Elástico)",
        "• El radio del aro colapsa (R → 0), de modo que el punto de enganche queda fijo en el origen O.\n"
        "• El sistema se reduce al PÉNDULO ELÁSTICO CLÁSICO suspendido de un punto fijo.\n"
        "• La ligadura holónoma impuesta por el aro desaparece en la práctica (r_e = 0).\n"
        "• La masa describe oscilaciones bidimensionales acopladas en el plano vertical."
    ),
    "resonancia": (
        "5. Resonancia Paramétrica no lineal (Relación 1:2)",
        "• Parámetros ajustados para que la frecuencia propia radial doble a la pendular (ω_r ≈ 2·ω_θ).\n"
        "• Se produce transferencia periódica no lineal de energía entre el modo pendular y el radial.\n"
        "• Fenómeno de batido: cuando la oscilación angular alcanza su máximo, la vibración radial\n"
        "  se atenúa, y a continuación la energía se transfiere de vuelta a la elongación del resorte."
    ),
    "personalizado": (
        "Configuración Dinámica Personalizada",
        "• Se han modificado los parámetros o condiciones iniciales con los deslizadores.\n"
        "• El integrador numérico calcula en tiempo real la solución exacta de las ecuaciones\n"
        "  de Euler-Lagrange acopladas, validando la conservación de la energía mecánica E = T + V."
    ),
}

# ==============================================================================
# 2. CONSTRUCCIÓN DE LA VENTANA Y SUBPLOTS
# ==============================================================================
fig = plt.figure(figsize=(16, 9.4))

# Layout de los paneles principales
ax_anim = fig.add_axes([0.04, 0.46, 0.35, 0.48])  # Animación 2D (izq)
ax_en   = fig.add_axes([0.43, 0.72, 0.34, 0.22])  # Energías (centro sup)
ax_info = fig.add_axes([0.43, 0.46, 0.34, 0.20])  # Explicación física (centro inf)
ax_data = fig.add_axes([0.80, 0.46, 0.18, 0.48])  # Telemetría HUD (dcha)
ax_data.set_facecolor("#0a0b16")
ax_data.axis("off")

ax_data.set_title("TELEMETRÍA FÍSICA", fontsize=9.5, fontweight="bold",
                  color="#00e5ff", pad=8)

# --- Elementos gráficos de la animación 2D ---
theta_circ = np.linspace(0, 2 * np.pi, 300)
linea_aro, = ax_anim.plot([], [], color="#00e5ff", lw=2.0, ls="--", alpha=0.75, label="Aro vertical")
ax_anim.plot([0], [0], "w+", ms=12, mew=2, label="Centro O")

linea_traza,  = ax_anim.plot([], [], color="#38bdf8", lw=1.2, alpha=0.45, label="Trayectoria masa")
linea_muelle, = ax_anim.plot([], [], color="#ff5376", lw=2.2, solid_capstyle="round")
punto_masa,   = ax_anim.plot([], [], "o", color="#fbbf24", ms=11, mec="white", mew=1.2, zorder=6, label="Masa $m$")
punto_enganche, = ax_anim.plot([], [], "o", color="#4ade80", ms=7.5, mec="white", mew=0.8, zorder=6, label="Enganche aro")

# Flecha vectorial de la fuerza de ligadura normal lambda
flecha_lam = FancyArrowPatch(
    posA=(0, 0), posB=(0.01, 0.01),
    arrowstyle="-|>", color="#e879f9",
    linewidth=2.4, mutation_scale=13, zorder=7,
)
ax_anim.add_patch(flecha_lam)

ax_anim.set_aspect("equal")
ax_anim.set_title("Animación del Sistema Físico", fontsize=11, pad=6)
ax_anim.set_xlabel("x (m)", fontsize=9.5)
ax_anim.set_ylabel("y (m)", fontsize=9.5)
ax_anim.grid(True)
ax_anim.legend(loc="upper right", fontsize=8, facecolor="#0a0b16", edgecolor="#2d325a")

# --- Elementos gráficos del panel de energías ---
curva_T, = ax_en.plot([], [], color="#fb923c", lw=1.6, label="Cinética (T)")
curva_V, = ax_en.plot([], [], color="#38bdf8", lw=1.6, label="Potencial (V)")
curva_E, = ax_en.plot([], [], color="white",   lw=2.2, label="Total (E = cte)")
linea_t_en = ax_en.axvline(0, color="#f43f5e", ls=":", lw=1.5, alpha=0.9)
ax_en.set_title("Conservación de la Energía Mecánica Total (E = T + V)", fontsize=10.5, pad=4)
ax_en.set_xlabel("t (s)", fontsize=9)
ax_en.set_ylabel("Energía (J)", fontsize=9)
ax_en.grid(True)
ax_en.legend(loc="upper right", fontsize=7.5, facecolor="#0a0b16", edgecolor="#2d325a")

# --- Elementos del panel de Explicación Física y Casos Límite ---
ax_info.set_facecolor("#0e1026")
ax_info.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax_info.spines.values():
    spine.set_color("#818cf8")
    spine.set_linewidth(1.2)
ax_info.set_title("EXPLICACIÓN FÍSICA / CASO LÍMITE ACTIVO", fontsize=9.5, fontweight="bold",
                  color="#fbbf24", pad=6)

txt_info_title = ax_info.text(
    0.03, 0.88, "1. Configuración General del Examen (Apartados b y c)",
    color="#38bdf8", fontsize=9.0, fontweight="bold", transform=ax_info.transAxes
)
txt_info_desc = ax_info.text(
    0.03, 0.72, "",
    color="#c5d1eb", fontsize=8.0, va="top", linespacing=1.35, transform=ax_info.transAxes
)

# --- Elementos gráficos del panel HUD ---
txt_hud = ax_data.text(
    0.04, 0.95, "",
    color="#00e5ff", fontsize=8.2, family="monospace",
    va="top", linespacing=1.35, transform=ax_data.transAxes
)

ax_data.text(
    0.04, 0.03,
    "Flecha magenta = Reacción normal λ\nejercida por el aro sobre el enganche",
    color="#e879f9", fontsize=7.2, transform=ax_data.transAxes
)

# ==============================================================================
# 3. VARIABLES DINÁMICAS Y CONTROL DE ANIMACIÓN
# ==============================================================================
_d = [resolver(P_DEFAULT)]
_frame_idx = [0]
_is_paused = [False]
_trail_mode = ["reciente"]
_updating_sliders = [False]

def fijar_explicacion(clave):
    titulo, desc = INFO_CASOS.get(clave, INFO_CASOS["personalizado"])
    txt_info_title.set_text(titulo)
    txt_info_desc.set_text(desc)
    fig.canvas.draw_idle()

fijar_explicacion("base")

def obtener_parametros_actuales():
    try:
        return dict(
            m=float(s_m.val), k=float(s_k.val), R=float(s_R.val),
            l0=float(s_l0.val), g=float(s_g.val),
            theta0_deg=float(s_th0.val), vtheta0=float(s_vth0.val),
            dr0=float(s_dr0.val), vr0=float(s_vr0.val),
            t_max=P_DEFAULT["t_max"], n_frames=P_DEFAULT["n_frames"]
        )
    except NameError:
        return P_DEFAULT.copy()

def actualizar_limites_graficos():
    d = _d[0]
    p = obtener_parametros_actuales()
    # Aro
    linea_aro.set_data(p["R"] * np.sin(theta_circ), -p["R"] * np.cos(theta_circ))
    
    # Límites de la animación 2D
    r_max = max(float(np.max(d["r"])), p["R"]) + 0.45
    ax_anim.set_xlim(-r_max, r_max)
    ax_anim.set_ylim(-r_max, r_max)

    # Límites de la gráfica de energía
    ax_en.set_xlim(0, p["t_max"])
    e_min = min(float(d["V"].min()), float(d["T"].min()), float(d["E"].min())) - 1.0
    e_max = max(float(d["T"].max()), float(d["E"].max())) + 1.0
    ax_en.set_ylim(e_min, e_max)

    # Curvas de energía estáticas
    curva_T.set_data(d["t"], d["T"])
    curva_V.set_data(d["t"], d["V"])
    curva_E.set_data(d["t"], d["E"])

# ==============================================================================
# 4. FUNCIÓN DE ACTUALIZACIÓN DE FRAME (BLIT OPTIMIZADO A 60 FPS)
# ==============================================================================
def update(frame):
    d = _d[0]
    n_total = len(d["t"])

    if not _is_paused[0]:
        _frame_idx[0] = (_frame_idx[0] + 1) % n_total

    idx = _frame_idx[0]

    # 1. Trayectoria de la masa
    if _trail_mode[0] == "reciente":
        start_idx = max(0, idx - 140)
        linea_traza.set_data(d["xm"][start_idx:idx+1], d["ym"][start_idx:idx+1])
    elif _trail_mode[0] == "completa":
        linea_traza.set_data(d["xm"][:idx+1], d["ym"][:idx+1])
    else:
        linea_traza.set_data([], [])

    # 2. Resorte dinámico reactivo
    xs, ys = puntos_muelle(d["xh"][idx], d["yh"][idx], d["xm"][idx], d["ym"][idx])
    linea_muelle.set_data(xs, ys)
    elong_i = d["elong"][idx]
    if elong_i > 0.05:
        linea_muelle.set_color("#f43f5e")  # Tensión
    elif elong_i < -0.05:
        linea_muelle.set_color("#38bdf8")  # Compresión
    else:
        linea_muelle.set_color("#4ade80")  # Longitud natural

    # 3. Puntos de masa y enganche
    punto_masa.set_data([d["xm"][idx]], [d["ym"][idx]])
    punto_enganche.set_data([d["xh"][idx]], [d["yh"][idx]])

    # 4. Flecha de reacción normal lambda del aro
    xh_i, yh_i = d["xh"][idx], d["yh"][idx]
    rh_i = np.hypot(xh_i, yh_i)
    if rh_i > 1e-6:
        ux, uy = -xh_i / rh_i, -yh_i / rh_i
    else:
        ux, uy = 0.0, -1.0
    escala_fuerza = 0.02
    lv = -d["lam"][idx] * escala_fuerza
    flecha_lam.set_positions((xh_i, yh_i), (xh_i + lv * ux, yh_i + lv * uy))

    # 5. Cursor temporal en energía
    linea_t_en.set_xdata([d["t"][idx], d["t"][idx]])

    # 6. Actualización del panel HUD de telemetría (actualizado periódicamente para 60 FPS)
    if idx % 3 == 0 or _is_paused[0]:
        th_deg_val = np.degrees(d["theta"][idx])
        hud_str = (
            f"Tiempo t:         {d['t'][idx]:>7.2f} s\n"
            f"Ángulo θ:         {th_deg_val:>+7.2f}°\n"
            f"Vel. angular ω:   {d['vtheta'][idx]:>+7.3f} rad/s\n"
            f"Distancia r:      {d['r'][idx]:>7.4f} m\n"
            f"Vel. radial ṙ:    {d['vr'][idx]:>+7.3f} m/s\n"
            f"Elongación Δl:    {d['elong'][idx]:>+7.4f} m\n"
            f"Energía Cinét. T: {d['T'][idx]:>7.3f} J\n"
            f"Energía Poten. V: {d['V'][idx]:>7.3f} J\n"
            f"Energía Total E:  {d['E'][idx]:>7.3f} J\n"
            f"Fuerza aro λ:     {d['lam'][idx]:>+7.2f} N\n"
            f"Momento p_θ:      {d['ptheta'][idx]:>+7.3f} J·s"
        )
        txt_hud.set_text(hud_str)
        return (linea_traza, linea_muelle, punto_masa, punto_enganche,
                flecha_lam, linea_t_en, txt_hud)

    return (linea_traza, linea_muelle, punto_masa, punto_enganche,
            flecha_lam, linea_t_en)

# ==============================================================================
# 5. SLIDERS DE PARÁMETROS Y CONDICIONES INICIALES
# ==============================================================================
def mk_slider(rect, label, lo, hi, init, step, fmt="%.2f"):
    ax_sl = fig.add_axes(rect, facecolor="#0e1022")
    sl = Slider(ax_sl, label, lo, hi, valinit=init, valstep=step,
                color="#00e5ff", valfmt=fmt)
    sl.label.set_fontsize(8)
    sl.label.set_color("#c5d1eb")
    sl.valtext.set_fontsize(8)
    sl.valtext.set_color("#00e5ff")
    return sl

# --- Columna 1: Parámetros del Sistema ---
s_g  = mk_slider([0.05, 0.35, 0.17, 0.022], "g (m/s²)", 0.0, 25.0, 9.81, 0.1)
s_m  = mk_slider([0.05, 0.31, 0.17, 0.022], "m (kg)",   0.1,  8.0,  1.0, 0.1)
s_k  = mk_slider([0.05, 0.27, 0.17, 0.022], "k (N/m)",  5.0, 500.0, 50.0, 5.0, fmt="%.0f")
s_R  = mk_slider([0.05, 0.23, 0.17, 0.022], "R (m)",    0.05, 3.0,  1.0, 0.05)
s_l0 = mk_slider([0.05, 0.19, 0.17, 0.022], "l0 (m)",   0.1,  2.5,  0.5, 0.05)

# --- Columna 2: Condiciones Iniciales ---
s_th0  = mk_slider([0.28, 0.35, 0.17, 0.022], "theta0 (deg)", -180.0, 180.0, 45.0, 1.0, fmt="%.0f")
s_vth0 = mk_slider([0.28, 0.31, 0.17, 0.022], "vtheta0 (rad/s)", -10.0, 10.0, 0.0, 0.1)
s_dr0  = mk_slider([0.28, 0.27, 0.17, 0.022], "Delta r0 (m)",  -0.4, 1.5, 0.5, 0.05)
s_vr0  = mk_slider([0.28, 0.23, 0.17, 0.022], "vr0 (m/s)",  -4.0, 4.0, 0.0, 0.1)

# Velocidad de reproducción
s_speed = mk_slider([0.28, 0.19, 0.17, 0.022], "Velocidad x", 0.1, 4.0, 1.0, 0.1)

# Inicialización de límites una vez creados los sliders
actualizar_limites_graficos()

def recalcular_simulacion(val=None):
    if _updating_sliders[0]:
        return
    p = obtener_parametros_actuales()
    _d[0] = resolver(p)
    _frame_idx[0] = 0
    actualizar_limites_graficos()
    fig.canvas.draw()
    if ani is not None and hasattr(ani, '_setup_blit'):
        ani._setup_blit()

for s in (s_g, s_m, s_k, s_R, s_l0, s_th0, s_vth0, s_dr0, s_vr0):
    s.on_changed(recalcular_simulacion)

def cambiar_velocidad(val=None):
    interval_ms = max(8, int(1000 / (60 * s_speed.val)))
    if 'ani' in globals():
        ani.event_source.interval = interval_ms

s_speed.on_changed(cambiar_velocidad)

# ==============================================================================
# 6. BOTONES DE CONTROL Y CASOS LÍMITE
# ==============================================================================
def mk_btn(rect, label, fc="#181c3b", tc="#00e5ff"):
    ax_b = fig.add_axes(rect)
    btn = Button(ax_b, label, color=fc, hovercolor="#282f63")
    btn.label.set_color(tc)
    btn.label.set_fontsize(8.2)
    btn.label.set_fontweight("bold")
    return btn

# Fila 1 de botones: Controles de reproducción
btn_play = mk_btn([0.50, 0.32, 0.11, 0.038], "|| Pausar")
btn_step = mk_btn([0.625, 0.32, 0.11, 0.038], ">> Paso")
btn_rst  = mk_btn([0.75, 0.32, 0.11, 0.038], "Reiniciar", tc="#4ade80")
btn_tr   = mk_btn([0.875, 0.32, 0.105, 0.038], "Estela: Rec.", tc="#fbbf24")

# Fila 2 de botones: Casos Límite de Interés
btn_c0 = mk_btn([0.50, 0.23, 0.09, 0.034], "1. Examen Base")
btn_c1 = mk_btn([0.60, 0.23, 0.09, 0.034], "2. Pendulo (k->inf)")
btn_c2 = mk_btn([0.70, 0.23, 0.09, 0.034], "3. Sin Gravedad")
btn_c3 = mk_btn([0.80, 0.23, 0.09, 0.034], "4. Pendulo Elast.")
btn_c4 = mk_btn([0.90, 0.23, 0.08, 0.034], "5. Resonancia")

btn_estudio = mk_btn([0.50, 0.16, 0.48, 0.042],
                     "GENERAR Y GUARDAR ESTUDIO SISTEMÁTICO COMPLETO",
                     fc="#2e1065", tc="#f43f5e")

def toggle_play(event):
    _is_paused[0] = not _is_paused[0]
    btn_play.label.set_text("> Reanudar" if _is_paused[0] else "|| Pausar")
    fig.canvas.draw_idle()

def paso_a_paso(event):
    if not _is_paused[0]:
        toggle_play(None)
    _frame_idx[0] = (_frame_idx[0] + 1) % len(_d[0]["t"])
    arts = update(_frame_idx[0])
    for a in arts:
        a.axes.draw_artist(a)
    for ax in (ax_anim, ax_en, ax_data):
        fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()

def resetear_tiempo(event):
    _frame_idx[0] = 0
    fig.canvas.draw()
    if ani is not None and hasattr(ani, '_setup_blit'):
        ani._setup_blit()

def alternar_estela(event):
    modos = ["reciente", "completa", "off"]
    curr = modos.index(_trail_mode[0])
    _trail_mode[0] = modos[(curr + 1) % len(modos)]
    lbl_map = {"reciente": "Estela: Rec.", "completa": "Estela: Full", "off": "Estela: Off"}
    btn_tr.label.set_text(lbl_map[_trail_mode[0]])
    fig.canvas.draw_idle()

# Callbacks para los casos límite
def fijar_parametros(m_val, k_val, R_val, l0_val, g_val, th0_val, vth0_val, dr0_val, vr0_val):
    _updating_sliders[0] = True
    s_m.set_val(m_val)
    s_k.set_val(k_val)
    s_R.set_val(R_val)
    s_l0.set_val(l0_val)
    s_g.set_val(g_val)
    s_th0.set_val(th0_val)
    s_vth0.set_val(vth0_val)
    s_dr0.set_val(dr0_val)
    _updating_sliders[0] = False
    s_vr0.set_val(vr0_val)  # El último recalcula todo una sola vez

def preset_base(event):
    fijar_explicacion("base")
    fijar_parametros(1.0, 50.0, 1.0, 0.5, 9.81, 45.0, 0.0, 0.5, 0.0)

def preset_rigido(event):
    fijar_explicacion("rigido")
    fijar_parametros(1.0, 500.0, 1.0, 0.5, 9.81, 60.0, 0.0, 0.0, 0.0)

def preset_singravedad(event):
    fijar_explicacion("singravedad")
    fijar_parametros(1.0, 40.0, 1.0, 0.5, 0.0, 0.0, 2.5, 0.4, 0.0)

def preset_elastico_puro(event):
    fijar_explicacion("elastico_puro")
    fijar_parametros(1.0, 40.0, 0.05, 1.0, 9.81, 45.0, 0.0, 0.3, 0.0)

def preset_resonancia(event):
    fijar_explicacion("resonancia")
    fijar_parametros(1.0, 22.0, 1.0, 0.5, 9.81, 18.0, 0.0, 0.0, 0.0)

def lanzar_estudio(event):
    print("\nGenerando estudio sistemático completo...")
    try:
        import estudio_sistematico
        estudio_sistematico.ejecutar_todo()
        print("¡Estudio sistemático guardado en la carpeta de la simulación!")
    except Exception as e:
        print(f"Error generando estudio sistemático: {e}")

btn_play.on_clicked(toggle_play)
btn_step.on_clicked(paso_a_paso)
btn_rst.on_clicked(resetear_tiempo)
btn_tr.on_clicked(alternar_estela)

btn_c0.on_clicked(preset_base)
btn_c1.on_clicked(preset_rigido)
btn_c2.on_clicked(preset_singravedad)
btn_c3.on_clicked(preset_elastico_puro)
btn_c4.on_clicked(preset_resonancia)
btn_estudio.on_clicked(lanzar_estudio)

# ==============================================================================
# 7. INICIALIZACIÓN DE LA ANIMACIÓN (BLIT=TRUE A 60 FPS)
# ==============================================================================
fig.suptitle("Simulación Interactiva: Dinámica Lagrangiana del Resorte en Aro Vertical",
             color="#c5d1eb", fontsize=11, fontweight="bold", y=0.985)

ani = None

def init():
    linea_traza.set_data([], [])
    linea_muelle.set_data([], [])
    punto_masa.set_data([], [])
    punto_enganche.set_data([], [])
    flecha_lam.set_positions((0, 0), (0.001, 0.001))
    linea_t_en.set_xdata([0, 0])
    txt_hud.set_text("")
    return (linea_traza, linea_muelle, punto_masa, punto_enganche,
            flecha_lam, linea_t_en, txt_hud)

def iniciar_animacion():
    global ani
    ani = animation.FuncAnimation(
        fig, update,
        init_func=init,
        frames=P_DEFAULT["n_frames"],
        interval=16,
        blit=True,
        repeat=True
    )

if __name__ == "__main__":
    iniciar_animacion()
    plt.show()
