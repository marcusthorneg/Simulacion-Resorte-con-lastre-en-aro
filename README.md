# Dinámica Lagrangiana de un Resorte en un Aro Vertical
### Trabajo de Simulación Numérica — Mecánica y Ondas I
**Grado en Física — Universidad de Córdoba (UCO)**  
**Convocatoria:** Curso 2025–2026 (Ejercicio 1)  
**Autor:** Marcus Thorne y José Carlos Arévalo

---

## 1. Descripción del Sistema Físico

El sistema consiste en un aro circular rígido vertical de radio $R$ fijo en el plano cartesiano $Oxy$ con centro en el origen $O(0,0)$. Sobre el aro desliza sin rozamiento el punto de enganche $A$ de un resorte ideal de constante elástica $k$, longitud natural $l_0$ y masa despreciable. En el otro extremo del resorte pende una masa puntual $m$, sometida a la acción de la gravedad uniforme $\vec{g} = -g\,\hat{\jmath}$.

* **Grados de libertad:** $l = 2$.
* **Coordenadas generalizadas:** $q = (r, \theta)$, donde $r(t)$ es la longitud instantánea del resorte (elongación $\Delta l = r - l_0$) y $\theta(t)$ es el ángulo respecto a la vertical descendente.
* **Función Lagrangiana:**
  $$L(r, \theta, \dot{r}, \dot{\theta}) = \frac{1}{2}m\left[ \dot{r}^2 + (R + r)^2\dot{\theta}^2 \right] - \frac{1}{2}k(r - l_0)^2 + mg(R + r)\cos\theta$$
* **Fuerza de ligadura normal del aro:** Obtenida mediante multiplicadores de Lagrange con la ligadura holónoma $f(\rho_A) = \rho_A - R = 0$:
  $$\vec{N}_{\text{aro}} = \lambda(t)\,\hat{u}_r = -k(r - l_0)\,\hat{u}_r$$

---

## 2. Requisitos e Instalación

Para ejecutar la simulación se requiere **Python 3.8 o superior** y las bibliotecas científicas estándar listadas en `requirements.txt`.

### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/marcusthorneg/Simulacion-Resorte-con-lastre-en-aro.git
cd Simulacion-Resorte-con-lastre-en-aro
```

### Paso 2: Instalar las dependencias
```bash
pip install -r requirements.txt
```

---

## 3. Instrucciones de Ejecución

### Simulación Interactiva Principal
Para abrir la interfaz gráfica interactiva en tiempo real:
```bash
python simulacion_interactiva.py
```

#### Controles y funciones de la interfaz:
* **Deslizadores en tiempo real (Sliders):**
  * Parámetros físicos del sistema: masa $m$, constante elástica $k$, radio del aro $R$, longitud natural $l_0$, aceleración de la gravedad $g$.
  * Condiciones iniciales: ángulo inicial $\theta_0$, velocidad angular $\dot{\theta}_0$, elongación inicial $\Delta r_0$, velocidad radial $\dot{r}_0$.
* **Botones de Casos Límite Físicos:**
  * **Caso Base:** Parámetros estándar de oscilación libre.
  * **Péndulo Rígido ($k \to \infty$):** Bloquea la elongación ($r = l_0$) y reproduce la dinámica exacta de un péndulo simple ideal de longitud $L = R + l_0$.
  * **Sin Gravedad ($g = 0$):** Coordenada $\theta$ cíclica; muestra la conservación estricta del momento angular canónico $p_\theta = m(R+r)^2\dot{\theta}$.
  * **Péndulo Elástico ($R \to 0$):** El radio del aro se anula y el sistema converge a un péndulo elástico suspendido del origen fijo $O(0,0)$.
  * **Resonancia 1:2:** Ajuste para observar transferencia periódica de energía entre modos ($\omega_r \approx 2\omega_\theta$).
* **Telemetría HUD:** Monitoreo dinámico de coordenadas $(r, \theta)$, velocidades, energías ($T, V, E$), error relativo de conservación $\Delta E/E_0$, fuerza normal $\lambda$ y momento angular canónico $p_\theta$.
* **Vector de Reacción:** Flecha magenta que representa en todo momento la fuerza vincular normal $\vec{N}_{\text{aro}}$ que el aro ejerce sobre el punto de enganche.

---

## 4. Estudio Numérico y de Parámetros

El script `estudio_sistematico.py` complementa la simulación interactiva resolviendo numéricamente las ecuaciones diferenciales de Euler-Lagrange bajo distintos escenarios:
```bash
python estudio_sistematico.py
```
Este script realiza barridos sistemáticos para analizar:
1. La trayectoria cartesiana espacial $(x, y)$ y la evolución temporal acoplada de $(r(t), \theta(t))$.
2. La conservación estricta de la energía mecánica total ($E = T + V = \text{cte}$).
3. La influencia de la rigidez del muelle $k$ en la frecuencia y elongación radial.
4. La respuesta ante diferentes amplitudes iniciales $\theta_0$ (pequeñas oscilaciones armónicas vs. gran amplitud no lineal).
5. La convergencia numérica hacia los límites teóricos ($k \to \infty$ y $g = 0$) y la fuerza de ligadura normal $\lambda(t)$.

---

## 5. Estructura del Repositorio

```text
├── simulacion_interactiva.py     # Programa ejecutable principal (interfaz interactiva a 60 FPS)
├── estudio_sistematico.py        # Script para la resolución numérica y análisis de parámetros
├── requirements.txt              # Dependencias de Python necesarias (numpy, scipy, matplotlib)
├── README.md                     # Documentación e instrucciones de uso
└── .gitignore                    # Archivos y cachés ignorados por Git
```
