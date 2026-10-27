"""Genera docs/presentacion.pdf (10 diapositivas) con Matplotlib."""
from common import *
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

AZ, RJ, GR = "#1b3a57", "#d1495b", "#444444"
W, H = 13.33, 7.5


def base(titulo, n):
    fig = plt.figure(figsize=(W, H), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.add_artist(plt.Rectangle((0, 0.89), 1, 0.11, transform=fig.transFigure, color=AZ))
    fig.text(0.04, 0.925, titulo, color="white", fontsize=24, weight="bold", va="center")
    fig.text(0.96, 0.02, f"{n}/10", ha="right", fontsize=10, color=GR)
    fig.text(0.04, 0.02, "La ruina del jugador - Modelacion y Simulacion - UVG", fontsize=9, color=GR)
    return fig


def texto(fig, x, y, lineas, size=16, w=0.9):
    for i, l in enumerate(lineas):
        fig.text(x, y - i * size / 400 * 1.25, l, fontsize=size, va="top", color="#222222")


def imagen(fig, nombre, rect):
    ax = fig.add_axes(rect); ax.imshow(mpimg.imread(os.path.join(FIG, nombre))); ax.axis("off")

pdf = PdfPages(os.path.join(ROOT, "docs", "presentacion.pdf"))
def fin(fig): pdf.savefig(fig); plt.close(fig)

# 1 portada
fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor(AZ)
fig.text(0.5, 0.62, "La ruina del jugador", color="white", fontsize=48, weight="bold", ha="center")
fig.text(0.5, 0.50, "Cadenas de Markov, modelos basados en agentes y redes", color="#cfe3f5", fontsize=24, ha="center")
fig.text(0.5, 0.30, "Proyecto Final - Modelacion y Simulacion\nUniversidad del Valle de Guatemala\nJose Lopez, Luis Palacios, Hugo Barillas, Roberto Najera", color="white", fontsize=16, ha="center")
fin(fig)

# 2 problema
fig = base("Problema y preguntas", 2)
texto(fig, 0.05, 0.82, [
    "Ruleta europea: p = 18/37 (ventaja de la casa 2.70%)  |  americana: p = 18/38 (5.26%)",
    "",
    "1. Probabilidad de ruina antes de la meta, segun la estrategia",
    "2. Tiempo medio hasta terminar",
    "3. Simulacion vs solucion analitica de Markov",
    "4. Una estrategia puede vencer a la casa, o solo cambia como se pierde?",
    "5. Si se copian en una red social, que estrategias se propagan?",
    "",
    "Tecnicas: espacio de estados + ABM + redes (ER, WS, BA)"], size=17)
fin(fig)

# 3 Markov
fig = base("Parte 1: cadena de Markov", 3)
texto(fig, 0.04, 0.84, ["Estados 0..N; 0 y N absorbentes", "i->i+1 (p), i->i-1 (q)", "Forma canonica [Q R; 0 I]",
                        "F=(I-Q)^-1  ->  visitas y T=F1", "Absorcion: B = F R", "", "N=50, i=25:", "Europea: ruina 0.794, 545 rondas",
                        "Americana: ruina 0.933, 411 rondas", "Error vs formula < 1e-10"], size=15)
imagen(fig, "p1_ruina_duracion_visitas.png", [0.40, 0.25, 0.58, 0.58]); fin(fig)

# 4 sin meta / mesa
fig = base("Sin meta y limite de mesa", 4)
imagen(fig, "p1_sin_meta.png", [0.02, 0.40, 0.96, 0.46])
texto(fig, 0.05, 0.36, ["Sin meta con p<1/2 la ruina es segura; semivida: 1056 rondas (europea) vs 397 (americana)",
                        "Martingala con limite de mesa: cadena exacta en (capital, racha). Ruina 100->200: limite 8 = 0.77, limite 64 = 0.58"], size=14)
fin(fig)

# 5 validacion
fig = base("Parte 2: ABM y validacion", 5)
imagen(fig, "p2_validacion_ruina.png", [0.02, 0.38, 0.62, 0.50])
texto(fig, 0.66, 0.80, ["8 estrategias, miles de agentes", "(NumPy vectorizado)", "", "Apuesta fija, 20,000 agentes:", "error max ruina 0.011", "96% de puntos en IC95%", "error duracion <= 3.1%", "",
                        "Martingala con mesa vs", "cadena exacta: 0.575 vs 0.579"], size=14)
imagen(fig, "p2_trayectorias.png", [0.02, 0.06, 0.62, 0.32]); fin(fig)

# 6 estrategias
fig = base("Ninguna estrategia vence a la casa", 6)
imagen(fig, "p2_histogramas_capital_final.png", [0.01, 0.34, 0.62, 0.54])
texto(fig, 0.65, 0.84, ["Perdida por unidad apostada", "(100 -> 200, europea):", "", "Fija 0.027   Martingala 0.026", "Paroli 0.027  Porcentaje 0.027",
                        "Falacia 0.027  Persecucion 0.027", "teoria 1-2p = 0.027", "",
                        "Martingala: ruina 58% vs", "99.6% de la fija, pero apuesta", "6x menos. Solo cambia la", "varianza y la forma."], size=14)
imagen(fig, "p2_martingala_asimetria.png", [0.01, 0.06, 0.5, 0.27]); fin(fig)

# 7 Kelly y casa
fig = base("Kelly (p=0.52) y la casa", 7)
imagen(fig, "p2_kelly.png", [0.01, 0.40, 0.52, 0.48]); imagen(fig, "p2_casa.png", [0.01, 0.06, 0.62, 0.33])
texto(fig, 0.56, 0.84, ["Kelly f*=4%: mediana 496 tras 2000", "rondas; 2x Kelly: crecimiento 0;", "3x Kelly: 70% de ruina.", "Medio Kelly: ~75% del crecimiento",
                        "con ruina ~0.", "En ruleta Kelly dice: no apostar.", "", "Casa: ganancia/unidad -> 0.027,", "desviacion ~ 1/sqrt(n)"], size=14); fin(fig)

# 8 red
fig = base("Parte 3: la Martingala se propaga", 8)
imagen(fig, "p3_proporciones_topologias.png", [0.01, 0.05, 0.70, 0.83])
texto(fig, 0.72, 0.84, ["500 agentes, imitan al vecino", "que mas gano en la epoca;", "arruinados invisibles.", "",
                        "Martingala final:", "ER 79%  WS 74%  BA 64%", "", "Ruina con imitacion:", "67-76% vs 35% sin imitar", "", "Hub que presume: 81%", "vs 76% honesto"], size=13); fin(fig)

# 9 intervenciones y metricas
fig = base("Metricas de red e intervenciones", 9)
imagen(fig, "p3_metricas_red.png", [0.01, 0.46, 0.60, 0.42]); imagen(fig, "p3_intervenciones.png", [0.01, 0.06, 0.60, 0.38])
texto(fig, 0.63, 0.84, ["Grado/centralidad casi", "no predicen la riqueza", "(|rho| < 0.05)", "", "Informantes en hubs: gran", "efecto en libre de escala",
                        "(ruina 67% -> 46%, 20 inf.)", "", "Limite de mesa baja la ruina", "pero no la perdida: el", "apostado total manda"], size=13); fin(fig)

# 10 conclusiones
fig = base("Conclusiones y limitaciones", 10)
texto(fig, 0.05, 0.82, [
    "1. Con ventaja de la casa, toda estrategia pierde 1-2p por unidad apostada",
    "2. Markov y simulacion coinciden (tambien con limite de mesa)",
    "3. La meta se alcanza mas seguido apostando mas, a costa de colas catastroficas",
    "4. Las estrategias malas se difunden: exitos visibles, fracasos invisibles",
    "5. Informantes en hubs y limites bajos de mesa son las intervenciones utiles",
    "",
    "Limitaciones: ruleta sin memoria, rondas independientes, sin costos",
    "de transaccion; reglas de imitacion son supuestos, no calibrados con datos.",
    "",
    "Codigo, informe y datos: https://github.com/Ultimate-Truth-Seeker/ProyectoFinalDL.git"], size=16)
fin(fig)
pdf.close()
