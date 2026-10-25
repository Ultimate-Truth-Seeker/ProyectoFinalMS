"""Parte 1: cadena de Markov (forma canonica, matriz fundamental, formulas)."""
from common import *
from src import markov as mk
import pandas as pd

out = {}
N = 50
ruedas = {"Europea (p=18/37)": mk.P_EUROPEA, "Americana (p=18/38)": mk.P_AMERICANA,
          "Justo (p=0.5)": 0.5, "Ventaja jugador (p=0.52)": 0.52}

# 1) Ejemplo pequeno de la matriz de transicion (N=6) y forma canonica
a = mk.analisis_absorcion(6, mk.P_EUROPEA)
Pc, Q, R, orden = mk.forma_canonica(a["P"])
np.set_printoptions(precision=4, suppress=True, linewidth=140)
with open(os.path.join(RES, "matriz_P_N6.txt"), "w") as f:
    f.write("Matriz P (N=6, p=18/37), estados 0..6\n%s\n\n" % a["P"])
    f.write("Orden canonico de estados: %s\n" % orden)
    f.write("Q (transitorios 1..5)\n%s\n\nR (columnas: ruina 0, meta 6)\n%s\n\n" % (Q, R))
    f.write("F = (I-Q)^-1\n%s\n\nt = F*1 (pasos esperados)\n%s\n\nB = F*R\n%s\n" % (a["F"], a["t"], a["B"]))

# 2) Comparacion F*R vs formula para N=50 en cada ruleta
filas = []
for nombre, p in ruedas.items():
    a = mk.analisis_absorcion(N, p)
    i = np.arange(1, N)
    ruina_f = mk.ruina_analitica(i, N, p)
    dur_f = mk.duracion_analitica(i, N, p)
    err_r = np.abs(a["B"][:, 0] - ruina_f).max()
    err_t = np.abs(a["t"] - dur_f).max()
    filas.append({"juego": nombre, "N": N, "i": 25, "P_ruina_FR": a["B"][24, 0],
                  "P_ruina_formula": ruina_f[24], "P_meta_FR": a["B"][24, 1],
                  "T_esperado_F": a["t"][24], "T_formula": dur_f[24],
                  "err_max_ruina": err_r, "err_max_tiempo": err_t})
    assert abs(a["B"].sum(axis=1) - 1).max() < 1e-10
tabla = pd.DataFrame(filas)
tabla.to_csv(os.path.join(RES, "parte1_markov_vs_formula.csv"), index=False)
print(tabla.round(5).to_string())

# 3) Figura: ruina y duracion vs capital inicial (ambas ruletas) + visitas esperadas
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
i = np.arange(1, N)
for nombre, p in ruedas.items():
    a = mk.analisis_absorcion(N, p)
    ax[0].plot(i, a["B"][:, 0], label=nombre)
    ax[1].plot(i, a["t"], label=nombre)
ax[0].set(xlabel="capital inicial i", ylabel="P(ruina)", title=f"Probabilidad de ruina (meta N={N})")
ax[1].set(xlabel="capital inicial i", ylabel="rondas esperadas", title="Tiempo esperado hasta absorcion")
ax[0].legend(fontsize=8)
a = mk.analisis_absorcion(N, mk.P_EUROPEA)
for i0 in (5, 25, 45):
    ax[2].plot(i, a["F"][i0 - 1], label=f"inicio i={i0}")
ax[2].set(xlabel="estado j", ylabel="F[i,j] (visitas esperadas)", title="Matriz fundamental (ruleta europea)")
ax[2].legend(fontsize=8)
guardar(fig, "p1_ruina_duracion_visitas.png")

# 4) Variante sin meta: solo ruina. Truncada en capital maximo C (reflejante)
C = 100
res_sm = {}
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for nombre, p in list(ruedas.items())[:2] + [list(ruedas.items())[3]]:
    P = mk.cadena_sin_meta(C, p)
    ev = mk.evolucion_distribucion(P, 20, 4000)
    ax[0].plot(ev[:, 0], label=nombre)
    qs, lam = mk.distribucion_cuasi_estacionaria(P)
    ax[1].plot(np.arange(1, C + 1), qs, label=f"{nombre}  lambda={lam:.5f}")
    res_sm[nombre] = {"P_ruina_en_4000": ev[-1, 0], "lambda": lam,
                      "semi_vida_rondas": float(np.log(0.5) / np.log(lam)) if lam < 1 else None}
ax[0].set(xlabel="ronda n", ylabel="P(arruinado en n)", title="Ruina acumulada (inicio i=20, sin meta)")
ax[1].set(xlabel="capital", ylabel="probabilidad", title="Distribucion cuasi-estacionaria (condicional a sobrevivir)")
ax[0].legend(fontsize=8); ax[1].legend(fontsize=7)
ii = np.arange(1, 61)
for nombre, p in ruedas.items():
    ax[2].plot(ii, mk.ruina_sin_meta(ii, p), label=nombre)
ax[2].set(xlabel="capital inicial i", ylabel="P(ruina alguna vez)", title="Ruina sin meta (horizonte infinito)")
ax[2].legend(fontsize=8)
guardar(fig, "p1_sin_meta.png")
out["sin_meta"] = res_sm

# 5) Extension: limite de mesa en Martingala (cadena exacta en (capital, racha))
N2, cap0 = 200, 100
filas = []
for nombre in ["Europea (p=18/37)", "Americana (p=18/38)", "Ventaja jugador (p=0.52)"]:
    p = ruedas[nombre]
    for L in (2, 4, 8, 16, 32, 64, 100, 1000):
        f, _, _ = mk.martingala_limite_mesa(N2, p, L)
        pr, T = f(cap0)
        filas.append({"juego": nombre, "limite_mesa": L, "P_ruina": pr, "T_esperado": T})
tm = pd.DataFrame(filas)
tm.to_csv(os.path.join(RES, "parte1_martingala_limite_mesa.csv"), index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for nombre, g in tm.groupby("juego"):
    ax[0].plot(g.limite_mesa, g.P_ruina, "o-", label=nombre)
    ax[1].plot(g.limite_mesa, g.T_esperado, "o-", label=nombre)
ax[0].axhline(mk.ruina_analitica(cap0, N2, mk.P_EUROPEA), ls=":", c="gray")
for a_ in ax: a_.set_xscale("log", base=2); a_.set_xlabel("limite de mesa (unidades)")
ax[0].set(ylabel="P(ruina)", title="Martingala exacta: capital 100 -> meta 200"); ax[1].set(ylabel="rondas esperadas", title="Duracion esperada")
ax[0].legend(fontsize=8)
guardar(fig, "p1_martingala_limite_mesa.png")
out["martingala_exacta_europea_limite100"] = tm[(tm.juego == "Europea (p=18/37)") & (tm.limite_mesa == 100)].to_dict("records")[0]
out["tabla_N50"] = filas and tabla.to_dict("records")
volcar("parte1.json", out)
print(tm.round(4).to_string())
print(res_sm)
