"""Parte 2: ABM. Validacion, estrategias, casa y sensibilidad."""
from common import *
from src import markov as mk
from src.engine import *
import pandas as pd
from scipy import stats

out = {}
rng = np.random.default_rng(2024)
PE, PA = mk.P_EUROPEA, mk.P_AMERICANA
ORDEN = [FIJA, MARTINGALA, PAROLI, PORCENTAJE, KELLY, FALACIA, PERSECUCION]

# ---------- A) Validacion: apuesta fija vs cadena de Markov ----------------
N, n = 50, 20000
casos = [i for i in range(2, 50, 4)]
filas = []
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, (nombre, p) in enumerate({"Europea": PE, "Americana": PA}.items()):
    sim_r, sim_t, ci = [], [], []
    for i in casos:
        pob = simular(FIJA, n, i, N, Reglas(p=p), max_rondas=200000, seed=100 + i)
        pr = (pob.estado == RUINA).mean()
        sim_r.append(pr); sim_t.append(pob.rondas.mean())
        se = np.sqrt(pr * (1 - pr) / n)
        ci.append(1.96 * se)
        teo = float(mk.ruina_analitica(i, N, p))
        z = (pr - teo) / max(se, 1e-9)
        filas.append({"juego": nombre, "i": i, "ruina_sim": pr, "ruina_teo": teo, "z": z,
                      "T_sim": pob.rondas.mean(), "T_teo": float(mk.duracion_analitica(i, N, p))})
    ii = np.arange(1, N)
    ax[k].plot(ii, mk.ruina_analitica(ii, N, p), "-", label="Formula / F·R (Markov)")
    ax[k].errorbar(casos, sim_r, yerr=ci, fmt="o", ms=4, capsize=2, c="#d1495b", label=f"Simulacion ABM (n={n})")
    ax[k].set(xlabel="capital inicial i", ylabel="P(ruina)", title=f"Ruleta {nombre.lower()} (p={p:.4f}), meta N={N}")
    ax[k].legend(fontsize=8)
guardar(fig, "p2_validacion_ruina.png")
val = pd.DataFrame(filas)
val.to_csv(os.path.join(RES, "parte2_validacion_fija.csv"), index=False)
out["validacion"] = {
    "max_abs_err_ruina": float((val.ruina_sim - val.ruina_teo).abs().max()),
    "frac_dentro_IC95": float((val.z.abs() <= 1.96).mean()),
    "max_rel_err_T": float(((val.T_sim - val.T_teo).abs() / val.T_teo).max()),
    "n_agentes": n}
print("validacion", out["validacion"])

# validacion martingala con limite de mesa vs cadena exacta
filas = []
for L in (8, 16, 32, 100):
    f, _, _ = mk.martingala_limite_mesa(200, PE, L)
    pr_t, T_t = f(100)
    pob = simular(MARTINGALA, 50000, 100, 200, Reglas(p=PE, limite_mesa=L), max_rondas=100000, seed=L)
    pr_s = (pob.estado == RUINA).mean()
    filas.append({"limite": L, "ruina_exacta": pr_t, "ruina_sim": pr_s, "T_exacto": T_t, "T_sim": pob.rondas.mean()})
vm = pd.DataFrame(filas)
vm.to_csv(os.path.join(RES, "parte2_validacion_martingala.csv"), index=False)
print(vm.round(4))

# ---------- B) Trayectorias --------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True)
cap0, meta = 100, 200
for ax_, s in zip(axs.ravel(), ORDEN):
    r = Reglas(p=PE, limite_mesa=100, fraccion=0.05)
    tray, pob = trayectorias(s, 10, cap0, meta, r, 1500, seed=7)
    for j in range(10):
        ax_.plot(tray[:, j], lw=0.9, alpha=0.85)
    ax_.axhline(meta, c="green", ls="--", lw=1); ax_.axhline(0, c="red", ls="--", lw=1)
    ax_.set_title(NOMBRES[s]); ax_.set_xlabel("ronda")
    ax_.set_ylim(-10, 260)
axs[0, 0].set_ylabel("capital"); axs[1, 0].set_ylabel("capital")
axs[1, 3].axis("off")
axs[1, 3].text(0, 0.6, "verde: meta (200)\nrojo: ruina (0)\nruleta europea,\ncapital inicial 100,\nlimite de mesa 100\n(Kelly: no apuesta)", fontsize=10)
guardar(fig, "p2_trayectorias.png")

# ---------- C) Comparacion de estrategias (ruleta europea) -------------------
n = 20000
def tabla_estrategias(p, cap0, meta, reglas, estrategias, n, max_rondas, seed):
    filas, pobs = [], {}
    for s in estrategias:
        pob = simular(s, n, cap0, meta, reglas, max_rondas=max_rondas, seed=seed + s)
        d = resumen(pob)
        d["estrategia"] = NOMBRES[s]
        d["asimetria_capital_final"] = float(stats.skew(pob.capital))
        d["perdida_por_unidad_apostada"] = d["perdida_esperada"] / d["apostado_medio"] if d["apostado_medio"] > 0 else np.nan
        d["max_racha_perdedora_media"] = pob.max_racha_perd.mean()
        filas.append(d); pobs[s] = pob
    df = pd.DataFrame(filas).set_index("estrategia")
    return df, pobs

r = Reglas(p=PE, limite_mesa=100, fraccion=0.05)
tE, pE = tabla_estrategias(PE, 100, 200, r, ORDEN, n, 20000, 1000)
tE.to_csv(os.path.join(RES, "parte2_estrategias_europea.csv"))
print(tE[["p_ruina", "p_meta", "p_sin_terminar", "capital_medio", "capital_mediano", "duracion_media", "perdida_esperada", "perdida_por_unidad_apostada"]].round(3))
rA = Reglas(p=PA, limite_mesa=100, fraccion=0.05)
tA, pA = tabla_estrategias(PA, 100, 200, rA, ORDEN, n, 20000, 2000)
tA.to_csv(os.path.join(RES, "parte2_estrategias_americana.csv"))

fig, axs = plt.subplots(2, 4, figsize=(15, 6.5))
for ax_, s in zip(axs.ravel(), ORDEN):
    cf = pE[s].capital
    ax_.hist(cf, bins=np.linspace(0, 260, 53), color="#2a6f97", alpha=0.9)
    ax_.axvline(100, c="k", ls=":", lw=1); ax_.axvline(cf.mean(), c="#d1495b", lw=1.5)
    ax_.set_title(f"{NOMBRES[s]}\nmedia={cf.mean():.0f}, mediana={np.median(cf):.0f}", fontsize=9.5)
    ax_.set_yscale("log"); ax_.set_xlabel("capital final")
axs[1, 3].axis("off")
axs[1, 3].text(0, 0.5, "Punteada: capital inicial (100)\nRoja: media del capital final\nEscala vertical logaritmica\nn=20,000 agentes por estrategia", fontsize=10)
guardar(fig, "p2_histogramas_capital_final.png")

# efecto de la varianza: Martingala sin limite de mesa gana poco muchas veces
r0 = Reglas(p=PE, limite_mesa=np.inf)
pm = simular(MARTINGALA, n, 100, 110, r0, max_rondas=30000, seed=5)  # meta pequena
pf = simular(FIJA, n, 100, 110, r0, max_rondas=30000, seed=5)
out["varianza_meta110"] = {
    "martingala": {"p_meta": float((pm.estado == META).mean()), "p_ruina": float((pm.estado == RUINA).mean()),
                    "capital_medio": float(pm.capital.mean()), "capital_mediano": float(np.median(pm.capital))},
    "fija": {"p_meta": float((pf.estado == META).mean()), "p_ruina": float((pf.estado == RUINA).mean()),
              "capital_medio": float(pf.capital.mean()), "capital_mediano": float(np.median(pf.capital))}}
print(out["varianza_meta110"])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax_, pob, t in zip(ax, [pf, pm], ["Apuesta fija 1u", "Martingala (sin limite de mesa)"]):
    ax_.hist(pob.capital, bins=np.linspace(0, 120, 61), color="#2a6f97")
    ax_.set_yscale("log"); ax_.set_title(f"{t}: meta +10, capital inicial 100\nmedia={pob.capital.mean():.1f}  gana {np.mean(pob.estado==META):.1%}  quiebra {np.mean(pob.estado==RUINA):.1%}", fontsize=9.5)
    ax_.set_xlabel("capital final")
guardar(fig, "p2_martingala_asimetria.png")

# ---------- C2) Juego con ventaja (p=0.52): Kelly ---------------------------
rK = lambda kf, **kw: Reglas(p=0.52, kelly_frac=kf, limite_mesa=np.inf, **kw)
filas = []
pobsK = {}
for nombre, s, reg in [("Apuesta fija 1u", FIJA, rK(1)), ("Martingala (limite 100)", MARTINGALA, Reglas(p=0.52, limite_mesa=100)),
                       ("Porcentaje 5%", PORCENTAJE, rK(1, fraccion=0.05)), ("Porcentaje 20% (sobreapuesta)", PORCENTAJE, rK(1, fraccion=0.20)),
                       ("Medio Kelly (2%)", KELLY, rK(0.5)), ("Kelly completo (4%)", KELLY, rK(1.0)), ("Doble Kelly (8%)", KELLY, rK(2.0))]:
    pob = simular(s, 10000, 100, 400, reg, max_rondas=20000, seed=77)
    d = resumen(pob); d["estrategia"] = nombre
    d["log_crecimiento_medio"] = float(np.mean(np.log(np.maximum(pob.capital, 1e-9) / 100) / np.maximum(pob.rondas, 1)))
    filas.append(d); pobsK[nombre] = pob
tK = pd.DataFrame(filas).set_index("estrategia")
tK.to_csv(os.path.join(RES, "parte2_kelly_p052.csv"))
print(tK[["p_ruina", "p_meta", "p_sin_terminar", "capital_medio", "capital_mediano", "duracion_media"]].round(3))
# horizonte fijo (sin meta): riqueza tras T=2000 rondas
T_K = 2000
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for nombre, f in [("Medio Kelly (2%)", 0.5), ("Kelly completo (4%)", 1.0), ("Doble Kelly (8%)", 2.0), ("Triple Kelly (12%)", 3.0)]:
    c = simular(KELLY, 10000, 100, np.inf, rK(f), max_rondas=T_K, seed=21).capital
    ax[0].hist(np.log10(np.maximum(c, 0.3)), bins=60, alpha=0.5, label=f"{nombre}: mediana={np.median(c):.0f}, media={c.mean():.0f}")
ax[0].set(xlabel="log10(capital tras 2000 rondas)", ylabel="agentes", title="p=0.52: distribucion de la riqueza (capital inicial 100)")
ax[0].legend(fontsize=7)
kf = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3])
med, mean_, ruin, lg = [], [], [], []
for f in kf:
    c = simular(KELLY, 6000, 100, np.inf, rK(f), max_rondas=T_K, seed=5)
    med.append(np.median(c.capital)); mean_.append(c.capital.mean()); ruin.append((c.estado == RUINA).mean())
    lg.append(np.mean(np.log(np.maximum(c.capital, 1e-9) / 100)) / T_K)
ax[1].plot(kf, np.array(lg) * 1000, "o-", label="crecimiento log medio por ronda (x1000)")
ax[1].axvline(1, c="gray", ls=":"); ax[1].set(xlabel="fraccion de Kelly (1 = f*=4%)", ylabel="crecimiento logaritmico (x1000)", title="Sensibilidad a la fraccion de Kelly (p=0.52)")
ax2 = ax[1].twinx(); ax2.plot(kf, ruin, "s--", c="#d1495b"); ax2.set_ylabel("P(capital<1) (rojo)"); ax2.grid(False)
ax[1].legend(loc="lower left", fontsize=8)
guardar(fig, "p2_kelly.png")
out["kelly_sens"] = {"frac": kf.tolist(), "mediana": med, "media": mean_, "ruina": ruin, "log_crec": lg}

# ---------- D) Casa ------------------------------------------------------------
T = 5000
rngc = np.random.default_rng(11)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for m in (1, 10, 100, 1000):
    g = np.where(rngc.random((T, m)) < PE, -1, 1).sum(axis=1)  # ganancia de la casa por ronda (todos apuestan 1)
    ax[0].plot(np.cumsum(g), label=f"{m} apostadores")
ax[0].plot(np.arange(1, T + 1) * 0 , c="k", lw=0.5)
ax[0].set(xlabel="ronda", ylabel="ganancia acumulada de la casa (u)", title="Ganancia acumulada de la casa"); ax[0].legend(fontsize=8)
for m in (1, 10, 100, 1000, 10000):
    reps = np.array([np.cumsum(np.where(rngc.random((T, m)) < PE, -1, 1).sum(axis=1)) / (m * np.arange(1, T + 1)) for _ in range(8)]) if m <= 100 else \
           np.array([np.cumsum(np.where(rngc.random((T, m)) < PE, -1, 1).sum(axis=1)) / (m * np.arange(1, T + 1)) for _ in range(2)])
    ax[1].plot(reps.mean(0), lw=1) if False else None
    for rr in reps[:3]:
        ax[1].plot(rr, lw=0.8, c=f"C{[1,10,100,1000,10000].index(m)}", label=f"{m} apostadores" if rr is reps[0] else None)
ax[1].axhline(1 - 2 * PE, c="k", ls="--", lw=1, label="ventaja teorica 2.70%")
ax[1].set(xlabel="ronda", ylabel="ganancia casa / unidad apostada", title="Ley de los grandes numeros"); ax[1].set_ylim(-0.4, 0.5); ax[1].legend(fontsize=7)
# desviacion de la ganancia por unidad apostada vs numero de apuestas
ns = np.array([10, 100, 1000, 10000, 100000])
sd = [np.std((1 - 2 * (rngc.random((300, k)) < PE).mean(1))) for k in ns]
ax[2].loglog(ns, sd, "o-", label="simulado"); ax[2].loglog(ns, 1 / np.sqrt(ns), "k--", label=r"$1/\sqrt{n}$")
ax[2].axhline(1 - 2 * PE, c="gray", ls=":")
ax[2].set(xlabel="numero total de apuestas", ylabel="desv. est. de la ganancia/unidad", title="Cuando la casa deja de tener riesgo"); ax[2].legend(fontsize=8)
guardar(fig, "p2_casa.png")

# casa con estrategias reales: ganancia = suma de (capital inicial - final)
filas = []
for s in ORDEN:
    for m in (10, 100, 1000, 10000):
        pob = pE[s]
        idx = np.arange(m)
        filas.append({"estrategia": NOMBRES[s], "agentes": m, "ganancia_casa_por_agente": (pob.capital0[idx] - pob.capital[idx]).mean(),
                      "apostado": pob.apostado[idx].sum(), "ganancia_casa": (pob.capital0[idx] - pob.capital[idx]).sum()})
th = pd.DataFrame(filas); th.to_csv(os.path.join(RES, "parte2_casa.csv"), index=False)

# ---------- E) Sensibilidad ----------------------------------------------------
n = 5000
ps = np.round(np.linspace(0.44, 0.54, 11), 3)
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
sens = {}
for s in [FIJA, MARTINGALA, PORCENTAJE, FALACIA]:
    y = []
    for p in ps:
        pob = simular(s, n, 100, 200, Reglas(p=p, limite_mesa=100, fraccion=0.05), max_rondas=20000, seed=3)
        y.append((pob.estado == RUINA).mean())
    axs[0, 0].plot(ps, y, "o-", ms=3, label=NOMBRES[s]); sens[f"p_{s}"] = y
axs[0, 0].axvline(PE, c="gray", ls=":"); axs[0, 0].axvline(PA, c="gray", ls=":"); axs[0, 0].axvline(.5, c="k", ls="--", lw=.8)
axs[0, 0].set(xlabel="p (prob. de ganar)", ylabel="P(ruina)", title="Sensibilidad a p (100 -> 200)"); axs[0, 0].legend(fontsize=7)
caps = [20, 40, 60, 80, 100, 120, 150, 180]
for s in [FIJA, MARTINGALA, PORCENTAJE]:
    y = [(simular(s, n, c0, 200, Reglas(p=PE, limite_mesa=100, fraccion=0.05), max_rondas=20000, seed=4).estado == RUINA).mean() for c0 in caps]
    axs[0, 1].plot(caps, y, "o-", ms=3, label=NOMBRES[s]); sens[f"cap_{s}"] = y
axs[0, 1].plot(np.arange(1, 200), mk.ruina_analitica(np.arange(1, 200), 200, PE), "k:", label="analitica (fija)")
axs[0, 1].set(xlabel="capital inicial", ylabel="P(ruina)", title="Sensibilidad al capital inicial (meta 200)"); axs[0, 1].legend(fontsize=7)
metas = [110, 125, 150, 200, 300, 500, 1000]
for s in [FIJA, MARTINGALA, PORCENTAJE]:
    y = [(simular(s, n, 100, m_, Reglas(p=PE, limite_mesa=100, fraccion=0.05), max_rondas=30000, seed=6).estado == RUINA).mean() for m_ in metas]
    axs[0, 2].plot(metas, y, "o-", ms=3, label=NOMBRES[s]); sens[f"meta_{s}"] = y
axs[0, 2].set_xscale("log"); axs[0, 2].set(xlabel="meta", ylabel="P(ruina)", title="Sensibilidad a la meta (capital inicial 100)"); axs[0, 2].legend(fontsize=7)
lims = [2, 4, 8, 16, 32, 64, 128, 512]
for nombre, p in [("europea", PE), ("americana", PA), ("p=0.52", 0.52)]:
    y = [(simular(MARTINGALA, n, 100, 200, Reglas(p=p, limite_mesa=L), max_rondas=30000, seed=8).estado == RUINA).mean() for L in lims]
    axs[1, 0].plot(lims, y, "o-", ms=3, label=nombre); sens[f"lim_{nombre}"] = y
axs[1, 0].set_xscale("log", base=2); axs[1, 0].set(xlabel="limite de mesa", ylabel="P(ruina)", title="Martingala: limite de mesa"); axs[1, 0].legend(fontsize=7)
# perdida por unidad apostada vs estrategia y p  (la ventaja no cambia)
for s in [FIJA, MARTINGALA, PAROLI, PORCENTAJE]:
    y = []
    for p in ps:
        pob = simular(s, n, 100, 200, Reglas(p=p, limite_mesa=100, fraccion=0.05), max_rondas=20000, seed=9)
        y.append((pob.capital0 - pob.capital).mean() / pob.apostado.mean())
    axs[1, 1].plot(ps, y, "o-", ms=3, label=NOMBRES[s]); sens[f"edge_{s}"] = y
axs[1, 1].plot(ps, 1 - 2 * ps, "k--", lw=1, label="1-2p"); axs[1, 1].set(xlabel="p", ylabel="perdida / unidad apostada", title="La ventaja de la casa no depende de la estrategia"); axs[1, 1].legend(fontsize=7)
# capital medio final por estrategia vs capital esperado teorico
axs[1, 2].axis("off")
guardar(fig, "p2_sensibilidad.png")
volcar("parte2_sens.json", sens)
volcar("parte2.json", out)
