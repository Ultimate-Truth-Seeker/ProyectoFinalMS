"""Parte 3: aprendizaje social en redes."""
from common import *
from src.network_model import *
from src.engine import NOMBRES
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

N_AG, GRADO, EPOCAS, K = 500, 6, 40, 25
TOPS = {"ER": "Erdos-Renyi", "WS": "Watts-Strogatz (mundo pequeno)", "BA": "Barabasi-Albert (libre de escala)"}
REGLAS = Reglas(p=18 / 37, limite_mesa=64, fraccion=0.05)
COL = {FIJA: "#2a6f97", MARTINGALA: "#d1495b", PAROLI: "#edae49", PORCENTAJE: "#00798c", PERSECUCION: "#8e5572", NO_APOSTAR: "#66a182"}
out = {}
LAB = [NOMBRES[s] for s in ESTRATEGIAS_RED]


def mezcla(rng, n=N_AG):
    return rng.choice(ESTRATEGIAS_RED, n)

# ---------------- E1: proporciones por topologia (poblacion mixta) -----------
R = 100
comp = {t: [] for t in TOPS}; act = {t: [] for t in TOPS}; ruina = {t: [] for t in TOPS}; ruina_ctrl = {t: [] for t in TOPS}
cap_t = {t: [] for t in TOPS}; cap_ctrl = {t: [] for t in TOPS}
for t in TOPS:
    for rep in range(R):
        rng = np.random.default_rng(10_000 * (list(TOPS).index(t) + 1) + rep)
        G = crear_red(t, N_AG, GRADO, rng)
        s0 = mezcla(rng)
        seed_state = rng.bit_generator.state
        r = simular_red(G, s0, REGLAS, EPOCAS, K, rng)
        comp[t].append(r["compartir"]); act[t].append(r["activos"]); ruina[t].append(r["ruina"]); cap_t[t].append(r["capital_medio"])
        rng2 = np.random.default_rng(10_000 * (list(TOPS).index(t) + 1) + rep + 500_000)
        rc = simular_red(G, s0, REGLAS, EPOCAS, K, rng2, imitar=False)
        ruina_ctrl[t].append(rc["ruina"]); cap_ctrl[t].append(rc["capital_medio"])
comp = {t: np.array(v) for t, v in comp.items()}; act = {t: np.array(v) for t, v in act.items()}
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharey="row")
for j, t in enumerate(TOPS):
    for i, s in enumerate(ESTRATEGIAS_RED):
        m = comp[t][:, :, i].mean(0); sd = comp[t][:, :, i].std(0)
        axs[0, j].plot(m, c=COL[s], label=NOMBRES[s]); axs[0, j].fill_between(range(EPOCAS + 1), m - sd, m + sd, color=COL[s], alpha=0.12)
        axs[1, j].plot(act[t][:, :, i].mean(0), c=COL[s])
    axs[0, j].plot(np.mean(ruina[t], 0), "k--", lw=1.2, label="fraccion arruinada")
    axs[0, j].set_title(TOPS[t]); axs[1, j].set_xlabel(f"epoca ({K} rondas c/u)")
axs[0, 0].set_ylabel("proporcion de agentes (todos)"); axs[1, 0].set_ylabel("proporcion entre agentes activos")
axs[0, 0].legend(fontsize=7)
guardar(fig, "p3_proporciones_topologias.png")
filas = []
for t in TOPS:
    fin = comp[t][:, -1, :].mean(0)
    d = {"topologia": t, "ruina_final_imitando": np.mean(ruina[t], 0)[-1], "ruina_final_sin_imitar": np.mean(ruina_ctrl[t], 0)[-1],
         "capital_medio_imitando": np.mean(cap_t[t], 0)[-1], "capital_medio_sin_imitar": np.mean(cap_ctrl[t], 0)[-1]}
    d["ruina_ic95"] = 1.96 * np.std(np.array(ruina[t])[:, -1]) / np.sqrt(R)
    for s, v in zip(LAB, fin): d["share_final_" + s] = v
    for s, v in zip(LAB, comp[t][:, 0, :].mean(0)): d["share_inicial_" + s] = v
    # pico de martingala
    mm = comp[t][:, :, 1].mean(0); d["martingala_pico"] = mm.max(); d["epoca_pico_martingala"] = int(mm.argmax())
    filas.append(d)
e1 = pd.DataFrame(filas); e1.to_csv(os.path.join(RES, "parte3_proporciones.csv"), index=False)
print(e1.round(3).T)

# ---------------- E2: semilla unica (grado del nodo inicial) ------------------
R2 = 100
filas = []; curvas = {}
for t in TOPS:
    for tipo in ["hub (grado maximo)", "hub presumido", "aleatorio", "hoja (grado minimo)"]:
        pk, shares, ruinas, deg_s, t25, s10 = [], [], [], [], [], []
        c = []
        for rep in range(R2):
            rng = np.random.default_rng(77_000 + 1000 * list(TOPS).index(t) + rep)
            G = crear_red(t, N_AG, GRADO, rng)
            deg = np.array([G.degree(i) for i in range(N_AG)])
            if tipo.startswith("hub"): sd = int(rng.choice(np.flatnonzero(deg == deg.max())))
            elif tipo.startswith("hoja"): sd = int(rng.choice(np.flatnonzero(deg == deg[deg > 0].min())))
            else: sd = int(rng.integers(N_AG))
            s0 = np.full(N_AG, FIJA); s0[sd] = MARTINGALA
            r = simular_red(G, s0, REGLAS, EPOCAS, K, rng, influencers=[sd] if tipo == "hub presumido" else ())
            m = r["compartir"][:, 1]
            c.append(m); t25.append(np.argmax(m >= 0.25) if (m >= 0.25).any() else EPOCAS + 1); s10.append(m[10]); pk.append(m.max()); shares.append(m[-1]); ruinas.append(r["ruina"][-1]); deg_s.append(deg[sd])
        filas.append({"topologia": t, "semilla": tipo, "grado_semilla": np.mean(deg_s), "martingala_max_share": np.mean(pk),
                      "martingala_share_final": np.mean(shares), "martingala_share_epoca10": np.mean(s10), "epoca_al_25pct": np.mean(t25), "prob_alcanza_10pct": np.mean(np.array(pk) >= 0.10),
                      "ruina_final": np.mean(ruinas)})
        curvas[(t, tipo)] = np.mean(c, 0)
e2 = pd.DataFrame(filas); e2.to_csv(os.path.join(RES, "parte3_semilla.csv"), index=False)
print(e2.round(3).to_string())
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for j, t in enumerate(TOPS):
    for tipo, ls in zip(["hub (grado maximo)", "hub presumido", "aleatorio", "hoja (grado minimo)"], ["-", "-", "--", ":"]):
        axs[j].plot(curvas[(t, tipo)], ls=ls, lw=2 if "presumido" in tipo else 1.4, label=tipo)
    axs[j].set_title(TOPS[t]); axs[j].set_xlabel("epoca")
axs[0].set_ylabel("proporcion de Martingala (inicia 1 agente de 500)"); axs[0].legend(fontsize=8)
guardar(fig, "p3_semilla_influencer.png")

# ---------------- E3: metricas de red vs riqueza final -----------------------
R3 = 20
filas = []; scat = None
for t in TOPS:
    rows = []
    for rep in range(R3):
        rng = np.random.default_rng(900_000 + 100 * list(TOPS).index(t) + rep)
        G = crear_red(t, N_AG, GRADO, rng)
        s0 = mezcla(rng)
        r = simular_red(G, s0, REGLAS, EPOCAS, K, rng)
        cap = r["pob"].capital
        deg = np.array([G.degree(i) for i in range(N_AG)])
        bet = np.array(list(nx.betweenness_centrality(G).values()))
        clu = np.array(list(nx.clustering(G).values()))
        cls = np.array(list(nx.closeness_centrality(G).values()))
        com = nx.community.greedy_modularity_communities(G)
        lab = np.zeros(N_AG, int)
        for ci, cc in enumerate(com): lab[list(cc)] = ci
        # riqueza media de la comunidad (media de los demas miembros) como medida de efecto de comunidad
        cmean = np.array([cap[lab == lab[i]].mean() for i in range(N_AG)])
        d = {"modularidad": nx.community.modularity(G, com), "n_comunidades": len(com), "clustering_medio": clu.mean(),
             "camino_medio": nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan}
        for nom, x in [("grado", deg), ("intermediacion", bet), ("cercania", cls), ("agrupamiento", clu), ("riqueza_comunidad", cmean)]:
            rho = spearmanr(x, cap).statistic
            d["rho_" + nom] = rho
        # riqueza final de los que arrancaron como cada estrategia
        rows.append(d)
        if t == "BA" and rep == 0: scat = (deg, cap, s0)
    df = pd.DataFrame(rows); m = df.mean().astype(object); m["topologia"] = t
    filas.append(m)
e3 = pd.DataFrame(filas).set_index("topologia"); e3.to_csv(os.path.join(RES, "parte3_metricas_red.csv"))
print(e3.round(3).T)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cols = [c for c in e3.columns if c.startswith("rho_")]
w = 0.25
for j, t in enumerate(TOPS):
    axs[0].bar(np.arange(len(cols)) + (j - 1) * w, e3.loc[t, cols].astype(float), w, label=t)
axs[0].set_xticks(range(len(cols))); axs[0].set_xticklabels([c[4:] for c in cols], rotation=20, fontsize=8)
axs[0].axhline(0, c="k", lw=0.6); axs[0].set(ylabel="correlacion de Spearman con capital final", title="Metricas de red vs riqueza final"); axs[0].legend(fontsize=8)
deg, cap, s0 = scat
for s in ESTRATEGIAS_RED:
    mk_ = s0 == s
    axs[1].scatter(deg[mk_] + np.random.default_rng(0).uniform(-0.2, 0.2, mk_.sum()), cap[mk_], s=10, alpha=0.6, c=COL[s], label=NOMBRES[s])
axs[1].set_xscale("log"); axs[1].set(xlabel="grado del nodo (BA)", ylabel="capital final", title="Capital final vs grado, color = estrategia inicial"); axs[1].legend(fontsize=6)
guardar(fig, "p3_metricas_red.png")
out["metricas"] = e3.reset_index().to_dict("records")

# ---------------- E4: intervenciones ---------------------------------------------
R4 = 60
configs = {
    "Base (limite 64)": dict(limite=64, inf=0),
    "Limite de mesa 8": dict(limite=8, inf=0),
    "Limite de mesa 2": dict(limite=2, inf=0),
    "5 informantes en hubs": dict(limite=64, inf=5),
    "20 informantes en hubs": dict(limite=64, inf=20),
    "Limite 8 + 5 informantes": dict(limite=8, inf=5),
}
filas = []
for t in TOPS:
    for nom, cfg in configs.items():
        ru, ca, mar, nap, apo = [], [], [], [], []
        for rep in range(R4):
            rng = np.random.default_rng(4_000_000 + 100 * list(TOPS).index(t) + rep)
            G = crear_red(t, N_AG, GRADO, rng)
            s0 = mezcla(rng)
            deg = np.array([G.degree(i) for i in range(N_AG)])
            infs = np.argsort(-deg)[:cfg["inf"]]
            reg = Reglas(p=18 / 37, limite_mesa=cfg["limite"], fraccion=0.05)
            r = simular_red(G, s0, reg, EPOCAS, K, rng, informantes=infs)
            ru.append(r["ruina"][-1]); ca.append(r["capital_medio"][-1])
            apo.append(r["pob"].apostado.mean()); mar.append(r["compartir"][-1, 1]); nap.append(r["compartir"][-1, 5])
        filas.append({"topologia": t, "intervencion": nom, "ruina_final": np.mean(ru), "capital_medio_final": np.mean(ca),
                      "capital_ic95": 1.96 * np.std(ca) / np.sqrt(R4), "apostado_medio": np.mean(apo), "martingala_final": np.mean(mar), "no_apostar_final": np.mean(nap)})
e4 = pd.DataFrame(filas); e4.to_csv(os.path.join(RES, "parte3_intervenciones.csv"), index=False)
print(e4.round(3).to_string())
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for j, t in enumerate(TOPS):
    g = e4[e4.topologia == t]
    axs[j].barh(g.intervencion, g.capital_medio_final, xerr=g.capital_ic95, color=["#555555"] + ["#2a6f97"] * 2 + ["#66a182"] * 2 + ["#8e5572"])
    axs[j].axvline(100, c="k", ls=":"); axs[j].set_title(TOPS[t]); axs[j].set_xlabel("capital medio final (inicial = 100)")
    axs[j].invert_yaxis()
guardar(fig, "p3_intervenciones.png")

# ---------------- visual de redes -----------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(13, 8))
for j, t in enumerate(TOPS):
    rng = np.random.default_rng(5 + j)
    n = 120
    G = crear_red(t, n, GRADO, rng)
    s0 = mezcla(rng, n)
    r = simular_red(G, s0, REGLAS, 15, K, rng)
    pos = nx.spring_layout(G, seed=3)
    deg = np.array([G.degree(i) for i in range(n)])
    for row, (estr, tit) in enumerate([(s0, "inicio"), (r["pob"].estrategia, "epoca 15")]):
        ax = axs[row, j]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5)
        nc = [COL[s] for s in estr]
        alpha_ = np.where(r["pob"].estado == RUINA, 0.25, 1.0) if row else np.ones(n)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=15 + 6 * deg, alpha=None, linewidths=0)
        if row:
            ar = np.flatnonzero(r["pob"].estado == RUINA)
            nx.draw_networkx_nodes(G, pos, nodelist=list(ar), ax=ax, node_color="white", edgecolors="black", node_size=15 + 6 * deg[ar], linewidths=0.8)
        ax.set_title(f"{TOPS[t]} - {tit}", fontsize=9); ax.axis("off")
from matplotlib.lines import Line2D
axs[0, 0].legend(handles=[Line2D([], [], marker="o", ls="", c=COL[s], label=NOMBRES[s]) for s in ESTRATEGIAS_RED] +
                 [Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="arruinado")], fontsize=6, loc="lower left")
guardar(fig, "p3_redes.png")
volcar("parte3.json", out)
