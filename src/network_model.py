"""Parte 3: aprendizaje social de apostadores en una red.

Cada nodo es un agente (Poblacion de engine.py). Cada `K` rondas ("epoca") los
agentes observan a sus vecinos y pueden imitar la estrategia del vecino que
mas gano en esa epoca. Los arruinados dejan de ser visibles (sesgo de
supervivencia: nadie ve su fracaso). Un "influencer" puede presumir solo su
mejor resultado reciente.
"""
import numpy as np
import networkx as nx
from .engine import *

ESTRATEGIAS_RED = [FIJA, MARTINGALA, PAROLI, PORCENTAJE, PERSECUCION, NO_APOSTAR]


def crear_red(tipo, n, grado_medio, rng):
    """ER, WS o BA con aproximadamente el mismo numero de aristas."""
    seed = int(rng.integers(1 << 31))
    k = int(grado_medio)
    if tipo == "ER":
        G = nx.gnm_random_graph(n, n * k // 2, seed=seed)
    elif tipo == "WS":
        G = nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    elif tipo == "BA":
        G = nx.barabasi_albert_graph(n, k // 2, seed=seed)
    else:
        raise ValueError(tipo)
    return G


def _aristas(G):
    e = np.array(G.edges(), dtype=int)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    return src, dst


def simular_red(G, estrategias0, reglas, epocas, K, rng, cap0=100.0, p_imitar=0.3,
                imitar=True, influencers=(), informantes=(), p_informar=0.3, ventana_presumir=5):
    """Simula la dinamica de imitacion.

    Devuelve dict con: pob (Poblacion final), compartir (epocas+1, S) con la
    proporcion de agentes por estrategia actual (los arruinados conservan su
    ultima estrategia), activos (idem solo entre agentes vivos), ruina (fraccion
    arruinada) y capital_medio por epoca.
    """
    n = G.number_of_nodes()
    pob = Poblacion(np.full(n, cap0), np.inf, np.asarray(estrategias0, int), reglas, rng)
    src, dst = _aristas(G)
    influencers = np.array(list(influencers), int)
    informantes = np.array(list(informantes), int)
    es_informante = np.zeros(n, bool); es_informante[informantes] = True
    pob.estrategia[informantes] = NO_APOSTAR
    S = len(ESTRATEGIAS_RED)
    pos = {s: i for i, s in enumerate(ESTRATEGIAS_RED)}

    def contar():
        lab = np.array([pos[s] for s in pob.estrategia])
        todos = np.bincount(lab, minlength=S) / n
        vivos = pob.estado == ACTIVO
        act = np.bincount(lab[vivos], minlength=S) / max(vivos.sum(), 1)
        return todos, act

    comp, actv, ruina, capm, capm_vivos = [], [], [], [], []
    hist = np.zeros((n, ventana_presumir))
    t, a = contar(); comp.append(t); actv.append(a); ruina.append(0.0); capm.append(pob.capital.mean()); capm_vivos.append(pob.capital.mean())
    for e in range(epocas):
        antes = pob.capital.copy()
        pob.jugar(K)
        ganancia = (pob.capital - antes) / cap0            # resultado de la epoca (en capitales iniciales)
        hist = np.roll(hist, 1, axis=1); hist[:, 0] = ganancia
        visible = ganancia.copy()
        if len(influencers):                                # solo presume su mejor resultado reciente
            visible[influencers] = np.maximum(hist[influencers].max(axis=1), 0.0)
        vivo = pob.estado == ACTIVO
        if imitar and e < epocas:
            v = np.where(vivo, visible, -np.inf)            # los arruinados no son visibles
            vd = v[dst] + 1e-9 * rng.random(len(dst))
            mejor = np.full(n, -np.inf)
            np.maximum.at(mejor, src, vd)
            es_mejor = vd >= mejor[src]
            vecino = np.full(n, -1)
            vecino[src[es_mejor]] = dst[es_mejor]
            cambia = (vivo & ~es_informante & (vecino >= 0) & (mejor > ganancia + 1e-9)
                      & (rng.random(n) < p_imitar))
            nuevo = pob.estrategia.copy()
            nuevo[cambia] = pob.estrategia[vecino[cambia]]
            # informantes: sus vecinos dejan de apostar con prob. p_informar
            if len(informantes):
                vecinos_inf = np.zeros(n, bool)
                m = np.isin(src, informantes)
                vecinos_inf[dst[m]] = True
                conv = vecinos_inf & vivo & ~es_informante & (rng.random(n) < p_informar)
                nuevo[conv] = NO_APOSTAR
                cambia |= conv
            cambia &= nuevo != pob.estrategia
            pob.estrategia[:] = nuevo
            pob.racha_perd[cambia] = 0; pob.racha_gan[cambia] = 0
            pob.pico[cambia] = pob.capital[cambia]
        t, a = contar(); comp.append(t); actv.append(a)
        ruina.append((pob.estado == RUINA).mean())
        capm.append(pob.capital.mean())
        capm_vivos.append(pob.capital[pob.estado == ACTIVO].mean() if (pob.estado == ACTIVO).any() else 0.0)
    return {"pob": pob, "compartir": np.array(comp), "activos": np.array(actv), "ruina": np.array(ruina),
            "capital_medio": np.array(capm), "capital_medio_vivos": np.array(capm_vivos)}
