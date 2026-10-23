"""Parte 2: modelo basado en agentes (vectorizado con NumPy).

Cada apostador es un agente con capital, capital inicial, meta, estrategia e
historial (rachas, perdidas acumuladas). Se guarda como arreglos para poder
simular decenas de miles de agentes a la vez; cada fila es un agente
independiente. Los agentes pueden tener estrategias distintas (se usa en la
Parte 3).
"""
from dataclasses import dataclass, field
import numpy as np

# identificadores de estrategia
FIJA, MARTINGALA, PAROLI, PORCENTAJE, KELLY, FALACIA, PERSECUCION, NO_APOSTAR = range(8)
NOMBRES = {
    FIJA: "Apuesta fija", MARTINGALA: "Martingala", PAROLI: "Anti-Martingala (Paroli)",
    PORCENTAJE: "Porcentaje fijo", KELLY: "Kelly", FALACIA: "Falacia del jugador",
    PERSECUCION: "Persecucion de perdidas", NO_APOSTAR: "No apostar",
}

# estado de un agente
ACTIVO, RUINA, META = 0, 1, 2


@dataclass
class Reglas:
    """Reglas del juego y parametros de las estrategias."""
    p: float = 18 / 37          # prob. de ganar (pago 1:1)
    base: float = 1.0           # apuesta base (unidades)
    limite_mesa: float = np.inf # apuesta maxima
    fraccion: float = 0.05      # f para PORCENTAJE
    kelly_frac: float = 1.0     # 1 = Kelly completo, 0.5 = medio Kelly
    paroli_max: int = 3         # victorias seguidas antes de retirar ganancias
    persecucion: float = 0.5    # fraccion de la perdida reciente que se intenta recuperar
    apuesta_min: float = 1.0    # por debajo de esto el agente ya no puede apostar (ruina)


class Poblacion:
    """Conjunto de agentes. Todos los atributos son arreglos de longitud n."""

    def __init__(self, capital0, meta, estrategia, reglas: Reglas, rng):
        n = len(capital0)
        self.n = n
        self.reglas = reglas
        self.rng = rng
        self.capital0 = np.asarray(capital0, float).copy()
        self.capital = self.capital0.copy()
        self.meta = np.broadcast_to(np.asarray(meta, float), (n,)).copy()
        self.estrategia = np.broadcast_to(np.asarray(estrategia, int), (n,)).copy()
        self.estado = np.zeros(n, int)
        self.racha_perd = np.zeros(n, int)   # historial: racha perdedora actual
        self.racha_gan = np.zeros(n, int)    # racha ganadora actual
        self.max_racha_perd = np.zeros(n, int)
        self.pico = self.capital0.copy()     # maximo capital alcanzado
        self.perdida_acum = np.zeros(n)      # perdidas acumuladas brutas
        self.rondas = np.zeros(n, int)
        self.apostado = np.zeros(n)
        self.fuera = np.zeros(n, bool)       # decidio no apostar (Kelly con ventaja negativa)
        self._chequear_estado(np.arange(n))

    # -- apuestas -------------------------------------------------------
    def _apuesta(self, ix):
        r = self.reglas
        s = self.estrategia[ix]
        cap = self.capital[ix]
        b = np.full(len(ix), r.base)
        m = s == MARTINGALA
        b[m] = r.base * 2.0 ** np.minimum(self.racha_perd[ix][m], 60)
        m = s == PAROLI
        b[m] = r.base * 2.0 ** (self.racha_gan[ix][m] % r.paroli_max)
        m = s == PORCENTAJE
        b[m] = r.fraccion * cap[m]
        m = s == KELLY
        f = max(2 * r.p - 1, 0.0) * r.kelly_frac
        b[m] = f * cap[m]
        m = s == FALACIA   # "ya toca ganar": sube la apuesta con la racha perdedora
        b[m] = r.base * (1 + self.racha_perd[ix][m])
        m = s == PERSECUCION  # intenta recuperar parte de lo perdido desde el pico
        b[m] = r.base + r.persecucion * (self.pico[ix][m] - cap[m])
        b = np.minimum(np.minimum(b, r.limite_mesa), cap)
        b[s == NO_APOSTAR] = 0.0
        return b

    def _chequear_estado(self, ix):
        r = self.reglas
        cap = self.capital[ix]
        arruinado = cap < r.apuesta_min
        gano = cap >= self.meta[ix]
        self.estado[ix[arruinado]] = RUINA
        self.estado[ix[gano & ~arruinado]] = META

    def jugar(self, k=1, ix=None):
        """Juega k rondas para los agentes activos (o el subconjunto ix).

        Devuelve el cambio de capital por agente en esas rondas.
        """
        r = self.reglas
        if ix is None:
            ix = np.arange(self.n)
        cap_ini = self.capital[ix].copy()
        for _ in range(k):
            act = ix[self.estado[ix] == ACTIVO]
            if len(act) == 0:
                break
            b = self._apuesta(act)
            # una estrategia que apuesta 0 se queda fuera del juego
            sin_apuesta = b <= 0
            if sin_apuesta.any():
                self.fuera[act[sin_apuesta]] = True
                self.estado[act[sin_apuesta]] = ACTIVO
            jug = ~sin_apuesta
            a, b = act[jug], b[jug]
            if len(a) == 0:
                break
            gana = self.rng.random(len(a)) < r.p
            self.capital[a] += np.where(gana, b, -b)
            self.apostado[a] += b
            self.rondas[a] += 1
            g, l = a[gana], a[~gana]
            self.racha_gan[g] += 1
            self.racha_perd[g] = 0
            self.racha_perd[l] += 1
            self.racha_gan[l] = 0
            self.max_racha_perd[l] = np.maximum(self.max_racha_perd[l], self.racha_perd[l])
            self.perdida_acum[l] += b[~gana]
            self.pico[a] = np.maximum(self.pico[a], self.capital[a])
            self._chequear_estado(a)
        return self.capital[ix] - cap_ini


def simular(estrategia, n, capital0, meta, reglas: Reglas, max_rondas=20000, seed=0):
    """Simula n agentes con la misma estrategia hasta ruina/meta/tope de rondas.

    Los agentes que no apuestan (Kelly con ventaja negativa) o llegan al tope
    quedan ACTIVO; sus rondas jugadas se reportan tal cual.
    """
    rng = np.random.default_rng(seed)
    cap0 = np.broadcast_to(np.asarray(capital0, float), (n,))
    pob = Poblacion(cap0, meta, estrategia, reglas, rng)
    t = 0
    paso = 50
    while t < max_rondas:
        act = np.flatnonzero((pob.estado == ACTIVO) & ~pob.fuera)
        if len(act) == 0:
            break
        pob.jugar(min(paso, max_rondas - t), act)
        t += paso
    return pob


def trayectorias(estrategia, k, capital0, meta, reglas, rondas, seed=0):
    """Capital de k agentes ronda a ronda (NaN tras terminar) para graficar."""
    rng = np.random.default_rng(seed)
    pob = Poblacion(np.full(k, float(capital0)), meta, estrategia, reglas, rng)
    tray = np.full((rondas + 1, k), np.nan)
    tray[0] = pob.capital
    for t in range(1, rondas + 1):
        vivos = pob.estado == ACTIVO
        if not vivos.any():
            break
        pob.jugar(1)
        tray[t, vivos] = pob.capital[vivos]
    return tray, pob


def resumen(pob: Poblacion):
    """Metricas por estrategia: ruina, meta, capital final, duracion."""
    ruina = pob.estado == RUINA
    meta = pob.estado == META
    return {
        "p_ruina": ruina.mean(),
        "p_meta": meta.mean(),
        "p_sin_terminar": max(0.0, 1 - ruina.mean() - meta.mean()),
        "capital_medio": pob.capital.mean(),
        "capital_mediano": np.median(pob.capital),
        "capital_p5": np.percentile(pob.capital, 5),
        "capital_p95": np.percentile(pob.capital, 95),
        "duracion_media": pob.rondas.mean(),
        "duracion_mediana": np.median(pob.rondas),
        "perdida_esperada": (pob.capital0 - pob.capital).mean(),
        "apostado_medio": pob.apostado.mean(),
    }
