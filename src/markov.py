"""Parte 1: cadena de Markov de la ruina del jugador.

Estados 0..N. Los estados 0 (ruina) y N (meta) son absorbentes; 1..N-1 son
transitorios. En cada ronda se apuesta 1 unidad: i -> i+1 con prob. p,
i -> i-1 con prob. q = 1-p.
"""
import numpy as np

P_EUROPEA = 18 / 37    # rojo/negro, ruleta europea (un cero)
P_AMERICANA = 18 / 38  # rojo/negro, ruleta americana (cero y doble cero)


def matriz_transicion(N, p):
    """Matriz P de (N+1)x(N+1) en el orden natural de estados 0..N."""
    P = np.zeros((N + 1, N + 1))
    P[0, 0] = P[N, N] = 1.0
    for i in range(1, N):
        P[i, i + 1] = p
        P[i, i - 1] = 1 - p
    return P


def forma_canonica(P):
    """Reordena estados (transitorios primero, absorbentes al final).

    Devuelve (Pc, Q, R, orden) con Pc = [[Q, R], [0, I]].
    """
    n = P.shape[0]
    absorb = [i for i in range(n) if P[i, i] == 1.0]
    trans = [i for i in range(n) if i not in absorb]
    orden = trans + absorb
    Pc = P[np.ix_(orden, orden)]
    t = len(trans)
    return Pc, Pc[:t, :t], Pc[:t, t:], orden


def analisis_absorcion(N, p):
    """Matriz fundamental F=(I-Q)^-1, tiempos esperados y prob. de absorcion.

    Devuelve dict con F, t (tiempo esperado hasta absorcion), B=F*R con
    columnas [ruina (0), meta (N)] para cada estado transitorio 1..N-1.
    """
    P = matriz_transicion(N, p)
    _, Q, R, orden = forma_canonica(P)
    F = np.linalg.inv(np.eye(Q.shape[0]) - Q)
    t = F.sum(axis=1)
    B = F @ R  # columnas en el orden de los absorbentes: [0, N]
    return {"P": P, "Q": Q, "R": R, "F": F, "t": t, "B": B}


def ruina_analitica(i, N, p):
    """P(ruina | capital i) con meta N (formula clasica)."""
    i = np.asarray(i, dtype=float)
    if abs(p - 0.5) < 1e-12:
        return 1 - i / N
    r = (1 - p) / p
    return (r ** i - r ** N) / (1 - r ** N)


def duracion_analitica(i, N, p):
    """Tiempo esperado hasta terminar partiendo de i."""
    i = np.asarray(i, dtype=float)
    if abs(p - 0.5) < 1e-12:
        return i * (N - i)
    q = 1 - p
    r = q / p
    return i / (q - p) - (N / (q - p)) * (1 - r ** i) / (1 - r ** N)


def ruina_sin_meta(i, p):
    """P(ruina alguna vez) sin meta: 1 si p<=1/2, (q/p)^i si p>1/2."""
    i = np.asarray(i, dtype=float)
    if p <= 0.5:
        return np.ones_like(i)
    return ((1 - p) / p) ** i


# ---- variante sin meta: solo ruina (truncada en capital maximo C) ----------

def cadena_sin_meta(C, p, reflejante=True):
    """Estados 0..C; 0 absorbente. En C: rebota (la casa no paga mas si C
    representa el capital total en juego) -> reflejante."""
    P = np.zeros((C + 1, C + 1))
    P[0, 0] = 1.0
    for i in range(1, C):
        P[i, i + 1] = p
        P[i, i - 1] = 1 - p
    if reflejante:
        P[C, C - 1] = 1.0
    else:
        P[C, C] = 1.0
    return P


def evolucion_distribucion(P, i0, pasos):
    """Distribucion pi_n = pi_0 P^n para n=0..pasos (matriz (pasos+1, S))."""
    pi = np.zeros(P.shape[0])
    pi[i0] = 1.0
    out = [pi.copy()]
    for _ in range(pasos):
        pi = pi @ P
        out.append(pi.copy())
    return np.array(out)


def distribucion_cuasi_estacionaria(P):
    """Distribucion de Yaglom: autovector izquierdo dominante de Q y la tasa
    de decaimiento lambda (prob. de supervivencia ~ lambda^n)."""
    Q = P[1:, 1:]
    w, V = np.linalg.eig(Q.T)
    k = np.argmax(w.real)
    v = np.abs(V[:, k].real)
    return v / v.sum(), w[k].real


# ---- extension: martingala con limite de mesa -------------------------------

def martingala_limite_mesa(N, p, limite, base=1):
    """Cadena exacta de la Martingala con limite de mesa.

    Estado = (capital c, racha perdedora k). Apuesta b = min(base*2^k, limite, c).
    Tras ganar -> (c+b, 0); tras perder -> (c-b, k+1) (k saturado cuando
    base*2^k >= limite). Absorbentes: c<=0 (ruina), c>=N (meta).

    Devuelve funcion f(c0) -> (P ruina, duracion esperada) resolviendo el
    sistema lineal exacto.
    """
    kmax = 0
    while base * 2 ** kmax < limite:
        kmax += 1
    K = kmax + 1  # rachas 0..kmax
    idx = lambda c, k: (c - 1) * K + k  # c en 1..N-1
    n = (N - 1) * K
    A = np.eye(n)
    r_ruina = np.zeros(n)
    for c in range(1, N):
        for k in range(K):
            b = int(min(base * 2 ** k, limite, c))
            s = idx(c, k)
            # gana
            cw = c + b
            if cw < N:
                A[s, idx(cw, 0)] -= p
            # pierde
            cl = c - b
            kn = min(k + 1, kmax)
            if cl <= 0:
                r_ruina[s] += 1 - p
            else:
                A[s, idx(cl, kn)] -= 1 - p
    Pr = np.linalg.solve(A, r_ruina)
    T = np.linalg.solve(A, np.ones(n))
    return lambda c0: (Pr[idx(c0, 0)], T[idx(c0, 0)]), Pr.reshape(N - 1, K)[:, 0], T.reshape(N - 1, K)[:, 0]
