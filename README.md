# Video Explicativo

[Enlace a Youtube](https://youtu.be/GAGw2TkokRE)

# La ruina del jugador y la estrategia de apuestas: Cadenas de Markov, ABM y Redes

Proyecto Final de Modelación y Simulación (Universidad del Valle de Guatemala).
Autores: Jose Lopez, Luis Palacios, Hugo Barillas, Roberto Najera.

## De qué trata

Un apostador juega repetidamente contra una casa con ventaja matemática (ruleta europea: p = 18/37, ventaja de la casa 2.70 %; americana: p = 18/38, 5.26 %). El proyecto estudia **qué le pasa a su capital** y si **alguna estrategia de apuesta** (Martingala, Kelly, apuesta fija, porcentaje del capital, entre otras) logra evitar la ruina. Es la base de la gestión de riesgo en casinos y apuestas en línea, de la teoría de la ruina en seguros y de la gestión de capital en trading.

Preguntas que responde:

1. ¿Con qué probabilidad se llega a la ruina antes de alcanzar la meta, según la estrategia?
2. ¿Cuánto tarda en promedio en terminar?
3. ¿Qué tan bien coincide la simulación con la solución analítica?
4. ¿Alguna estrategia vence a la casa, o solo cambia *cómo* se pierde?
5. Si los apostadores copian a sus vecinos en una red, ¿qué estrategias se propagan y por qué?

El proyecto une tres partes del curso:

| Parte | Técnica | Qué se hace |
|---|---|---|
| 1 | Espacio de estados | El capital es una cadena de Markov con estados absorbentes (ruina y meta): forma canónica, matriz fundamental, fórmulas cerradas, variante sin meta (distribución cuasi-estacionaria) y Martingala exacta con límite de mesa. |
| 2 | Agent-Based Modeling | Apostadores individuales con 8 estrategias, simulados con NumPy y validados contra la Parte 1. |
| 3 | Redes | 500 apostadores en redes Erdős–Rényi, Watts–Strogatz y Barabási–Albert que imitan al vecino que más ganó; influencers, informantes y límites de mesa como intervenciones. |

## Hallazgos principales

- **Ninguna estrategia vence a la casa**: la pérdida esperada por unidad apostada es 1−2p en todas (2.7 % europea, 5.3 % americana). Las estrategias solo cambian la varianza, la duración y la forma de la distribución.
- La simulación coincide con la cadena de Markov (error máximo 0.012 en la probabilidad de ruina con 20 000 agentes), también con límite de mesa.
- La Martingala llega más seguido a la meta, pero solo porque apuesta menos en total; el costo son pérdidas catastróficas poco frecuentes.
- Con ventaja positiva (p = 0.52) Kelly maximiza el crecimiento logarítmico; medio Kelly conserva ~75 % del crecimiento con ruina casi nula.
- En red, la Martingala se propaga porque sus éxitos son visibles y sus quiebras no (sesgo de supervivencia): la imitación duplica la ruina. Los informantes en los hubs son la intervención más eficaz en redes libres de escala.

Detalle completo en [`docs/informe.pdf`](docs/informe.pdf) (fuente: [`docs/informe.md`](docs/informe.md)) y [`docs/presentacion.pdf`](docs/presentacion.pdf).

## Estructura

```
src/markov.py          Parte 1: matriz de transición, forma canónica, matriz fundamental, fórmulas,
                       variante sin meta, Martingala exacta con límite de mesa
src/engine.py          Parte 2: ABM vectorizado (8 estrategias)
src/network_model.py   Parte 3: imitación en redes ER/WS/BA, influencers, informantes
experiments/           run_part1.py, run_part2.py, run_part3.py, make_slides.py, common.py
results/               CSV, JSON y logs generados
figures/               gráficas usadas en el informe
docs/                  informe (md y pdf) y presentación (pdf)
```

## Cómo reproducir

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd experiments
python run_part1.py   # segundos
python run_part2.py   # ~2 min en macOS; varios minutos en Windows
python run_part3.py   # ~4 min en macOS; varios minutos en Windows
python make_slides.py
```

Los scripts escriben en `results/` y `figures/`. Las semillas son fijas, así que los resultados son reproducibles.

Para regenerar el informe en PDF (requiere pandoc y XeLaTeX; cambie la fuente por una instalada en su sistema):

```bash
cd docs
pandoc informe.md -o informe.pdf --pdf-engine=xelatex -V lang=es -V mainfont="Arial"
```

## Nota sobre herramientas

No se usó Mesa: el ABM es propio, con arreglos NumPy, porque la Parte 2 necesita decenas de miles de agentes por miles de rondas (ver sección 5 del informe).
