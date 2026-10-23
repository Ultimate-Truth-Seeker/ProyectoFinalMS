# La ruina del jugador: Markov, ABM y redes

Proyecto Final de Modelación y Simulación (UVG). Estudia si alguna estrategia de apuesta vence a una casa con ventaja y cómo se propagan las estrategias en una red social de apostadores.

## Estructura
- `src/markov.py` – Parte 1: matriz de transición, forma canónica, matriz fundamental, fórmulas, variante sin meta, Martingala exacta con límite de mesa.
- `src/engine.py` – Parte 2: ABM vectorizado (8 estrategias).
- `src/network_model.py` – Parte 3: imitación en redes ER/WS/BA, influencers, informantes.
- `experiments/run_part{1,2,3}.py` – experimentos; `make_slides.py` – diapositivas.
- `results/` (CSV/JSON/logs), `figures/`, `docs/informe.md|pdf`, `docs/presentacion.pdf`.

## Reproducir
```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scipy matplotlib networkx seaborn
cd experiments
../.venv/bin/python run_part1.py   # segundos
../.venv/bin/python run_part2.py   # ~2 min
../.venv/bin/python run_part3.py   # ~4 min
../.venv/bin/python make_slides.py
```
Informe: `cd docs && pandoc informe.md -o informe.pdf --pdf-engine=xelatex -V lang=es -V mainfont="Helvetica Neue"`.

Semillas fijas; los resultados son reproducibles. No se usó Mesa: el ABM es propio (NumPy) por rendimiento (ver informe, sección 5).
