<div align='center'>
    <h1>Modelado y Simulación de Sistemas Complejos<br>con Aplicaciones en Economía</br></h1>
    <h3>Trabajo práctico final</h3>
</div>

### Organización del trabajo:

- `requirements.txt` tiene todas las bibliotecas necesarias. Se pueden instalar con `pip` de la siguiente forma:
```bash
pip install -r requirements.txt
```

- `SETUP` tiene instrucciones para crear un `venv` e instalar las bibliotecas necesarias. Este procedimiento fue testeado en Windows 10, Windows 11 y Ubuntu 22.04 LTS. Recomendamos Python 3.10+ para ejecutar el código.

- En `src/` está el código fuente.
    - `src/tp.ipynb` es la notebook principal
    - `src/diffseed.ipynb` es la notebook donde probamos con distintas semillas iniciales.
    - `src/run2.py` es un script para poder ejecutar simulaciones para distintos rangos de visión usando multiprocesos en Windows. (Uso: `python3 run2.py --help`)
    - `src/original` tiene los archivos fuente originales del modelo presentado.
    - `src/tp` tiene todos los módulos creados para el trabajo:
        - `src/tp/schelling.py` tiene el modelo del mercado inmobiliario.
        - `src/tp/util/simulador.py` tiene el simulador. (Toma un `mercado_inmobiliario`) y lo simula.
        - `src/tp/presentacion` tiene funciones definidas para graficar y armar la presentación.
        - `src/tp/definiciones.py` tiene algunas definiciones ad-hoc generales al tp.
        - `src/tp/data` tiene dos carpetas: `mapas/` y `resultados/`. En la última están los datos crudos de las simulaciones ejecutadas. En la primera está la representación textual de cada mapa y de los barrios.
        - `src/tp/util/barrios.py` tiene la implementación de `Mapa` y `Barrio`.
        - `src/tp/util/interfaces.py` tiene las clases base de `simulador` y `mercado_inmobiliario`.
        - `src/tp/util/colors.py`, `types.py` y `json/` son conveniencias para poder trabajar.
- `docs/` tiene los PDFs relevantes al trabajo.
- `assets/` tiene gráficos viejos que no se usaron
- `results/` tiene datos viejos.
- `old/` tiene, como su nombre lo indica, borradores viejos del trabajo.