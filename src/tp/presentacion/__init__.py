from tp.definiciones import MutableStorage, correr_en_paralelo, correr_secuencialmente
from IPython.display import display, Markdown
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from tp.util.types import Lattice
from tp.util.barrios import Barrio
from tp.definiciones import gini
import ipywidgets as widgets
from typing import Iterable
import pandas as pd
import numpy as np
import warnings
import textwrap
import platform

def mostrar_texto(s: str) -> None:
    """
    Utilidad para renderizar Markdown en un entorno IPython.
    """
    display(Markdown(s))

def graficar_gini(data: Iterable[float], coef: float, color_linea='red', color_relleno='blue', **kwargs):
    """
    Grafica la curva de Lorenz y el coeficiente de Gini asociado a una distribución de datos.
    """

    titulo = kwargs.pop('titulo', 'Curva de Lorenz y Coeficiente de Gini')
    tooltip = kwargs.pop('tooltip', 'Gini Index = {coef:.2f}')
    subtitle = kwargs.pop('subtitulo', 'Curva de Lorenz y Coeficiente de Gini')

    match data.shape:
        case (_,):
            pass
        case _:
            warnings.warn("Aplanando la matriz para calcular el coeficiente de Gini.")
            return graficar_gini(data.flatten(), coef, color_linea, color_relleno)
    
    sorted_data = np.sort(data)
    
    n = len(sorted_data)
    lorenz_curve = np.cumsum(sorted_data) / np.sum(sorted_data)
    lorenz_curve = np.insert(lorenz_curve, 0, 0) # Agrego el (0,0)
    
    # Línea de Igualdad
    equality_line = np.linspace(0, 1, len(lorenz_curve))
    
    # Curva de Lorenz
    plt.style.use('ggplot')
    plt.ion()
    #plt.figure(figsize=(8, 8))
    plt.gca().set_facecolor('white')
    plt.plot(equality_line, lorenz_curve, label='Curva de Lorenz', color=color_relleno)
    plt.plot(equality_line, equality_line, label='Línea de Igualdad', linestyle='--', color=color_linea)

    # Relleno entre la línea de igualdad y la curva de Lorenz
    plt.fill_between(equality_line, equality_line, lorenz_curve, color=color_linea, alpha=0.1)

    # Relleno abajo de la curva de Lorenz
    plt.fill_between(equality_line, 0, lorenz_curve, color=color_relleno, alpha=0.2)
    
    # Muestro el índice
    tooltip_elem = plt.text(0.6, 0.2, tooltip.format(coef=coef), fontsize=12, bbox=dict(facecolor='white', alpha=0.5), color='black')
    tooltip_elem.set_color('black')

    titulo = plt.title(titulo, fontsize=16)
    titulo.set_color('black')

    subtitulo = plt.text(0.5, 1.05, textwrap.fill(subtitle, 70), fontsize=10, ha='center', va='center', transform=plt.gca().transAxes)
    subtitulo.set_color('black')
    
    plt.xlabel('Proporción acumulada de la población de menor a mayor ingreso', fontsize=11)
    plt.ylabel('Proporción acumulada de riqueza', fontsize=14)

    legend = plt.legend(loc='best')
    for t in legend.get_texts():
        t.set_color('black')
    
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Numeros como porcentajes
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y * 100)}%'))
    
    plt.show()

def gini_de(data: Iterable[float], graficar=True, **kwargs):
    """
    Toma un iterable de valores, calcula y grafica el indice de Gini.
    """
    index = gini(data)

    if graficar:
        graficar_gini(data, index, **kwargs)
        
    return index

def graficar_capital(min_val, max_val, capital: Lattice[float], **config_grafica):
    """
    Grafica el capital de la simulación como un heatmap, side-to-side con una gráfica
    de la desigualdad en la distribución: muestra el coeficiente de Gini junto con
    la curva de Lorenz asociada.
    """
    # Extraigo keyword-arugments de la configuración de la gráfica
    cmap = config_grafica.pop('cmap', 'grey')
    suptitle = config_grafica.pop('suptitle', 'Capital inicial\n\n')
    title_text = config_grafica.pop('title', 'Mapa del capital inicial\n\n')
    subtitle_text = config_grafica.get('subtitle', 'Capital inicial por agente en la cuadricula generado por el código aleatoriamente.')

    normalizer = plt.Normalize(min_val, max_val)
    
    plt.style.use('ggplot')
    plt.figure(figsize=(20, 6))
    
    subtitulo = plt.suptitle(suptitle, fontsize=14)
    subtitulo.set_color('black')

    # subplot izq (heatmap)
    plt.subplot(1, 2, 1)
    plt.grid(False)
    title = plt.title(title_text)
    title.set_color('black')

    subtitle_text = textwrap.fill(subtitle_text, 50)
    subtitle = plt.text(0.5, 1.055, subtitle_text, fontsize=11, ha='center', va='center', transform=plt.gca().transAxes)
    subtitle.set_color('black')

    plt.imshow(capital, cmap=cmap, norm=normalizer)
    colorbar = plt.colorbar()
    colorbar.set_label('Capital', rotation=270, labelpad=20)

    # subplot der (gini)
    plt.subplot(1, 2, 2)
    gini = gini_de(capital.flatten(), **config_grafica)

    return normalizer, gini


def generar_capital_inicial(method: callable, *args, config_grafica={}, **kwargs) -> tuple[Lattice[float], plt.Normalize, float]:
    """
    Genera el capital inicial y lo grafica.
    Se espera que 'method' sea un método de
    instancia de la clase 'numpy.Generator'

    'config_grafica' es un diccionario que será utilizado para graficar la distribución.
    El resto de argumentos y keyword-arguments pasan directo a 'method'.
    """
    
    capital_inicial = method(*args, **kwargs)
    flat = capital_inicial.flatten()
    min_val = min(flat)
    max_val = max(flat) * 2 # Multiplico por dos para reusar la escala más adelante

    # Grafico el capital inicial y el coeficiente de Gini de la distribución

    normalizer, gini = graficar_capital(min_val, max_val, capital_inicial, **config_grafica)
    return capital_inicial, normalizer, gini

# ==

def mostrar_satisfechos_antes_despues(satisfechos_inicial, satisfechos_final, **opciones_graficas):
    """
    Grafica dos heatmaps de satisfacción side-to-side.
    A la izquierda 'antes' de la simulación; a la derecha 'después'.
    """
    pasos_totales = opciones_graficas['pasos_totales']
    scale = opciones_graficas.pop('scale', 1)
    suptitle_text = opciones_graficas.pop('suptitle', 'Satisfacción de los agentes\n\n')
    subtitulo_text = opciones_graficas.pop('subtitulo', 'Mapa de satisfacción de los agentes antes y después del equilibrio.')


    fig, axs = plt.subplots(1, 2, figsize=(2*scale, 1*scale))
    suptitle = fig.suptitle(suptitle_text, fontsize=12)
    suptitle.set_color('black')

    subtitle = fig.text(0.5, 0.86, textwrap.fill(subtitulo_text, 70), fontsize=10, ha='center', va='center')
    subtitle.set_color('black')

    plt.tight_layout(pad=.2)

    discrete_cmap = plt.cm.get_cmap('inferno', 5)
    norm = plt.Normalize(0, 5)

    plt.subplot(1,2,1)
    plt.grid(False)
    title = plt.title('Satisfechos iniciales ($t=0$)')
    title.set_color('black')
    plt.imshow(satisfechos_inicial, cmap=discrete_cmap, norm=norm)
    plt.colorbar(label='Nivel de insatisfacción')

    plt.subplot(1,2,2)
    plt.grid(False)
    title = plt.title(f'Satisfechos finales ($t={pasos_totales}$)')
    title.set_color('black')
    plt.imshow(satisfechos_final, cmap=discrete_cmap, norm=norm)
    plt.colorbar(label='Nivel de insatisfacción')


def subsample(ys, amount=1000):
    """
    'Subdivide' una lista de valores 'ys' en 'amount' partes iguales,
    con 'amount' < len(ys).

    Es una utilidad para probar distintas cantidades de datos en un gráfico.
    """
    indices = np.linspace(0, len(ys)-1, amount, dtype=int)
    subdivided_y = np.array(ys)[indices]
    subdivided_x = np.linspace(0, 1, len(subdivided_y))
    return subdivided_x, subdivided_y

def smooth_data(ys, window_length=None, **kwargs):
    """
    Toma una lista de valores 'ys' y devuelve una lista de valores suavizados,
    garantiza que ambas listas tengan la misma longitud.

    En particular, filtra el ruido usando un filtro de Savitzky-Golay.
    Si no se pasa un largo de ventana, se calcula automáticamente como:

    window_length = |ys| / (1+log_10(|ys|))
    """

    if window_length is None:
        window_length = len(ys) // len(str(len(ys)))
    
    polyorder = kwargs.get('polyorder', 3)
    return savgol_filter(ys, window_length=window_length, polyorder=polyorder)

def plot_satisfacciones_para_alpha(alpha: float, 
                                   satisfacciones: dict[float, dict[float, list[int]]], 
                                   habitantes_por_barrio: list[int], 
                                   barrios_definidos: list[Barrio],
                                   _subsample: None|int =None,
                                   _smoothing_cutoff=15,
                                   ax=None):
    """
    Grafica las satisfacciones en función de el rango de visión para
    un alpha particular.
    Espera un diccionario que tenga el siguiente esquema:
    {float: {float: list[int], ... }, ... }

    Donde el primer float es el alpha, el segundo float es el rango de visión.

    'habitantes_por_barrio' es una lista con la cantidad de habitantes por barrio.
    'barrios_definidos' es una lista de instancias de 'Barrio' que definen los colores.

    '_subsample' es un parámetro de prueba, para ver el gráfico con distintas cantidades de datos.
    '_smoothing_cutoff' es un umbral que determina si debe suavizarse el gráfico o no.
    """

    if not ax:
        fig, ax = plt.figure(figsize=(6, 4))
    
    if not alpha in satisfacciones.keys():
        raise KeyError('Alfa inválido.')
    
    df = pd.DataFrame(satisfacciones[alpha]).T
    n_barrios = len(habitantes_por_barrio)
    for i in range(n_barrios):
        color = barrios_definidos[i].color
        satisfechos_i = 100 * df[i] / habitantes_por_barrio[i]
        x = np.linspace(0, 1, len(satisfechos_i))
        if _subsample is not None:
            x, satisfechos_i = subsample(satisfechos_i, _subsample)
        
        noisy_alpha = 1
        if len(satisfechos_i) >= _smoothing_cutoff:
            smooth_subdivided_savgol = smooth_data(satisfechos_i)
            ax.plot(x, smooth_subdivided_savgol, label=f'Barrio {i} (suavizada)', color=color, alpha=1)
            noisy_alpha = .3

        ax.plot(x, satisfechos_i, label=f'Barrio {i} (original)', color=color, alpha=noisy_alpha)

    legend = ax.legend(loc='best', ncol=n_barrios)
    for t in legend.get_texts():
        t.set_color('black')
        t.set_fontsize(7)

    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1)
    ax.figure.tight_layout()
    return ax.figure

def detect_os():
    """
    Detecta el sistema operativo actual y devuelve una 'string' que lo representa.
    """
    current_os = platform.system()
    if current_os == "Windows":
        return 'windows'
    elif current_os == "Linux":
        return 'linux'
    else:
        return 'unknown'

def create_os_buttons(detected_os, on_click: callable):
    """
    Crea botones para ejecutar la simulación.
    'Correr en paralelo' solo debería estar habilitado en Linux,
    en Windows se rompen las pools de procesos.
    """
    layout = widgets.Layout(width='auto', height='40px')
    windows_button = widgets.Button(description="Correr con un solo hilo", layout=layout)
    linux_button = widgets.Button(description="Correr en paralelo", layout=layout)

    if detected_os == "windows":
        windows_button.disabled = False
        linux_button.disabled = True
    elif detected_os == "linux":
        windows_button.disabled = False
        linux_button.disabled = False
    else:
        windows_button.disabled = True
        linux_button.disabled = True

    def disable_buttons():
        windows_button.disabled = True
        linux_button.disabled = True
    
    def _on_click(b):
        disable_buttons()
        on_click(b)

    windows_button.on_click(_on_click)
    linux_button.on_click(_on_click)

    buttons = widgets.HBox([windows_button, linux_button], width='auto')
    display(buttons)


def opciones_ejecutar_modelo(inputs, store: MutableStorage):
    """
    Fabrica de funciones, toma parámetros para ser usados por los callbacks
    de los botones.

    Devuelve una función que espera un botón, ejecuta la función correspondiente y muta el Storage.
    """
    def _(b):
        result = correr_en_paralelo(inputs) if b.description == 'Correr en paralelo' else correr_secuencialmente(inputs)
        store.set_store(result)
        return store
    return _


