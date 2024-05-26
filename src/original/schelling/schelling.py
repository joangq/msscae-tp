import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Schelling:
    
    def __init__(self, L, T, configuracion=None, vecindario = 8):
        """
        Inicializa el modelo de Schelling.

        Parámetros:
            L (int): Tamaño de la cuadrícula.
            T (int): Umbral de satisfacción en el rango (1, 8].
            configuracion (ndarray): Configuración inicial, si es None, se genera una configuración aleatoria.
        """
        self.L = L
        self.vecindario = vecindario
        assert 1 < T <= self.vecindario, "El umbral debe estar en el rango (1,{}]".format(self.vecindario)
        self.T = T

        if configuracion is not None:
            self.configuracion = configuracion
        else:
            # Generar una configuración aleatoria si no se proporciona ninguna
            self.configuracion = np.random.randint(2, size=(self.L, self.L))

    def satisfaction(self, i, j):
        """
        Calcula la satisfacción de un agente en la posición (i, j).

        Parámetros:
            i (int): Índice de fila.
            j (int): Índice de columna.

        Return:
            int: Valor de satisfacción.
        """
        suma_vecinos = 0

        # Define los índices de los vecinos 
        if self.vecindario == 8:
            indices_vecinos = [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j),
                           (i + 1, j - 1), (i + 1, j + 1), (i - 1, j - 1), (i - 1, j + 1)]
        else:
            indices_vecinos = [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]
            
        
        for (m, n) in indices_vecinos:
            # Tomamos el módulo para manejar las condiciones de borde periódicas
            i_vecino = m % self.L  # Se convierte el -1 y el 30 en 29 y 0, respectivamente
            j_vecino = n % self.L

            if self.configuracion[i_vecino][j_vecino] == 1:
                suma_vecinos += 1  # Cantidad de vecinos pertenecientes a la clase 1

        if self.configuracion[i][j] == 1:  # Nivel de satisfacción del individuo (i, j) dependiendo de su clase y la clase de sus vecinos
            sat = suma_vecinos
        else:
            
            sat = self.vecindario - suma_vecinos

        return sat

    def satisfaccion_total_avg(self):
        """
        Calcula la satisfacción total de todos los agentes en la cuadrícula.

        El par (i, j) especifica una locación en la cuadrícula.

        Return:
            int: Satisfacción total.
        """
        sat_tot = 0

        for i in range(self.L):
            for j in range(self.L):
                sat_ij = self.satisfaction(i, j)
                sat_tot += sat_ij

        sat_tot_avg = sat_tot / self.L**2  # Satisfacción promedio de los individuos

        return sat_tot_avg

    def proponer_intercambio(self):
        """
        Propone un intercambio entre dos agentes aleatorios si ambos están insatisfechos.
        El par (i, j) especifica una locación en la cuadrícula.

        El intercambio ocurre si la satisfacción de ambos agentes está por debajo del umbral.
        """
        # Elige dos individuos al azar
        i1, j1 = np.random.randint(0, self.L), np.random.randint(0, self.L)
        i2, j2 = np.random.randint(0, self.L), np.random.randint(0, self.L)

        if self.configuracion[i1][j1] != self.configuracion[i2][j2]:  # Checkea que no sean el mismo individuo
            sat_1, sat_2 = self.satisfaction(i1, j1), self.satisfaction(i2, j2)

            if sat_1 < self.T and sat_2 < self.T: # evalúa su satisfacción y, si corresponde, realiza el intercambio
                temp = self.configuracion[i1][j1]
                self.configuracion[i1][j1] = self.configuracion[i2][j2]
                self.configuracion[i2][j2] = temp
                
                
    def ronda_intercambio(self):
        
        N_intercambios = self.L**2 
        
        for intercambio in range(N_intercambios):
            self.proponer_intercambio()
    
    
    def lattice_plot(self):
        """
        Crea un gráfico de malla basado en los niveles de satisfacción de los agentes.

        """
        lattice = self.lattice()
        ####################
        
        fig = plt.figure(figsize=(6,4))
        plt.imshow(lattice, cmap = 'inferno')
        plt.colorbar()
        plt.title("Sistema L = {} y T = {} \n (sitios amarillos y naranjas corresponden a agentes insatisfechos)".format(self.L, self.T))
        plt.show()
        

    def lattice(self):
        """
        Return:
            ndarray: Copia de la cuadrícula con agentes insatisfechos marcados de manera diferente.
        """
        lattice = np.copy(self.configuracion)

        for i in range(self.L):
            for j in range(self.L):
                sat_ij = self.satisfaction(i, j)

                if sat_ij < self.T:
                    if self.configuracion[i][j] == 1:
                        lattice[i][j] = 2  # Marcar agente insatisfecho del tipo 1
                    else:
                        lattice[i][j] = 3  # Marcar agente insatisfecho del tipo 0
        
        return lattice
        
    
    def generar_animacion(self, frames):
        fig, ax = plt.subplots()
        img = ax.imshow(self.lattice(), cmap='inferno', interpolation='nearest')

        def actualizar(i):
            self.ronda_intercambio()
            img.set_array(self.lattice())
            return img,

        animacion = FuncAnimation(fig, actualizar, frames=frames, interval=200, blit=True)
        plt.title("Evolución del sistema L = {} y T = {} \n (sitios amarillos y naranjas corresponden a agentes insatisfechos)".format(self.L, self.T))
        plt.close(fig)
        
        return animacion
