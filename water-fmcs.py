# Simulador Difuso de Monte Carlo para WDN
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control
from math import log, pow, sqrt, ceil, floor
import pandas as pd
import matplotlib.colors as colors
import matplotlib.cm as cm
import pyprind


# Clases - Numeros Difusos trapeciales
class Difuso:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        assert a <= b <= c <= d, "Los parámetros tienen que ir de menor a mayor"

    def __str__(self):
        return "({}, {}, {}, {})".format(self.a, self.b, self.c, self.d)

    def __add__(self, other):
        return Difuso(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    def __sub__(self, other):
        return Difuso(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)

    def a_cut(self, alpha):
        assert 0 <= alpha <= 1, "El valor de alpha debe estar en [0,1]"
        return [self.a + alpha * (self.b - self.a), self.d - alpha * (self.d - self.c)]

    def graf(self):
        plt.figure(1)
        plt.plot([self.a, self.b, self.c, self.d], [0, 1, 1, 0])
        plt.fill([self.a, self.b, self.c, self.d], [0, 1, 1, 0], alpha=0.3)
        plt.subplot(111).set_ylim(0, 1)
        plt.xlabel("Valor de x")
        plt.ylabel("Función de membresía")
        plt.show()


# Definiendo funciones
def m_log(m, s):
    return log(m / (pow(s / m, 2) + 1))


def s_log(m, s):
    return sqrt(log(pow(s / m, 2) + 1))


def plot_res(e, p, id, nx, prb, psb):
    # nx es el número de puntos a tomar en el gráfico
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Definiendo parámetros del gráfico
    dic_par = {'Q': 'Caudal (l/s)', 'V': 'Velocidad (m/s)', 'P': 'Presión (m)'}
    a_total = db.Alpha.unique()
    # Definiendo mapas de colores
    cmap_t = plt.get_cmap('plasma')
    cnorm = colors.Normalize(0, max(a_total))
    cmap_sm = cm.ScalarMappable(norm=cnorm, cmap=cmap_t)
    # Ploteo iterativo
    for a in a_total:
        d_i = db[(db.Elem == e) & (db.Param == p) & (db.ID == id) & (db.Alpha == a) & (db.Niv == 'Inf')].Val
        d_s = db[(db.Elem == e) & (db.Param == p) & (db.ID == id) & (db.Alpha == a) & (db.Niv == 'Sup')].Val
        n = d_i.count()
        pl1 = []
        pl2 = []
        bins = np.linspace(min(d_i.min(), d_s.min()), max(d_i.max(), d_s.max()), nx)
        for x in bins:
            pl1.append(sum([1 for y in d_i if y >= x]) / n)
            pl2.append(sum([1 for y in d_s if y >= x]) / n)
        cval = cmap_sm.to_rgba(a)
        ax.plot(bins, pl1, color=cval, label='Posib. exced.={}'.format(a))
        ax.plot(bins, pl2, color=cval)
        if psb == 1 - a:
            pr1 = np.percentile(d_i, 100 * prb, axis=0)
            pr2 = np.percentile(d_s, 100 * prb, axis=0)
    print("Rango de diseño:", [round(min(pr1, pr2), 4), round(max(pr1, pr2), 4)])
    ax.plot([min(pr1, pr2), max(pr1, pr2)], [1 - prb, 1 - prb], color='Red', marker='o', markersize=3, label="Rango de diseño:")
    ax.plot([min(pr1, pr2), min(pr1, pr2)], [0, 1 - prb], color='Red', linestyle='--', linewidth=1)
    ax.plot([max(pr1, pr2), max(pr1, pr2)], [0, 1 - prb], color='Red', linestyle='--', linewidth=1)
    ax.plot([], [], ' ', label="Prob.={:.0%}/Posib.={:.0%}".format(prb, psb))
    ax.plot([], [], ' ', label="[{:.2f} - {:.2f}]".format(min(pr1, pr2), max(pr1, pr2)))

    ax.set_title(e+"o "+id)
    ax.set_xlabel(dic_par[p])
    ax.set_ylabel('Probab. de excedencia')
    xmin = floor(min(db[(db.Elem == e) & (db.Param == p) & (db.ID == id)].Val))
    xmax = ceil(max(db[(db.Elem == e) & (db.Param == p) & (db.ID == id)].Val))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(xmin, xmax + 1, 1))
    ax.set_xticks(np.arange(xmin, xmax + 1, 0.1), minor=True)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.02), minor=True)
    ax.grid(which='major', alpha=0.75)
    ax.grid(which='minor', alpha=0.25)
    ax.legend()
    plt.show()


def error(e, p, id, st):
    # st es la cantidad de elementos en cada grupo
    pm = np.zeros(10000)
    a_total = db.Alpha.unique()
    dp = db[(db.Elem == e) & (db.Param == p) & (db.ID == id)]
    k = 0
    i = 0
    for index, row in dp.iterrows():
        # print("Fila ",index," - Row",row)
        if i <= st-1:
            pm[k] = pm[k] + row.Val / st
        else:
            i = 0
            k = k+1
            pm[k] = pm[k] + row.Val / st
        i = i+1
    pm = pm[:k, ]
    print("Error en estimación de {} en {}: {:.4%}".format(p, id, pm.std() / pm.mean()))
    return pm.std() / pm.mean() / np.sqrt(len(pm))


# Definiendo variables principales del sistema (a ser modificadas por el usuario si se requiere)
ns = 10  # Numero de experimentos aleatorios - Simulacion de monte Carlo
archivo_red = 'input/modelo_red.inp'  # Nombre del archivo de la red a analizar
archivo_temp = 'temp/temp.inp'  # Nombre para el archivo de red temporal
archivo_dem = 'input/demanda_prob.csv'  # Nombre del archivo con los datos de la demanda
archivo_resultados = 'output/db_resultados.csv'  # Nombre del archivo base de datos de los resultados
prb = 0.95  # Nivel de confianza probabilístico
psb = 0.90  # Nivel de confianza posibilística
n_an = ['J-74', 'J-80', 'J-112', 'J-126', 'J-143', 'J-145', 'J-191', 'J-236', 'J-253', 'J-266',
        'J-286', 'J-292', 'J-297', 'J-299', 'J-304', 'J-332', 'J-336', 'J-357', 'J-361', 'J-371']  # ID de los nodos a mostrar
t_an = ['P-2', 'P-4', 'P-6', 'P-7', 'P-9', 'P-10', 'P-11', 'P-13', 'P-15', 'P-17', 'P-20', 'P-22',
        'P-24', 'P-25', 'P-27', 'P-30', 'P-44', 'P-58', 'P-176', 'P-182']  # ID de los tubos a mostrar
cor = Difuso(0, 0.5, 0.5, 1)  # Valores de la correlación difusa
a = [0.1, 0.2, 0.5, 0.8, 0.9]  # Valores de a-cut para el análisis Difuso
st = 50  # Cantidad de muestras agrupadas para análisis de precisión
nx = 100  # Número de puntos a tomar para el gráfico

# Definiendo el nodo a graficar y análizar error:
e = "Tub"
p = "Q"
id = "P-4"

# Definiendo variables de uso interno
nn = 0  # Numero total de nodos
nd = 0  # Numero de nodos con demanda
nl = 0  # Numero de tuberias
nr = 0  # Numero de reservorios
nb = 0  # Numero de bombas
nt = 0  # Numero de tanques
tt = 0  # Numero de tubos

# Definiendo tipos de valores en red Epanet
de = Node.value_type['EN_DEMAND']
pr = Node.value_type['EN_PRESSURE']
pa = Node.value_type['EN_PATTERN']
h = Node.value_type['EN_HEAD']
di = Link.value_type['EN_DIAMETER']
l = Link.value_type['EN_LENGTH']
v = Link.value_type['EN_VELOCITY']
q = Link.value_type['EN_FLOW']

# Creando la db en pandas
ind = ['ID', 'Elem', 'Alpha', 'Niv', 'Param', 'NS', 'Val']
db=pd.DataFrame([],columns=ind) # Activar esta columna para un análisis desde cero
#db = pd.read_csv(archivo_resultados, index_col=0)

# Leyendo archivo red de Epanet (archivo *.inp)
file = os.path.join(os.getcwd(), archivo_red)
es = EPANetSimulation(file)
es.run()

# Definiendo tuberias, reservorios y nodos
nodos_total = es.network.nodes
lineas_total = es.network.links

# Datos varios
for i in np.arange(len(nodos_total)):
    if nodos_total[i+1].node_type == 0:
        nn += 1
    if nodos_total[i+1].node_type == 1:
        nr += 1
    if nodos_total[i+1].node_type == 2:
        nt += 1
    if nodos_total[i+1].results[de][0] > 0:
        nd += 1

for i in np.arange(len(lineas_total)):
    if lineas_total[i+1].link_type == 1:
        tt += 1

print("Nodos en total=", nn)
print("Nodos con demanda=", nd)
print("Total de tubos=", tt)

# Definiendo fila de inicio de las demandas en archivo
red_ini = open(archivo_red, 'r')
fi = 0
for line in red_ini:
    fi += 1
    if "[DEMANDS]" in line:
        break
fi += 1
red_ini.close()

# Leyendo archivo de demanda
temp = np.loadtxt(archivo_dem, delimiter=",", dtype='str')
dn = list(temp[:, 0])              # ID de nodo
dm = list(map(float, temp[:, 1]))  # Media de la demanda nodal
dd = list(map(float, temp[:, 2]))  # Desviacion estandar de la demanda nodal

# Generando parámetros de distribución normal
dm_l = list(map(m_log, dm, dd))
dd_l = list(map(s_log, dm, dd))
dm_m = np.array(dm_l)
dd_m = np.array(dd_l)

mu = dm_l
md = np.identity(nd)
for i in np.arange(nd):
    md[i][i] = dd_l[i]

# Inicio del proceso SDMC <-----------------------------------------------
barra = pyprind.ProgBar(ns * len(a) * 2, stream=sys.stdout)

for niv in a:
    for co in cor.a_cut(niv):
        if co == cor.a_cut(niv)[0]:
            tipo = 'Inf'
        else:
            tipo = 'Sup'
        # -->Simulación de Monte Carlo<--
        
        # Generación de demanda correlacionada
        cr = np.full((nd, nd), co)
        for i in np.arange(nd):
            cr[i][i] = 1
        cv = np.dot(np.dot(md, cr), md)
        rm = np.random.multivariate_normal(mu, cv, size=int(ns / 2))
        rl = rm
        for x in rm:
            d = 2 * dm_m - x
            d = d[np.newaxis]
            rl = np.vstack([rl, d])
        rl = np.exp(rl)

        red_ini = open(archivo_red, 'r')
        for j in np.arange(ns):
            # Leyendo demanda en el archivo
            red_ini.seek(0)
            red_temp = open(archivo_temp, 'w+')
            i = 1
            for line in red_ini:
                if fi <= i < fi + nd:
                    red_temp.write("{} {:.6}\n".format(dn[i - fi], rl[j][i - fi]))
                else:
                    red_temp.write(line)
                i += 1
            red_temp.close()
            # Simulación con el archivo temporal
            file = os.path.join(os.getcwd(), archivo_temp)
            es = EPANetSimulation(file)
            es.run()
            nodos_total = es.network.nodes
            lineas_total = es.network.links
            # Guardando datos de nodos en db
            for nod in n_an:
                se = pd.Series([nod, 'Nod', niv, tipo, 'P', j, nodos_total[nod].results[pr][0]], index=ind)
                db = db.append(se, ignore_index=True)
            # Guardando datos de tubos en db
            for tub in t_an:
                se = pd.Series([tub, 'Tub', niv, tipo, 'Q', j, lineas_total[tub].results[q][0]], index=ind)
                db = db.append(se, ignore_index=True)
                se = pd.Series([tub, 'Tub', niv, tipo, 'V', j, lineas_total[tub].results[v][0]], index=ind)
                db = db.append(se, ignore_index=True)
            # Limpiando
            es.clean()
            es._close()
            barra.update()
        red_ini.close()
barra.stop()

# Exportando CSV
db.to_csv(archivo_resultados)

# Ploteos
plot_res(e, p, id, nx, prb, psb)
error(e, p, id, st)

plt.show()
