import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from math import ceil, floor

def plot_res(e,p,id,nx,prb,psb):
    #nx es el número de puntos a tomar en el gráfico
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #Definiendo parámetros del gráfico
    dic_par = {'Q': 'Caudal (l/s)', 'V': 'Velocidad (m/s)', 'P': 'Presión (m)'}
    a_total = db.Alpha.unique()
    #Definiendo mapas de colores
    cmap_t = plt.get_cmap('plasma')
    cnorm = colors.Normalize(0, max(a_total))
    cmap_sm = cm.ScalarMappable(norm=cnorm, cmap=cmap_t)
    #Ploteo iterativo
    for a in a_total:
        d_i = db[(db.Elem == e) & (db.Param == p) & (db.ID == id) & (db.Alpha == a) & (db.Niv == 'Inf')].Val
        d_s = db[(db.Elem == e) & (db.Param == p) & (db.ID == id) & (db.Alpha == a) & (db.Niv == 'Sup')].Val
        n=d_i.count()
        pl1=[]
        pl2=[]
        bins=np.linspace(min(d_i.min(),d_s.min()),max(d_i.max(),d_s.max()),nx)
        for x in bins:
            pl1.append(sum([1 for y in d_i if y >= x]) / n)
            pl2.append(sum([1 for y in d_s if y >= x]) / n)
        cval=cmap_sm.to_rgba(a)
        ax.plot(bins,pl1,color=cval,label='Posib. exced.='+str(a))
        ax.plot(bins,pl2,color=cval)
        if psb==1-a:
            pr1=np.percentile(d_i, 100*(prb), axis=0)
            pr2=np.percentile(d_s, 100*(prb), axis=0)
    print("Rango de diseño:",[round(min(pr1, pr2),4), round(max(pr1, pr2),4)])
    ax.plot([min(pr1, pr2), max(pr1, pr2)], [1 - prb, 1 - prb], color='Red', marker='o', markersize=3,label="Rango de diseño:")
    ax.plot([min(pr1, pr2), min(pr1, pr2)], [0, 1 - prb], color='Red', linestyle='--', linewidth=1)
    ax.plot([max(pr1, pr2), max(pr1, pr2)], [0, 1 - prb], color='Red', linestyle='--', linewidth=1)
    ax.plot([], [], ' ', label="Prob.=" + "{:.0%}".format(prb) + "/""Posib.=" + "{:.0%}".format(psb))
    ax.plot([], [], ' ', label="[" + "{:.2f}".format(min(pr1, pr2)) + " - " + "{:.2f}".format(max(pr1, pr2)) + "]")
    ax.set_title(e+"o "+id)
    ax.set_xlabel(dic_par[p])
    ax.set_ylabel('Probab. de excedencia')
    xmin = min(db[(db.Elem == e) & (db.Param == p) & (db.ID == id)].Val)
    xmax = max(db[(db.Elem == e) & (db.Param == p) & (db.ID == id)].Val)
    if xmax-xmin<0.1:
        fe=100
    elif xmax-xmin<1:
        fe=10
    elif xmax-xmin<10:
        fe=1
    else:
        fe=0.5
    xmin = floor(xmin*fe)/fe
    xmax = ceil(xmax*fe)/fe
    ax.set_xticks(np.arange(xmin, xmax+1, 1/fe))
    ax.set_xticks(np.arange(xmin, xmax+1, 0.1/fe), minor=True)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.02), minor=True)
    ax.grid(which='major', alpha=0.75)
    ax.grid(which='minor', alpha=0.25)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.show()
def error(e,p,id,st):
    #st es la cantidad de elementos en cada grupo
    pm=np.zeros(10000)
    a_total = db.Alpha.unique()
    dp=db[(db.Elem == e) & (db.Param == p) & (db.ID == id)]
    k=0
    i=0
    for index, row in dp.iterrows():
        #print("Fila ",index," - Row",row)
        if i<=st-1:
            pm[k] = pm[k] + row.Val /st
        else:
            i=0
            k=k+1
            pm[k] = pm[k] + row.Val / st
        i=i+1
    pm=pm[:k,]
    print("Error en estimación de ",p," en ",id,": ","{:.4%}".format(pm.std()/pm.mean()))
    return pm.std()/pm.mean()/np.sqrt(len(pm))

#n_an=['J-74','J-80','J-112','J-126','J-143','J-145','J-191','J-236','J-253','J-266',
#      'J-286','J-292','J-297','J-299','J-304','J-332','J-336','J-357','J-361','J-371'] #ID de los nodos a mostrar
#t_an=['P-2','P-4','P-6','P-7','P-9','P-10','P-11','P-13','P-15','P-17','P-20','P-22',
#      'P-24','P-25','P-27','P-30','P-44','P-58','P-176','P-182'] #ID de los tubos a mostrar
e='Nod' #Tipo de elemento a analizar (Tub o Nod)
p='P' #Parámetro a calcular ('Q', 'V' o 'P')
id='J-74' #ID del emento
nx = 200 #Cantidad de puntos a graficar
prb=0.95 #Nivel de confianza probabilístico
psb=0.9 #Nivel de confianza posibilístico
st=50 #Cantidad de muestras agrupadas para análisis de error


db=pd.read_csv('output/db_resultados_total.csv',index_col=0)
plot_res(e,p,id,nx,prb,psb)

#for tub in t_an:
#    plot_res('Tub', 'Q', tub, nx, prb, psb)
#    error('Tub','Q',tub,st)
#    plot_res('Tub', 'V', tub, nx, prb, psb)
#    error('Tub','V',tub,st)