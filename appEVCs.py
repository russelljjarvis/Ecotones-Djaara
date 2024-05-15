import geopandas as gpd
import matplotlib.pyplot as plt
from geo_adjacency.adjacency import AdjacencyEngine
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
import pandas as pd
import holoviews as hv
import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from pyvis.network import Network

def Compute_EVC_DF_FromData():
    evcs = gpd.read_file('dataframe.geojson', driver='GeoJSON')
    vic_geo = gpd.read_file("suburb-10-vic.geojson")
    source_geoms = gpd.read_file('gda2020_vicgrid/filegdb/whole_of_dataset/victoria/FLORAFAUNA1.gdb', layer=0)
    Bendigodf = source_geoms[source_geoms["BIOREGION"].values == "Goldfields"]
    with open("EVCS_Goldfields_geom.p","wb") as f:
        pickle.dump(Bendigodf,f)
    return None

def Load_EVC_DF():
    with open("EVCS_Goldfields_geom.p","rb") as f:
        Bendigodf = pickle.load(f)
    return Bendigodf

def compute_adjacency(Bendigodf):
    engine = AdjacencyEngine(Bendigodf.geometry)
    adjacency_dict = engine.get_adjacency_dict()
    #print(adjacency_dict)
    #engine.plot_adjacency_dict()
    return adjacency_dict,engine


Bendigodf = Load_EVC_DF()

try:
    with open("adjacency_dict.p","rb") as f:
        [adjacency_dict,EVC_name_dict,named_connectome,links] = pickle.load(f)
except:

    adjacency_dict,engine = compute_adjacency(Bendigodf)
    EVC_name_dict = dict((k,v) for k,v in enumerate(Bendigodf["X_GROUPNAME"].values))
    named_connectome = dict((EVC_name_dict[k],list(set([EVC_name_dict[v_] for v_ in v]))) for (k,v) in adjacency_dict.items() )
    #enumerated_connectome = dict((EVC_name_dict[k],list(set([EVC_name_dict[v_] for v_ in v]))) for (k,v) in adjacency_dict.items() )

    simple_connectome = [ [v_ for v_ in v] for (k,v) in adjacency_dict.items() ]

    links_ = []

    for (k,v) in adjacency_dict.items():
        for target in v:
            temp = v.count(target)

            links_.append({"source":k,"target":target,"value":temp})

    links = pd.DataFrame(links_)


    with open("adjacency_dict.p","wb") as f:
        pickle.dump([adjacency_dict,EVC_name_dict,named_connectome,links],f)


inverted_EVC_name_dict = dict((v,k) for k,v in EVC_name_dict.items())
choice_EVC = st.radio("choose EVC",set(EVC_name_dict.values()))
number_choice = inverted_EVC_name_dict[choice_EVC]

links_ = []
for v in adjacency_dict[number_choice]:
    #if len(v)!=0:
        #print
    #temp = v.count(target)
    links_.append({"source":inverted_EVC_name_dict[number_choice],"target":inverted_EVC_name_dict[v]})#,"weight":temp})

links = pd.DataFrame(links_)

    #print(len(links))
    #from pyvis import Network
G = nx.from_pandas_edgelist(links, 'source', 'target')#, 'weight')


fig, ax = plt.subplots()
pos = nx.kamada_kawai_layout(G)
nx.draw(G,pos, with_labels=True)
st.pyplot(fig)
#nx.draw(G)
#plt.show()

#inverted_EVC_name_dict

#EVC_name_dict
def later():
    links_ = []
    for (k,v) in adjacency_dict.items():
        #if len(v)!=0:
            #print
        for target in set(v):
            temp = v.count(target)
            links_.append({"source":k,"target":target,"weight":temp})

    links = pd.DataFrame(links_)

    print(len(links))
    #from pyvis import Network
    G = nx.from_pandas_edgelist(links, 'source', 'target', 'weight')
    nx.draw(G)
    plt.show()
    # Initiate PyVis network object
    drug_net = Network(
                        height='400px',
                        width='100%',
                        bgcolor='#222222',
                        font_color='white'
                        )

    # Take Networkx graph and translate it to a PyVis graph format
    drug_net.from_nx(G)

    # Generate network with specific layout settings
    drug_net.repulsion(
                        node_distance=420,
                        central_gravity=0.33,
                        spring_length=110,
                        spring_strength=0.10,
                        damping=0.95
                        )

    import os
    # Save and read graph as HTML file (on Streamlit Sharing)
    try:
        path = os.pwd()
        drug_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    except:
        path = '/html_files'
        drug_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=435)

    #columns of links:
    #source, target, value

    #links = pd.DataFrame(simple_connectome)
    fig = hv.Chord(links)
    p = hv.render(fig, backend='bokeh')
    #st.bokeh_chart(p)
    #hv.save(fig ,'fig.html')
    hv.save(fig ,'fig.html')
    HtmlFile = open("fig.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, width=1800, height=1200, scrolling=True)

#clip points using small polys
#points_subset = gpd.clip(points,vic_geo)

#filter polys based on remaining points


#labels = ["BENDIGO","STRATHFIELDSAYE","AXE CREEK","AXE DALE","MANDURANG","SEDGWICK","MANDURANG SOUTH","EMU CREEK","EPPALOCK","SPRING GULLY"]
#gdfBendigo = vic_geo[vic_geo["vic_loca_2"].isin(labels)]
#polys_subset_evc_bendigo = evcs[evcs.geometry.apply(lambda x: gdfBendigo.geometry.within(x).any())]
#-36.7569, 144.2786
def plot_EVC(evcs):
    
    fig, ax = plt.subplots(1, 1)

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    evcs.plot(ax=ax,legend=True,cax=cax)
    
    plt.show()
    #plt.ion()

#plot_EVC(Bendigodf)
#mallard_0.to_file('dataframe.geojson', driver='GeoJSON')
