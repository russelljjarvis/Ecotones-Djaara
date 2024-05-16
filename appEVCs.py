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
#import seaborn as sns
import os
import folium
#import altair as alt


#from streamlit_folium import st_folium
#from folium.plugins import Draw
st.set_option('deprecation.showPyplotGlobalUse', False)
#import plotly.graph_objects as go
#import urllib, json

#import pickle
st.set_page_config(layout="wide")


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

st.header("Ecoterms and EVCS of North Central Catchment Region/Gold Fields")


def main():
    Bendigodf = Load_EVC_DF()

    try:
        with open("adjacency_dict.p","rb") as f:
            [adjacency_dict,EVC_name_dict,named_connectome,links] = pickle.load(f)
    except:

        adjacency_dict,engine = compute_adjacency(Bendigodf)
        EVC_name_dict = dict((k,v) for k,v in enumerate(Bendigodf["X_GROUPNAME"].values))
        named_connectome = dict((EVC_name_dict[k],list(set([EVC_name_dict[v_] for v_ in v]))) for (k,v) in adjacency_dict.items() )
        #enumerated_connectome = dict((EVC_name_dict[k],list(set([EVC_name_dict[v_] for v_ in v]))) for (k,v) in adjacency_dict.items() )
        #simple_connectome = [ [v_ for v_ in v] for (k,v) in adjacency_dict.items() ]
        links_ = []
        for (k,v) in adjacency_dict.items():
            for target in v:
                temp = v.count(target)
                links_.append({"source":k,"target":target,"value":temp})
        links = pd.DataFrame(links_)
        with open("adjacency_dict.p","wb") as f:
            pickle.dump([adjacency_dict,EVC_name_dict,named_connectome,links],f)



    EVC_name_dict = dict((k,v) for k,v in enumerate(Bendigodf["X_GROUPNAME"].values))
    #inverted_EVC_name_dict = dict((v,k) for k,v in EVC_name_dict.items())
    choice_EVC = st.radio("choose EVC",set(EVC_name_dict.values()),index=2)
    choice_df_index = Bendigodf[Bendigodf["X_GROUPNAME"]==choice_EVC].index
    ecoterns,adjacencies = get_adjacency_net(choice_df_index,adjacency_dict,EVC_name_dict)
    
    #temp = adjacency_dict[number_choice]
    #st.write(adjacency_dict)
    slow_do_last(Bendigodf,ecoterns,adjacencies,choice_EVC)


def slow_do_last(Bendigodf,ecoterns,adjacencies,choice_EVC):
    ecoterns = list(ecoterns)
    col1, col2 = st.columns(2)

    col1.subheader("All the EVCs")
    col2.subheader("Selected EVC")
    with col1:
        Bendigodf.plot(column="X_GROUPNAME",figsize=(5, 5), legend=True)#, legend_kwds={"orientation": "horizontal"},)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='lower center')
        st.pyplot()
    with col2:
        filtered_Bendigodf = Bendigodf[Bendigodf["X_GROUPNAME"]==choice_EVC]
        filtered_Bendigodf.plot(column="X_GROUPNAME",figsize=(5, 5), legend=True)
        st.pyplot()

    col1, col2 = st.columns(2)
    col1.subheader("Ecotern")
    col2.subheader("Selected EVC")

    #st.write(adjacencies)
    #st.write(Bendigodf.index.isin(adjacencies))
    filtered_eco_tern = Bendigodf[Bendigodf.index.isin(adjacencies)]
    SingleEVC = Bendigodf[Bendigodf["X_GROUPNAME"]==choice_EVC]

    with col1:

        #st.write(ecoterns)
        #st.write(filtered_Bendigodf)
        #vic_geo[vic_geo["vic_loca_2"].isin(labels)]
        #filtered_Bendigodf = pd.merge(filtered_Bendigodf, gdfBendigo, on='geometry', how='outer', suffixes=('_df1', '_df2')).fillna(0)
        filtered_eco_tern.plot(column="X_GROUPNAME",figsize=(5, 5), legend=True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='lower center')

        st.pyplot()

    with col2:
        #filtered_Bendigodf = pd.merge(filtered_Bendigodf, gdfBendigo, on='geometry', how='outer', suffixes=('_df1', '_df2')).fillna(0)
        SingleEVC.plot(column="X_GROUPNAME",figsize=(5, 5), legend=True)
        st.pyplot()

    bothdf = pd.concat([SingleEVC, filtered_eco_tern])#, on='geometry', how='outer', suffixes=('_df1', '_df2')).fillna(0)

    bothdf.plot(column="X_GROUPNAME",figsize=(5, 5), legend=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='lower center')

    st.pyplot()
    

def get_adjacency_net(choice_df_index,adjacency_dict,EVC_name_dict):
    links_ = []
    nodes_ = []
    weight_dict_ = dict()
    cnt = 0
    #st.write(choice_df_index)

    for number_choice in choice_df_index:
        for v in adjacency_dict[number_choice]:
            keyd = str(set([EVC_name_dict[number_choice],EVC_name_dict[v]]))
            if keyd not in weight_dict_.keys():
                weight_dict_[keyd] = 1
            else:
                weight_dict_[keyd]+=1


    category_weight_dict_= dict()
    for (ind,k) in enumerate(weight_dict_.keys()):
        category_weight_dict_[k] = ind

    ecoterns = []

    for number_choice in choice_df_index:
        for v in adjacency_dict[number_choice]:
            keyd = str(set([EVC_name_dict[number_choice],EVC_name_dict[v]]))
            links_.append({"source":EVC_name_dict[number_choice],"target":EVC_name_dict[v],"value":weight_dict_[keyd]})
            nodes_.append({"index":cnt, "group":category_weight_dict_[keyd], "name":EVC_name_dict[number_choice]})
            ecoterns.append(EVC_name_dict[number_choice])
            cnt+=1

    links = pd.DataFrame(links_)
    nodes = pd.DataFrame(nodes_)

    #for number_choice in choice_df_index:
    
    ecoterns = set(ecoterns)
    G = nx.from_pandas_edgelist(links, 'source', 'target', 'value')
    EVC_net = Network(
                        height='400px',
                        width='100%',
                        bgcolor='#222222',
                        font_color='white'
                        )

    # Take Networkx graph and translate it to a PyVis graph format
    EVC_net.from_nx(G)

    # Generate network with specific layout settings
    EVC_net.repulsion(
                        node_distance=420,
                        central_gravity=0.33,
                        spring_length=110,
                        spring_strength=0.10,
                        damping=0.95
                        )

    # Save and read graph as HTML file (on Streamlit Sharing)
    try:
        path = os.getcwd()
        EVC_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    except:
        path = os.getcwd()
        EVC_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=435)
    adjacent_indexs = [v for v in adjacency_dict[number_choice] ]
    print(adjacency_dict.keys())
    return ecoterns,adjacent_indexs

#url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
#response = urllib.request.urlopen(url)
#data = json.loads(response.read())



if __name__ == "__main__":
    main()



def hist():

    fig, ax = plt.subplots()
    for k,v in weight_dict_.items():
        if v!=0:
            ax.hist(v,label=k, bins=10)

    fig.legend()
    st.pyplot(fig)

def goldfieldslocations():
    vic_geo = gpd.read_file("suburb-10-vic.geojson")

    vic_geo = vic_geo[['vic_loca_2','geometry']]
    Bendigo = vic_geo[vic_geo["vic_loca_2"] == "BENDIGO"]
    Strath = vic_geo[vic_geo["vic_loca_2"] == "STRATHFIELDSAYE"]
    Axe_C = vic_geo[vic_geo["vic_loca_2"] == "AXE CREEK"]
    Axe_D = vic_geo[vic_geo["vic_loca_2"] == "AXE DALE"]
    Mandu = vic_geo[vic_geo["vic_loca_2"] == "MANDURANG"]
    Sedgwick = vic_geo[vic_geo["vic_loca_2"] == "SEDGWICK"]
    labels = ["BENDIGO","STRATHFIELDSAYE","AXE CREEK","AXE DALE","MANDURANG","SEDGWICK","MANDURANG SOUTH","EMU CREEK","EPPALOCK","SPRING GULLY"]
    labels_enum = [i for (i,J) in enumerate(labels)]    
    gdfBendigo = vic_geo[vic_geo["vic_loca_2"].isin(labels)]
    gdfBendigo.plot(column="vic_loca_2",figsize=(6, 6), legend=True)
    st.pyplot()

    return gdfBendigo

def folium():
    ##
    # CRS is necessary for folium
    
    filtered_Bendigodf = filtered_Bendigodf.to_crs(epsg=4326)
    #print(filtered_Bendigodf.crs)
    #st.write(filtered_Bendigodf.crs)

    filtered_Bendigodf.explore(
    column="X_GROUPNAME",  # make choropleth based on "BoroName" column
    tooltip="X_GROUPNAME",  # show "BoroName" value in tooltip (on hover)
    popup=True,  # show all values in popup (on click)
    tiles="CartoDB positron",  # use "CartoDB positron" tiles
    cmap="Set1",  # use "Set1" matplotlib colormap
    style_kwds=dict(color="black"),  # use black outline
    )

    st_data = st_folium(vic_inc_map, width=1725)




def dontdo():
    #st.write(nodes)
    #chord = hv.Chord((links, nodes)).select(value=(5, None))
    fig = hv.Chord((links,nodes)).select(value=(5, None))
    fig.opts(opts.Chord(labels='name', 
                            cmap='Category20', 
                            edge_cmap='Category20'))

    #p = hv.render(fig, backend='bokeh')
    #st.bokeh_chart(p)
    #hv.save(fig ,'fig.html')
    #st.bokeh_chart(hv.render(fig, backend='bokeh'))

    hv.save(fig ,'fig.html')
    HtmlFile = open("fig.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, width=1800, height=1200, scrolling=True)


def sankey():
    fig = go.Figure(data=[go.Sankey(
        valueformat = ".0f",
        valuesuffix = "TWh",
        # Define nodes
        node = dict(
        pad = 15,
        thickness = 15,
        line = dict(color = "black", width = 0.5),
        label =  nodes['name'].values,
        ),
        # Add links
        link = dict(
        source =  links['source'].values,
        target =  links['target'].values,
        value =  links['value'].values
    ))])

    fig.update_layout(title_text="sankey diagram",
                    font_size=10)
    #fig.show()
    st.plotly_chart(fig)#, use_container_width=False, sharing="streamlit", theme="streamlit")# **kwargs)

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
            links_.append({"source":k,"target":target,"value":temp})

    links = pd.DataFrame(links_)

    print(len(links))
    #from pyvis import Network
    G = nx.from_pandas_edgelist(links, 'source', 'target', 'value')
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
    st.bokeh_chart(p)
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
def cruft():
        
    #filtered_Bendigodf["centroid"] = filtered_Bendigodf.centroid
    #filtered_Bendigodf['lat'] = filtered_Bendigodf.centroid.y
    #filtered_Bendigodf['lon'] = filtered_Bendigodf.centroid.x

    #import streamlit as st
    #import plotly.express as px
    
    #fig = px.choropleth(filtered_Bendigodf, geojson=filtered_Bendigodf.geometry, locations=filtered_Bendigodf.index, color_continuous_scale="Viridis", projection="mercator")
    #fig.update_geos(fitbounds="locations", visible=False)

    #st.plotly_chart(fig)

    #chart = alt.Chart(filtered_Bendigodf).mark_geoshape()
    #st.altair_chart(chart)
    """
    #vic_inc_map = folium.Map([-36.7569, 144.2786], zoom_start=5, tiles='CartoDB positron')
    #temp_json = Bendigodf[Bendigodf["X_GROUPNAME"]==choice_EVC].to_json()

    st.write(filtered_Bendigodf)
    #plt.show()
    for _, r in filtered_Bendigodf.iterrows():
        # Without simplifying the representation of each borough,
        # the map might not be displayed
        sim_geo = gpd.GeoSeries(r["geometry"]).simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        print(geo_j)
        geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "orange"})
        folium.Popup(r["X_GROUPNAME"]).add_to(geo_j)
        geo_j.add_to(vic_inc_map)

        #for _, r in df.iterrows():
        folium.Marker(location=[r['lat'], r['lon']]).add_to(vic_inc_map)
    folium.LayerControl().add_to(vic_inc_map)

    Draw(export=True).add_to(vic_inc_map)


    #vic_inc_map.show_in_browser()
    st_data = st_folium(vic_inc_map, width=1725)

    """

    #fig = plt.figure()
    #filtered_Bendigodf.plot()
    #st.pyplot(fig)
    #st.map(filtered_Bendigodf)
    #st.write(filtered_Bendigodf)
    #folium.Choropleth(geo_data=temp_json,
    #            name='Choropleth',         
    #            data=filtered_Bendigodf,
    #            columns=['EVC','geometry'],
    #            fill_color='YlOrRd',
    #            fill_opacity=0.4, 
    #            line_opacity=0.4,
    #            smooth_factor=0,     
    #                ).add_to(vic_inc_map) 
    #folium.LayerControl().add_to(vic_inc_map)
    
    # Adding labels to map
    # style_function = lambda x: {'fillColor': '#ffffff', 
    #                             'color':'#000000', 
    #                             'fillOpacity': 0.0, 
    #                             'weight': 0.1}
    # BioRegionName = folium.features.GeoJson(
    #     Bendigodf,
    #     style_function=style_function, 
    #     control=False,
    #     tooltip=folium.features.GeoJsonTooltip(
    #         fields=['X_GROUPNAME'],
    #         aliases=['EVC'],
    #         style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    #     )
    # )
    # vic_inc_map.add_child(BioRegionName)
    # vic_inc_map.keep_in_front(BioRegionName)

