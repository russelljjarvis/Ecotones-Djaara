import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
import pandas as pd
import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from pyvis.network import Network
import pyvis
import os
from matplotlib import pyplot as plt
#from scipy.spatial import voronoi_plot_2d
from geo_adjacency.adjacency import AdjacencyEngine
import geocoder
from shapely.geometry import Point       


# https://stackoverflow.com/questions/29797435/get-precise-android-gps-location-in-python

#import osmnx as ox
#from pyproj import Transformer
import folium
#from folium.plugins import Draw
import geopandas
import copy
#import seaborn as sns
from streamlit_folium import st_folium
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
import plotly.express as px


def goldfieldslocations():
    vic_geo = gpd.read_file("suburb-10-vic.geojson")

    vic_geo = vic_geo[['vic_loca_2','geometry']]
    labels = ["BENDIGO","STRATHFIELDSAYE","AXE CREEK","AXE DALE","MANDURANG","SEDGWICK","MANDURANG SOUTH","EMU CREEK","EPPALOCK","SPRING GULLY"]
    gdfBendigo = vic_geo[vic_geo["vic_loca_2"].isin(labels)]
    #gdfBendigo.plot(column="vic_loca_2",figsize=(6, 6), legend=True)
    #gdfBendigo.set_ylabel(legendAsLatex(gdfBendigo)) 
    #st.pyplot()

    return gdfBendigo


def Compute_EVC_DF_FromData():
    evcs = gpd.read_file('dataframe.geojson', driver='GeoJSON')
    vic_geo = gpd.read_file("suburb-10-vic.geojson")
    source_geoms = gpd.read_file('gda2020_vicgrid/filegdb/whole_of_dataset/victoria/FLORAFAUNA1.gdb', layer=0)
    Bendigodf = source_geoms[source_geoms["BIOREGION"].values == "Goldfields"]
    with open("EVCS_Goldfields_geom.p","wb") as f:
        pickle.dump(Bendigodf,f)
    return None

#@st.cache_data
def Load_EVC_DF():
    with open("EVCS_Goldfields_geom.p","rb") as f:
        Bendigodf = pickle.load(f)
    return Bendigodf

def compute_adjacency(Bendigodf):

    engine = AdjacencyEngine(Bendigodf.geometry)
    adjacency_dict = engine.get_adjacency_dict()
    fig, ax = plt.subplots(1, figsize=(5,5))

    engine.plot_adjacency_dict()
    Bendigodf.plot(ax=ax)

    st.pyplot()

    return adjacency_dict,engine

st.title("Nature Stewards: Ecological Vegetation Classes (EVCs) of the Gold Fields Forest")

def source_data():
    try:
        with open("adjacency_dict.p","rb") as f:
            [adjacency_dict,EVC_name_dict,named_connectome,links] = pickle.load(f)
    except:

        adjacency_dict,_ = compute_adjacency(Bendigodf)
        EVC_name_dict = dict((k,v) for k,v in enumerate(Bendigodf[used_scheme].values))
        named_connectome = dict((EVC_name_dict[k],list(set([EVC_name_dict[v_] for v_ in v]))) for (k,v) in adjacency_dict.items() )
        links_ = []
        for (k,v) in adjacency_dict.items():
            for target in v:
                temp = v.count(target)
                links_.append({"source":k,"target":target,"value":temp})
        links = pd.DataFrame(links_)
        with open("adjacency_dict.p","wb") as f:
            pickle.dump([adjacency_dict,EVC_name_dict,named_connectome,links],f)

    return adjacency_dict,EVC_name_dict,named_connectome,links

def renewed(source,Bendigodf):
    engine = AdjacencyEngine(source.geometry)
    adjacency_dict = engine.get_adjacency_dict()

    for source_i, target_i_list in output.items():
        source_geom = source_geometries[source_i]
        target_geoms = [target_geometries[i] for i in target_i_list]


def main():
    Bendigodf = Load_EVC_DF()
    Bendigodf_ = copy.copy(Bendigodf)
    Bendigodf.reset_index(inplace=True)
    Bendigodf.drop(columns=["EVC_MUT","EVC_CODE","SCALE","EVC_BCS","EVC_GP","EVC_GO","EVC_GO_DESC","BIOEVC","EVC_BCS_DESC","EVC_BCS_SRC","EVC_SUBGP","BIOREGION","BIOREGION_CODE","VEG_CODE","BIOREGION_NO"],inplace=True)
    Bendigodf = Bendigodf.to_crs(epsg=4326)

    adjacency_dict,EVC_name_dict,named_connectome,links = source_data()
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander("What is a Plant Community?"):
            st.write('''
                This is not a formal definition, but a plant community is a group of the same or similar groups of plants that occur in close proximity to each other. 
                When the same group of plants co-occur, they forming an observable re-occuring pattern to observers.
                Plant communities, have membership and structural relationships.
            ''')
            st.image("commass.jpg")


        with st.expander("What is an Ecological Vegetation Class (EVC)?"):
            st.write('''
                This is not a formal definition, but an Ecological Vegetation Class (EVC) is a designated area of Victorian forest, which agrees with a particular class of known plant community structure.
            ''')
            st.image("for_website_EVC.png")

        with st.expander("What is a Ecotone?"):
            st.write('''
                An ecotone is an area of Victorian forest, that is poorly described by an EVC. 
                The area in question may be situated between two or more EVCs, and because the neighbouring EVCs are blending into each other the area that is between the two EVCs has characteristics common to both 
                of the peripheral EVCs at once. The result of blending two or more switch ecotones togethor is that they may be a poor fit for the pre-existing EVCs.
            ''')
            st.header("Possible Ecotone")
            st.image("4212399074159334139.jpg")

            st.image("ecotonesSwitchPicture.png")

        with st.expander("What is the Motivation for this DashBoard?"):
            st.write('''
            The motivation for this dashboard is to iterate over all combinations of neighbouring EVCs, and for each EVC to ask what are the EVCs possible neighbours?
                    For each possible pair of EVC there can be an Ecotones, and the locations of these Ecotones can be plotted on a map. 
                    To make this dashboard, I downloaded the data file gda2020_vicgrid EVCs from 2020 for all of Victoria, from the Victorians government open data portal. It contains the file "FLORAFAUNA1.gdb" which is readable by Geopandas in Python.
            ''')

        with st.expander("About Me:"):
            st.markdown('''
            Russell Jarvis (he/him) PhD
            I respectfully acknowledge the Traditional Owners of the land on which we work and learn, and pay respect to the First Nations Peoples and their elders, past, present and future.
            [linkedin](https://www.linkedin.com/in/russell-jarvis-jarrod/)
            [github](https://github.com/russelljjarvis)
            [email](russelljarvis42@gmail.com)                        
            [The Code For This](https://github.com/russelljjarvis/NatureStewards/blob/main/appEVCs.py)
            [Personal Website]()
            ''')

    with col2:

        
        used_scheme = st.radio("Selection Descriptor:",["Short EVC Description","Long EVC Description (higher specificity of Bioregion)"],index=0)

        if used_scheme == "Long EVC Description (higher specificity of Bioregion)":
            used_scheme = "X_EVCNAME"
            
            
        if used_scheme == "Short EVC Description":
            used_scheme = "X_GROUPNAME"

        EVC_name_dict = dict((k,v) for k,v in enumerate(Bendigodf[used_scheme].values))

        # Add vertical scroll for radio.
        st.markdown("""
        <style>
            .row-widget {
                height: 200px;
                overflow-y: scroll;
            }
        </style>
        """,
        unsafe_allow_html=True)
        big_list = set(EVC_name_dict.values())
        choice_EVC = st.radio('Scrollable EVC Select', big_list, label_visibility='collapsed',index=1, key='rb_1')
    with col3:

        choice_Plot = st.radio("Choose Plot Type",["EVC at My Current Location","Ecotones+Selected EVC","All the EVCs togethor","Selected EVC","EVC Relative Area Pie Chart","Ecotone","Network of Neighbouring EVCs","Static Network of Neighbours","Municipilities of Bendigo","Re-Hashed"],index=0)
    if choice_Plot == "EVC at My Current Location":
        g = geocoder.ip('me')
  
        polygon_index = Bendigodf.geometry.distance(Point(g.latlng)).sort_values().index[0]
        subset = Bendigodf[Bendigodf.index==polygon_index]

        m = subset.explore()
        outfp = r"your_current_location.html"
        m.save(outfp)
        HtmlFile = open(f'your_current_location.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=435)

    if choice_Plot=="Re-Hashed":

        source_geometries = Bendigodf[Bendigodf[used_scheme]==choice_EVC]
        target_geometries = Bendigodf.geometry
        target_geometries_ = Bendigodf
        with st.spinner('Wait for it...'):
            engine = AdjacencyEngine(source_geometries.geometry, target_geometries.values)
            output = engine.get_adjacency_dict()
            total_df = geopandas.GeoDataFrame(columns=Bendigodf.columns)#, geometry='feature')#, crs='EPSG:4326')

            for source_i, target_i_list in output.items():                
                source_geom =  source_geometries[source_geometries.index==source_i]
                total_df = pd.concat([total_df,source_geom])

                dests = target_geometries_[target_geometries_.index.isin(target_i_list)]    

                total_df = pd.concat([total_df,dests])

            #st.success("Finished")                
        #total_df.explore(used_scheme)

        m = total_df.explore(used_scheme)
        outfp = r"adjacent_indexs.html"
        m.save(outfp)
        HtmlFile = open(f'adjacent_indexs.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=435)

        #adjacency_dict = output
    if choice_Plot=="Static Network of Neighbours":
        temp = Bendigodf[Bendigodf[used_scheme]==choice_EVC]
        compute_adjacency(temp)
    #temp = adjacency_dict[number_choice]
    #st.write(adjacency_dict)
    choice_df_index = Bendigodf_[Bendigodf_[used_scheme]==choice_EVC].index

    ecotones = get_adjacency_net_old(choice_df_index,adjacency_dict,EVC_name_dict,choice_Plot)

    #for k,v in adjacency_dict.items():
    #    if len(v)!=0:
    #        print(choice_df_index)
    targets_locations = []
    for i in choice_df_index:
        if len(adjacency_dict[i]) !=0:
            targets_locations.extend(adjacency_dict[i]) 
    adjacent_indexs = targets_locations    
    adjacent_indexs.extend(choice_df_index)

    st.header("Ecotones of: {0}".format(choice_EVC))

    dests = Bendigodf[Bendigodf.index.isin(adjacent_indexs)]    
    origins = Bendigodf[Bendigodf.index.isin(choice_df_index)]    

    temp = Bendigodf[Bendigodf.index.isin(adjacent_indexs)]    
    #st.write(g.latlng)
    slow_do_last(Bendigodf,ecotones,adjacent_indexs,choice_EVC,choice_Plot,used_scheme)

    if False:
    
        m = temp.explore(used_scheme)
        outfp = r"adjacent_indexs.html"
        m.save(outfp)
        HtmlFile = open(f'adjacent_indexs.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=435)

    # #m = folium.Map()#location=[39.949610, -75.150282], zoom_start=16)


    # # call to render Folium map in Streamlit
    # st_data = st_folium(m, width=725)
    # transformer = Transformer.from_crs("epsg:2154", "epsg:4326")
    # for origin,dest in zip(origins["geometry"],dests["geometry"]):
    #     latlong_origin = list(transformer.transform(origin.centroid.x, origin.centroid.y))
    #     latlong_destination = list(transformer.transform(dest.centroid.x, dest.centroid.y))
    #     #for origin_, destination in zip(latlong_origin, latlong_destination):
    #     #    st.write(origin_)
    #     folium.CircleMarker([latlong_origin[0], latlong_origin[1]],
    #                         radius=15,
    #                         fill_color="#3db7e4", # divvy color
    #                     ).add_to(m)
    #     folium.Marker(
    #         [39.949610, -75.150282], popup="Liberty Bell", tooltip="Liberty Bell"
    #     ).add_to(m)
    #     folium.CircleMarker([latlong_destination[0], latlong_destination[1]],
    #                         radius=15,
    #                         fill_color="red", # divvy color
    #                     ).add_to(m)

    #     folium.PolyLine([[latlong_origin[0], latlong_origin[1]], 
    #                     [latlong_destination[0], latlong_destination[1]]]).add_to(m)
    # #m
    # #folium.LayerControl().add_to(m)

    # #Draw(export=True).add_to(m)
    # #st_data = st_folium(m, width=725)
    
    #print(targets_locations)
    #import pdb
    #pdb.set_trace()


    #adjacent_indexs = [v for number_choice in choice_df_index for v in adjacency_dict[number_choice] ]
    #adjacent_indexs.extend(choice_df_index)
# def venn_diagram():
#     import pylab as plt
#     from matplotlib_venn import venn3, venn3_circles

#     v = venn3(subsets=(1,1,0,1,0,0,0))
#     v.get_label_by_id('100').set_text('First')
#     v.get_label_by_id('010').set_text('Second')
#     v.get_label_by_id('001').set_text('Third')
#     plt.title("Not a Venn diagram")
#     plt.show() 
    #compute_adjacency(Bendigodf)
#@st.cache_data
def whole_population_render(Bendigodf,used_scheme):
    if os.path.isfile("fully_populated_map_evcs.html"):
        HtmlFile = open(f'fully_populated_map_evcs.html', 'r', encoding='utf-8')
    else:
        m = Bendigodf.explore(used_scheme)#, cmap="Blues")
        outfp = r"fully_populated_map_evcs.html"
        m.save(outfp)
        HtmlFile = open(f'fully_populated_map_evcs.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=435)

def slow_do_last(Bendigodf,ecotones,adjacencies,choice_EVC,choice_Plot,used_scheme):
    ecotones = list(ecotones)
    filtered_eco_tern = Bendigodf[Bendigodf[used_scheme].isin(ecotones)]
    filtered_eco_tern_close = Bendigodf[Bendigodf.index.isin(adjacencies)]

    SingleEVC = Bendigodf[Bendigodf[used_scheme]==choice_EVC]

    if choice_Plot == "EVC Relative Area Pie Chart":

        height_ = 600


        st.header("Chart of All EVCs (Cluttered)")
        fig = px.pie(Bendigodf, values='AREASQM', names=used_scheme, title="Metres Square Area  SQM of each EVC",)
        fig.update_layout(height=height_)
        st.plotly_chart(fig, theme=None, use_container_width = True,height=height_)

        st.header("EVCs with total area > 10000000m^{2}")
        temp = Bendigodf[Bendigodf["AREASQM"]>=10000000]
        fig = px.pie(temp, values='AREASQM', names=used_scheme, title="Metres Square Area  SQM of each EVC",)
        fig.update_layout(height=height_)
        st.plotly_chart(fig, theme=None, use_container_width = True,height=height_)

        st.header("EVCs with total area <= 10000000m^{2}")

        temp = Bendigodf[Bendigodf["AREASQM"]<=10000000]
        fig = px.pie(temp, values='AREASQM', names=used_scheme, title="Metres Square Area of each EVC",)
        fig.update_layout(height=height_)
        st.plotly_chart(fig, theme=None, use_container_width = True,height=height_)

    if choice_Plot == "All the EVCs togethor":
        st.subheader("All the EVCs")
        whole_population_render(Bendigodf,used_scheme)



    def displayPDF(file):
        # Opening file from file path
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        # Embedding PDF in HTML
        pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

        # Displaying File
        st.markdown(pdf_display, unsafe_allow_html=True)        

    if choice_Plot == "Municipilities of Bendigo":
        gdfBendigo = goldfieldslocations()
        m = gdfBendigo.explore("vic_loca_2")
        outfp = r"base_map.html"
        m.save(outfp)
        HtmlFile = open(f'base_map.html', 'r', encoding='utf-8')
        col1, col2 = st.columns(3)
        with col1:
        # Load HTML file in HTML component for display on Streamlit page
            components.html(HtmlFile.read(), height=435)
        with col2:
           displayPDF("NN Network Map.pdf")



        #st.map(m)

    if choice_Plot == "Selected EVC":
        st.subheader("Selected EVCs")

        filtered_Bendigodf = Bendigodf[Bendigodf[used_scheme]==choice_EVC]

        #filtered_Bendigodf.plot(column=used_scheme,figsize=(5, 5), legend=True)
        #plt.legend(fontsize="x-small")

        #st.pyplot()

        #filtered_Bendigodf = filtered_Bendigodf.to_crs(epsg=4326)
        #mixture = pd.concat([filtered_Bendigodf,gdfBendigo])
        m = filtered_Bendigodf.explore(used_scheme)
        outfp = r"base_map.html"
        m.save(outfp)
        HtmlFile = open(f'base_map.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=435)


    if choice_Plot == "Ecotone":


        #filtered_eco_tern = filtered_eco_tern.to_crs(epsg=4326)
        m = filtered_eco_tern_close.explore(used_scheme)
        outfp = r"base_map.html"


        m.save(outfp)
        HtmlFile = open(f'base_map.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=435)


        #tab0,tab1 = st.tabs(["proximal","anywhere"])
        #with tab0:
    if choice_Plot == "Ecotones+Selected EVC":

        bothdf = pd.concat([SingleEVC, filtered_eco_tern])#, on='geometry', how='outer', suffixes=('_df1', '_df2')).fillna(0)
        outfp = r"base_map.html"
        m = bothdf.explore(used_scheme)        
        m.save(outfp)
        HtmlFile = open(f'base_map.html', 'r', encoding='utf-8')
        # Load HTML file in HTML component for display on Streamlit page
        st.header("Ecotones of: {0}".format(choice_EVC))
        components.html(HtmlFile.read(), height=435)
#from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#import matplotlib.cm as cm


#def dontdoagain():
def get_adjacency_net_old(choice_df_index,adjacency_dict,EVC_name_dict,choice_Plot):
    links_ = []
    nodes_ = []
    weight_dict_ = dict()
    cnt = 0
    #st.write(choice_df_index)

    for src in choice_df_index: #indexs of the geopandas geometries corresponding to the choice of EVC
        for tgt in adjacency_dict[src]:
            keyd = str(set([EVC_name_dict[src],EVC_name_dict[tgt]]))
            if keyd not in weight_dict_.keys():
                weight_dict_[keyd] = 1
            else:
                weight_dict_[keyd] += 1

    #st.write(weight_dict_.keys())
    category_weight_dict_= dict()
    for (ind,k) in enumerate(weight_dict_.keys()):
        category_weight_dict_[k] = ind

    ecotones = []

    for src in choice_df_index:
        for tgt in adjacency_dict[src]:
            keyd = str(set([EVC_name_dict[src],EVC_name_dict[tgt]]))
            
            links_.append({"source":EVC_name_dict[src],"target":EVC_name_dict[tgt],"value":weight_dict_[keyd]})
            nodes_.append({"index":cnt, "group":category_weight_dict_[keyd], "name":EVC_name_dict[src]})
            ecotones.append(EVC_name_dict[src])
            cnt+=1

    links = pd.DataFrame(links_)
    nodes = pd.DataFrame(nodes_)
    ecotones = set(ecotones)

    if "Network of Neighbouring EVCs" == choice_Plot and len(links)==0:
        st.error("Sorry No Ecoterns Found for that EVC")

    #for number_choice in choice_df_index:
    if "Network of Neighbouring EVCs" == choice_Plot and len(links)!=0:

        G = nx.from_pandas_edgelist(links, 'source', 'target', 'value')
        EVC_net = Network(
                            height='400px',
                            width='100%',
                            bgcolor='#222222',
                            font_color='white'
                            )

        EVC_net.from_nx(G)

        neighbor_map = EVC_net.get_adj_list()
        for ind,node in enumerate(EVC_net.nodes):
            #node["id"] = ind
            #node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
            node["value"] = len(neighbor_map[node["id"]])
            #print(colors[ind])
            node["group"] = ind
            node["color"] = ind

        #for ind,e in enumerate(EVC_net.edges):
        #    e["value"] = weight_dict_[ind]
            #    src = e[0]
            #    dst = e[1]
            #    w = e[2]            
        # Generate network with specific layout settings
        EVC_net.repulsion(
                            node_distance=420,
                            central_gravity=0.33,
                            spring_length=110,
                            spring_strength=0.10,
                            damping=0.95
                            )
        #viridis = cm.get_cmap('viridis', len(EVC_net.nodes))
        #colors = cm.rainbow(np.linspace(0, 1, len(EVC_net.nodes)))
        # Save and read graph as HTML file (on Streamlit Sharing)
        path = os.getcwd()
        EVC_net.save_graph(f'pyvis_graph.html')
        HtmlFile = open(f'pyvis_graph.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page

        #c = st.container()
        components.html(HtmlFile.read(), height=635)


    #st.write(len(adjacent_indexs))
    #st.write(len(ecotones))
    #st.write(ecotones)
    #components.html(HtmlFile.read(), height=635)

    #adjacent_indexs = [v for v in adjacency_dict[number_choice] ]
    #print(adjacent_indexs)
    #print(number_choice)
    return ecotones
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

def folium():
    ##
    # CRS is necessary for folium
    
    filtered_Bendigodf = filtered_Bendigodf.to_crs(epsg=4326)
    #print(filtered_Bendigodf.crs)
    #st.write(filtered_Bendigodf.crs)

    filtered_Bendigodf.explore(
    column=used_scheme,  # make choropleth based on "BoroName" column
    tooltip=used_scheme,  # show "BoroName" value in tooltip (on hover)
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
    #temp_json = Bendigodf[Bendigodf[used_scheme]==choice_EVC].to_json()

    st.write(filtered_Bendigodf)
    #plt.show()
    for _, r in filtered_Bendigodf.iterrows():
        # Without simplifying the representation of each borough,
        # the map might not be displayed
        sim_geo = gpd.GeoSeries(r["geometry"]).simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        print(geo_j)
        geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "orange"})
        folium.Popup(r[used_scheme]).add_to(geo_j)
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
    #         fields=['X_EVCNAME'],
    #         aliases=['EVC'],
    #         style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    #     )
    # )
    # vic_inc_map.add_child(BioRegionName)
    # vic_inc_map.keep_in_front(BioRegionName)



def get_adjacency_net(choice_df_index,adjacency_dict,EVC_name_dict,choice_Plot):
    #def dumb2():
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

    ecotones = []

    for number_choice in choice_df_index:
        for v in adjacency_dict[number_choice]:
            keyd = str(set([EVC_name_dict[number_choice],EVC_name_dict[v]]))
            links_.append({"source":EVC_name_dict[number_choice],"target":EVC_name_dict[v],"value":weight_dict_[keyd]})
            nodes_.append({"index":cnt, "group":category_weight_dict_[keyd], "name":EVC_name_dict[number_choice]})
            ecotones.append(EVC_name_dict[number_choice])
            cnt+=1

    links = pd.DataFrame(links_)
    nodes = pd.DataFrame(nodes_)
    ecotones = set(ecotones)


    #for number_choice in choice_df_index:
    if "Network of Neighbouring EVCs" == choice_Plot:
        def dumb():
            sources = []
            targets = []
            weights_ = []
            weights = {}
            for src,v in adjacency_dict.items():
                for tgt in v:
                    weights[tgt] = 0

            for src,v in adjacency_dict.items():
                for tgt in v:
                    weights[tgt] += 1


            for src,v in adjacency_dict.items():
                for tgt in v:
                    sources.append(src)
                    targets.append(tgt)
                    weights_.append(weights[tgt])
        

        new_adjacency_dict = dict()

        for number_choice in choice_df_index:
            #for v in adjacency_dict[number_choice]
            if len(adjacency_dict[number_choice]) != 0:
                new_adjacency_dict[number_choice] = adjacency_dict[number_choice]
        #import pdb
        #pdb.set_trace()
        H = nx.Graph(new_adjacency_dict) 


        EVC_net = Network(
                    height='400px',
                    width='100%',
                    bgcolor='#222222',
                    font_color='white'
                    )

        EVC_net.from_nx(H)
        for ind,node in enumerate(EVC_net.nodes):
            #node["id"] = ind
            node["title"] = str(EVC_name_dict[ind])
        
        #EVC_net =Network.from_nx(H)

        #st.write(sources,targets)
        #G = pyvis.Network.barnes_hut()
        #EVC_net = Network()
        #EVC_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
        def dumb3():

            edge_data = zip(sources, targets, weights)
            #st.write(edge_data)
            for e in edge_data:
                src = e[0]
                dst = e[1]
                w = e[2]

                EVC_net.add_node(src, src, title=src)
                EVC_net.add_node(dst, dst, title=dst)
                EVC_net.add_edge(src, dst, value=w)

        #st.write(len(EVC_net.nodes))
        #G = nx.from_pandas_edgelist(links, 'source', 'target', 'value')
        #EVC_net = Network(
        #                    height='400px',
        #                    width='100%',
        #                    bgcolor='#222222',
        #                    font_color='white'
        #                    )


        #neighbor_map = EVC_net.get_adj_list()
        
        
        #for ind,node in enumerate(EVC_net.nodes):
            #node["id"] = ind
            #node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
        #    node["value"] = len(neighbor_map[node["id"]])
            #print(colors[ind])
        #    node["group"] = ind
        #    node["color"] = ind
        # Generate network with specific layout settings
        EVC_net.repulsion(
                            node_distance=420,
                            central_gravity=0.33,
                            spring_length=110,
                            spring_strength=0.10,
                            damping=0.95
                            )
        #viridis = cm.get_cmap('viridis', len(EVC_net.nodes))
        #colors = cm.rainbow(np.linspace(0, 1, len(EVC_net.nodes)))
        # Save and read graph as HTML file (on Streamlit Sharing)

        #EVC_net.show("gameofthrones.html")
        EVC_net.save_graph(f'pyvis_graph.html')

        HtmlFile = open(f'pyvis_graph.html', 'r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
            #EVC_net.save_graph(f'{path}/pyvis_graph.html')
            #HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page

        #c = st.container()
        components.html(HtmlFile.read(), height=635)


    adjacent_indexs = [v for number_choice in choice_df_index for v in adjacency_dict[number_choice] ]
    #adjacent_indexs = [v for v in adjacency_dict[number_choice] ]
    #print(adjacent_indexs)
    #print(number_choice)
    return ecotones,adjacent_indexs
