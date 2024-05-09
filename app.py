# https://jingwen-z.github.io,/how-to-draw-a-variety-of-maps-with-folium-in-python/
#import folium
#from matplotlib import pyplot as plt
# importing all necessary libraries
import geopandas as gpd
import pandas as pd	 
import pdb
import streamlit as st
from streamlit_folium import st_folium

#import plotly.express as px
#import plotly.graph_objects as go
#from urllib.request import urlopen
#import json
##from copy import deepcopy
#from plotly.subplots import make_subplots
import folium
import copy
from folium.plugins import Draw
import pickle
st.set_page_config(layout="wide")
#st.set_page_config(layout="wide")



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


with open("EVCS_Goldfields_geom.p","rb") as f:
    Bendigodf = pickle.load(f)

#import pdb
#pdb.set_trace()


def main():
    st.header("EVCS of North Central Catchment Region")


    """
    fig = px.choropleth_mapbox(Bendigodf, geojson=Bendigodf.to_json(), color=Bendigodf.EVC,
                            color_continuous_scale="Viridis",
                            range_color=(0, 12),
                            mapbox_style="carto-positron",
                            zoom=12, 
                            center = {"lat": -36.7569, "lon": 144.2786},
                            opacity=0.1,
                            labels=Bendigodf.EVC,
                            width=900,
                            height=700
                            )

    """
    #fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    #st.plotly_chart(fig)
    # Creating a map object
    #melb_inc_map = folium.Map(location=[y_map, x_map], zoom_start=6,tiles=None)
    #gdfBendigo['new_col'] = range(1, len(gdfBendigo) + 1)
    #gdfBendigo
    # Creating choropleth
    vic_inc_map = folium.Map([-36.7569, 144.2786], zoom_start=10,tiles=None)
    #folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(vic_inc_map)
    #st.write(Bendigodf.columns)
    folium.Choropleth(geo_data=Bendigodf.to_json(),
                name='Choropleth',         
                data=Bendigodf,
                columns=['EVC','geometry'],
                fill_color='YlOrRd',
                fill_opacity=0.6, 
                line_opacity=0.8,
                smooth_factor=0,     
                highlight=False,
                    ).add_to(vic_inc_map) 
    folium.LayerControl().add_to(vic_inc_map)

    # Adding labels to map
    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.0, 
                                'weight': 0.1}
    BioRegionName = folium.features.GeoJson(
        Bendigodf,
        style_function=style_function, 
        control=False,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['X_GROUPNAME'],
            aliases=['EVC'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
        )
    )
    vic_inc_map.add_child(BioRegionName)
    vic_inc_map.keep_in_front(BioRegionName)
    folium.LayerControl().add_to(vic_inc_map)

    Draw(export=True).add_to(vic_inc_map)


    #vic_inc_map.show_in_browser()
    st_data = st_folium(vic_inc_map, width=1725)

    #output = st_folium(m, width=700, height=500)

if __name__ == "__main__":
    main()

def dontdo():
    suburbs = set(vic_geo["vic_loca_2"])
    sorted_suburbs = sorted(suburbs, reverse=False)
    print(sorted_suburbs)


def dontdo2():
    nbh_count_df = listing_df.groupby('neighbourhood')['id'].nunique().reset_index()
    nbh_count_df.rename(columns={'id':'nb'}, inplace=True)
    nbh_geo_count_df = pd.merge(nbh_geo_df, nbh_count_df, on='neighbourhood')
    nbh_geo_count_df['QP'] = nbh_geo_count_df['nb'] / nbh_geo_count_df['nb'].sum()
    nbh_geo_count_df['QP_str'] = nbh_geo_count_df['QP'].apply(lambda x : str(round(x*100, 1)) + '%')

    from branca.colormap import linear
    nbh_count_colormap = linear.YlGnBu_09.scale(min(nbh_count_df['nb']),
                                                max(nbh_count_df['nb']))

    nbh_locs_map = folium.Map(location=[48.856614, 2.3522219],
                            zoom_start = 12, tiles='cartodbpositron')

    style_function = lambda x: {
        'fillColor': nbh_count_colormap(x['properties']['nb']),
        'color': 'black',
        'weight': 1.5,
        'fillOpacity': 0.7
    }

    folium.GeoJson(
        nbh_geo_count_df,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['neighbourhood', 'nb', 'QP_str'],
            aliases=['Neighbourhood', 'Location amount', 'Quote-part'],
            localize=True
        )
    ).add_to(nbh_locs_map)

    nbh_count_colormap.add_to(nbh_locs_map)
    nbh_count_colormap.caption = 'Airbnb location amount'
    nbh_count_colormap.add_to(nbh_locs_map)

