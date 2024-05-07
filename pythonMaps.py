# https://jingwen-z.github.io/how-to-draw-a-variety-of-maps-with-folium-in-python/
import folium
from matplotlib import pyplot as plt

# importing all necessary libraries
import folium
import geopandas as gpd
import pandas as pd	 
import pdb
#syd_inc_map = folium.Map(location=[y_map, x_map], zoom_start=11,tiles=None)

syd_inc_map = folium.Map([-36.7569, 144.2786], zoom_start=11,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(syd_inc_map)



#Loading data dependencies using geopandas
vic_geo = gpd.read_file("suburb-10-vic.geojson")
#print(vic_geo.head(1))
	
vic_geo = vic_geo[['vic_loca_2','geometry']]

Bendigo = vic_geo[vic_geo["vic_loca_2"] == "BENDIGO"]

print(Bendigo.head(1))
print(dir(syd_inc_map))

#pdb.set_trace()
# Creating a map object
#melb_inc_map = folium.Map(location=[y_map, x_map], zoom_start=6,tiles=None)

# Creating choropleth 
folium.Choropleth(geo_data=vic_geo,
             name='Choropleth',         
             data=vic_geo,
             columns=['vic_loca_2','geometry'], 
             fill_color='YlOrRd',
             fill_opacity=0.6, 
             line_opacity=0.8,
             smooth_factor=0,     
             highlight=True,
                 ).add_to(syd_inc_map) 


folium.LayerControl().add_to(syd_inc_map)
syd_inc_map.show_in_browser()


"""
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

"""
