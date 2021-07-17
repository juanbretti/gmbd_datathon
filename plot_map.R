# /https://gis.stackexchange.com/questions/354984/igraph-matching-coordinates-of-map-project-from-lon-lat-to-a-x-y-scale

# Load packages.
library(rgeos)
library(rnaturalearth)
library(igraph)
library(sf)
library(tmap)
library(dplyr)
library(sfnetworks)
library(tidygraph)

# Load countries from package. Use European countries.
countries <- st_as_sf(ne_countries(scale = 110, continent = "Europe"))
countries <- countries[!countries$adm0_a3 %in% c("RUS", "FRA"),]

countries <- st_as_sf(ne_states(country = 'spain'))
# Populate matrix for with a three ties.
m <- matrix(0, length(countries$adm0_a3), length(countries$adm0_a3))
m[2,3] <- 1
m[3,6] <- 1
m[10,2] <- 1
# Add row and column names so that there is an index available 
# for identifying the nodes in the graph
row.names(m) <- countries$adm0_a3
colnames(m) <- countries$adm0_a3

# Get igraph object from matrix.
g <- graph_from_adjacency_matrix(m)

# Convert to tbl_graph
g <- as_tbl_graph(g) 

# Get centroids from countries 
countries_centroid <- countries %>% 
  st_transform(3042) %>% 
  st_centroid()
#> Warning in st_centroid.sf(.): st_centroid assumes attributes are constant over
#> geometries of x

g_sfn <- g %>% 
    left_join(countries_centroid, by = c('name'='adm0_a3')) %>% 
    # This can be directly converted to an sfnetwork because a POINT
    # geometry column was added during the left join
    # edges_as_lines creates spatially explicit edges
    as_sfnetwork(directed = TRUE, edges_as_lines = TRUE)

# Plot with tmap
tm_shape(countries) +
  tm_polygons(col = "white") +
  tm_shape(st_as_sf(g_sfn, 'edges')) +
  tm_lines(lwd = 2, col = 'red') +
  tm_shape(st_as_sf(g_sfn, 'nodes')) +
  tm_dots(col = 'yellow', size = 0.5) 
