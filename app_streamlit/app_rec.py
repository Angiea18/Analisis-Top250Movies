import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os 

# Cargar el DataFrame original
df = pd.read_csv("IMDB_Top250Movies.csv")

# Formatear los años para eliminar los separadores de miles
df['year'] = df['year'].apply(lambda x: str(x).replace(',', ''))

# Divide la columna de género en columnas separadas (usando get_dummies para la codificación one-hot)
df_genre = df['genre'].str.get_dummies(sep=',')

# Combina el DataFrame original con el DataFrame de género
df = pd.concat([df, df_genre], axis=1)

# Calcular la matriz de similitud de coseno
features = df_genre.columns
similarity_matrix = cosine_similarity(df[features], df[features])

def obtener_recomendaciones(pelicula_referencia, similarity_matrix, df, n=10):
    indice_referencia = df[df['name'] == pelicula_referencia].index[0]
    similitud_pelicula_referencia = similarity_matrix[indice_referencia]
    puntuaciones_similitud = list(enumerate(similitud_pelicula_referencia))
    puntuaciones_similitud = sorted(puntuaciones_similitud, key=lambda x: x[1], reverse=True)
    recomendaciones = puntuaciones_similitud[1:n+1]
    nombres_recomendados = [df.iloc[i]['name'] for i, _ in recomendaciones]
    return nombres_recomendados

# Crear una aplicación Streamlit
st.title("Welcome to FilmFinder!")
st.markdown("Movie Recommendation System.")

# Sección de la página de inicio con GIF animado
st.image("cine.gif", width=500)

# Crear un menú desplegable con la lista de películas
pelicula_referencia = st.selectbox('Select a reference movie', df['name'])

# Botón para generar recomendaciones
if st.button("Generate Recommendations"):
    if pelicula_referencia:
        # Corregir el cálculo de recomendaciones
        recomendaciones = obtener_recomendaciones(pelicula_referencia, similarity_matrix, df)
        
        # Excluir la película de referencia de las recomendaciones
        recomendaciones = [r for r in recomendaciones if r != pelicula_referencia]

        # Crear una tabla para mostrar las recomendaciones
        table_data = []

        for recomendacion in recomendaciones:
            info_adicional = df[df['name'] == recomendacion][['year', 'run_time', 'genre']].values

            # Asegurarse de que info_adicional es un array unidimensional
            if len(info_adicional) > 0:
                info_adicional = info_adicional[0]

                # Añadir datos a la tabla
                table_data.append({
                    'Name': recomendacion,
                    'Year': info_adicional[0],
                    'Duration': info_adicional[1],
                    'Genre': info_adicional[2]
                })

        if table_data:
            # Intentar crear el DataFrame
            df_table = pd.DataFrame(table_data)

            # Mostrar la tabla
            st.table(df_table[['Name', 'Year', 'Duration', 'Genre']])
        else:
            st.write("No recommendations found.")
    else:
        st.write("Please select a reference movie.")
    