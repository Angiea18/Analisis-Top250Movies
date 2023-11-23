
# Top 250 IMDB Movies: Análisis y Sistema de Recomendación

Este proyecto se sumerge en el fascinante mundo del cine a través del análisis del Top 250 IMDB Movies, explorando las tendencias, patrones y características de las películas mejor valoradas. Además, implementamos un sistema de recomendación cinematográfica que utiliza datos detallados para sugerir películas personalizadas basadas en las preferencias del usuario. La combinación de análisis exhaustivo y recomendaciones inteligentes proporciona una experiencia cinematográfica única y adaptada a cada individuo, llevando la magia del cine directamente a los gustos personales de los espectadores.


## Dataset

El Dataset [IMDB Top 250 Movies Dataset](https://github.com/Angiea18/Analisis-Top250Movies/blob/main/IMDB_Top250Movies.csv) lo obtuve de la plataforma `**Kaggle**`, que esta conformado por 13 columnas y 250 filas, con información sobre el nombre, año de lanzamiento, duración, género, entre otros relacionados con las películas. 
Las descripciones de este se hacen en el [Diccionario de datos](https://github.com/Angiea18/Analisis-Top250Movies/blob/main/Diccionario.md)


- **Kaggle** es una plataforma en línea para científicos de datos que ofrece competiciones, conjuntos de datos y herramientas de codificación en la nube para fomentar la colaboración y el aprendizaje en ciencia de datos y aprendizaje automático.
# Exploración y Recomendación Cinematográfica

### EDA

Para comprender la estructura y composición del conjunto de datos, se realizó un Análisis Exploratorio de Datos (EDA). Este proceso permitió examinar en detalle la información contenida en el conjunto de datos, identificar patrones, entender las relaciones entre las variables y obtener una visión general de su distribución. El objetivo principal fue obtener insights valiosos que faciliten la interpretación y utilización efectiva de los datos en el análisis posterior.


Visualizaciones Destacadas:

- Top del Rating según el Género: Descubre las mejores películas clasificadas por género.
![]()
- Top 10 Películas por Género: Una mirada a las joyas cinematográficas en diferentes géneros.
![]()
- Top 10 Películas por Calificación (Rating): Las películas que destacan por sus altas calificaciones.
![]()
- Top 10 de Directores por Rating: Reconocimiento a los directores con las mejores calificaciones.
![]()
- Distribución de Ratings: Un histograma que revela la diversidad en las calificaciones.
![]()
- Calificaciones de Películas por Género y Año: Una visión temporal de las calificaciones en función del género.
![]()
- Calificación Promedio a lo Largo del Tiempo: Un viaje a través de las tendencias de calificación a lo largo de los años.
![Linechart](https://github.com/Angiea18/Analisis-Top250Movies/blob/main/_src/AvgCalificaciones.png)
- Calificaciones Promedio por Certificado: Descubre cómo se distribuyen las calificaciones según los certificados.
![]()
- Películas Más Antiguas: Una tabla que presenta las películas más antiguas del conjunto.
![]()
- Películas Más Recientes: Una vista de las películas más recientes.
![]()


### Sistema de Recomendación

En la sección de **Sistema de Recomendación de Películas**, se ha implementado un mecanismo que aprovecha el algoritmo de la similitud del coseno de la biblioteca Sklearn para ofrecer recomendaciones de películas personalizadas. Aquí se explica cómo funciona paso a paso:

1. Carga de Datos y Preprocesamiento:
- Se carga el conjunto de datos original de las 250 mejores películas de IMDB.
- Se formatean los años para eliminar los separadores de miles y se utiliza la función get_dummies de pandas para realizar la codificación one-hot de la columna 'genre'.
2. Cálculo de la Matriz de Similitud del Coseno:
- Se calcula la matriz de similitud del coseno utilizando la biblioteca scikit-learn. Esta matriz captura las relaciones de similitud entre todas las películas basándose en características clave como el género, duración, etc.
3. Interacción del Usuario:
- Los usuarios pueden seleccionar una película de referencia desde un menú desplegable.
4. Generación de Recomendaciones:
- Al hacer clic en el botón "Generar Recomendaciones", el sistema utiliza la película de referencia seleccionada para identificar las películas más similares.
- Se excluye la película de referencia de la lista de recomendaciones.
5. Presentación de Resultados:
- Se muestra una tabla con información detallada sobre las 10 mejores recomendaciones, incluyendo nombre, año de lanzamiento, duración y género.
![Resultado]()

Este sistema permite a los usuarios descubrir nuevas películas que son similares a sus elecciones favoritas, proporcionando una experiencia personalizada de exploración cinematográfica.


Tanto el Análisis Exploratorio de Datos como el Sistema de Recomendación están integrados en una interfaz de usuario atractiva creada con [Streamlit](https://6nmfcappdldccqiaub3yy5c.streamlit.app/). ¡Disfruta explorando y descubre nuevas joyas cinematográficas de manera personalizada! 🍿🎬




## Conclusiones

- La exploración de 'IMDB Top 250 Movies' destaca la diversidad y atemporalidad del cine. La presencia de diversos géneros, directores y años refleja la riqueza de la industria. Desde clásicos hasta películas contemporáneas, la lista demuestra que la calidad cinematográfica trasciende las décadas, ofreciendo una mirada fascinante a través del tiempo.
