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


# Directorio que contiene los pósters de las películas
posters_directory = "Movies_posters"

movies_posters = {
"The Shawshank Redemption": "Movies_posters/The_Shawshank_Redemption.jpg",
"The Godfather": "Movies_posters/The_Godfather.jpg",
"The Dark Knight": "Movies_posters/The_Dark_Knight.jpg",
"The Godfather Part II": "Movies_posters\\The_Godfather_Part _II.jpg",
"12 Angry Men": "Movies_posters/12_Angry_Men.jpg",
"Schindler's List": "Movies_posters/Schindler's_List.jpg",
"The Lord of the Rings: The Return of the King": "Movies_posters/The_Lord_of_the_Rings_The_Return_of_the_King.jpg",
"Pulp Fiction": "Movies_posters/Pulp_Fiction.jpg",
"The Lord of the Rings: The Fellowship of the Ring": "Movies_posters/The_Lord_of_the_Rings_The_Fellowship_of_the_Ring.jpg",
"The Good, the Bad and the Ugly": "Movies_posters/The_Good_the_Bad_and_the_Ugly.jpg",
"Forrest Gump": "Movies_posters/Forrest_Gump.jpg",
"Fight Club": "Movies_posters/Fight_Club.jpg",
"The Lord of the Rings: The Two Towers": "Movies_posters/The_Lord_of_the_Rings_The_Two_Towers.jpg",
"Inception": "Movies_posters\Inception.jpg",
"Star Wars: Episode V - The Empire Strikes Back": "Movies_posters/Star_Wars_Episode_V_The_Empire_Strikes_Back.jpg",
"The Matrix": "Movies_posters/The_Matrix.jpg",
"Goodfellas": "Movies_posters/Goodfellas.jpg",
"One Flew Over the Cuckoo's Nest": "Movies_posters/One_Flew_Over_the_Cuckoo's_Nest.jpg",
"Se7en": "Movies_posters/Se7en.jpg",
"Seven Samurai": "Movies_posters/Seven_Samurai.jpg",
"It's a Wonderful Life": "Movies_posters/It's_a_Wonderful_Life.jpg",
"The Silence of the Lambs": "Movies_posters/The_Silence_of_the_Lambs.jpg",
"City of God": "Movies_posters/City_of_God.jpg",
"Saving Private Ryan": "Movies_posters/Saving_Private_Ryan.jpg",
"Interstellar": "Movies_posters/Interstellar.jpg",
"Life Is Beautiful": "Movies_posters/Life_Is_Beautiful.jpg",
"The Green Mile": "Movies_posters/The_Green_Mile.jpg",
"Star Wars: Episode IV - A New Hope": "Movies_posters/Star_Wars_Episode_IV_A_New_Hope.jpg",
"Terminator 2: Judgment Day": "Movies_posters/Terminator_2_Judgment_Day.jpg",
"Back to the Future": "Movies_posters/Back_to_the_Future.jpg",
"Spirited Away": "Movies_posters/Spirited_Away.jpg",
"The Pianist": "Movies_posters/The_Pianist.jpg",
"Psycho": "Movies_posters/Psycho.jpg",
"Parasite": "Movies_posters/Parasite.jpg",
"Léon: The Professional": "Movies_posters/Léon_The_Professional.jpg",
"The Lion King": "Movies_posters/The_Lion_King.jpg",
"Gladiator": "Movies_posters/Gladiator.jpg",
"American History X": "Movies_posters/American_History_X.jpg",
"The Departed": "Movies_posters/The_Departed.jpg",
"The Usual Suspects": "Movies_posters/The_Usual_Suspects.jpg",
"The Prestige": "Movies_posters/The_Prestige.jpg",
"Whiplash": "Movies_posters/Whiplash.jpg",
"Casablanca": "Movies_posters/Casablanca.jpg",
"Grave of the Fireflies": "Movies_posters/Grave_of_the_Fireflies.jpg",
"Harakiri": "Movies_posters/Harakiri.jpg",
"The Intouchables": "Movies_posters/The_Intouchables.jpg",
"Modern Times": "Movies_posters/Modern_Times.jpg",
"Once Upon a Time in the West": "Movies_posters/Once_Upon_a_Time_in_the_West.jpg",
"Rear Window": "Movies_posters/Rear_Window.jpg",
"Cinema Paradiso": "Movies_posters/Cinema_Paradiso.jpg",
"Alien": "Movies_posters/Alien.jpg",
"City Lights": "Movies_posters/City_Lights.jpg",
"Apocalypse Now": "Movies_posters/Apocalypse_Now.jpg",
"Memento": "Movies_posters/Memento.jpg",
"Django Unchained": "Movies_posters/Django_Unchained.jpg",
"Indiana Jones and the Raiders of the Lost Ark": "Movies_posters/Raiders_of_the_Lost_Ark.jpg",
"WALL·E": "Movies_posters/WALL·E.jpg",
"The Lives of Others": "Movies_posters/The_Lives_of_Others.jpg",
"Sunset Blvd.": "Movies_posters/Sunset_Blvd.jpg",
"Paths of Glory": "Movies_posters/Paths_of_Glory.jpg", 
"The Shining": "Movies_posters/The_Shining.jpg",
"The Great Dictator": "Movies_posters/The_Great_Dictator.jpg", 
"Avengers: Infinity War": "Movies_posters/Avengers_Infinity_War.jpg", 
"Witness for the Prosecution": "Movies_posters/Witness_for_the_Prosecution.jpg", 
"Aliens": "Movies_posters/Aliens.jpg", 
"Spider-Man: Into the Spider-Verse": "Movies_posters/Spider_Man_Into_the_Spider_Verse.jpg", 
"American Beauty": "Movies_posters/American_Beauty.jpg", 
"Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb": "Movies_posters/Dr_Strangelove_or_How_I_Learned_to_Stop_Worrying_and_Love_the_Bomb.jpg", 
"The Dark Knight Rises": "Movies_posters/The_Dark_Knight_Rises.jpg", 
"Oldboy": "Movies_posters/Oldboy.jpg", 
"Inglourious Basterds": "Movies_posters/Inglourious_Basterds.jpg", 
"Amadeus": "Movies_posters/Amadeus.jpg", 
"Coco": "Movies_posters/Coco.jpg", 
"Toy Story": "Movies_posters/Toy_Story.jpg", 
"Joker": "Movies_posters/Joker.jpg", 
"Braveheart": "Movies_posters/Braveheart.jpg", 
"The Boat": "Movies_posters/The_Boat.jpg", 
"Avengers: Endgame": "Movies_posters/Avengers_Endgame.jpg", 
"Princess Mononoke": "Movies_posters/Princess_Mononoke.jpg", 
"Once Upon a Time in America": "Movies_posters/Once_Upon_a_Time_in_America.jpg", 
"Good Will Hunting": "Movies_posters/Good_Will_Hunting.jpg", 
"Your Name": "Movies_posters/Your_Name.jpg", 
"3 Idiots": "Movies_posters/3_Idiots.jpg", 
"Singin' in the Rain": "Movies_posters/Singin_the_Rain.jpg", 
"Requiem for a Dream": "Movies_posters/Requiem_for_a_Dream.jpg", 
"Toy Story 3": "Movies_posters/Toy_Story_3.jpg", 
"High and Low": "Movies_posters/High_and_Low.jpg", 
"Capernaum": "Movies_posters/Capernaum.jpg", 
"Star Wars: Episode VI - Return of the Jedi": "Movies_posters/Star_Wars_Episode_VI_Return_of_the_Jedi.jpg", 
"Eternal Sunshine of the Spotless Mind": "Movies_posters/Eternal_Sunshine_of_the_Spotless_Mind.jpg", 
"2001: A Space Odyssey": "Movies_posters/2001_A_Space_Odyssey.jpg", 
"Reservoir Dogs": "Movies_posters/Reservoir_Dogs.jpg", 
"Come and See": "Movies_posters/Come_and_See.jpg", 
"The Hunt": "Movies_posters/The_Hunt.jpg", 
"Citizen Kane": "Movies_posters/Citizen_Kane.jpg", 
"M": "Movies_posters/M.jpg", 
"Lawrence of Arabia": "Movies_posters/Lawrence_of_Arabia.jpg", 
"North by Northwest": "Movies_posters/North_by_Northwest.jpg", 
"Vertigo": "Movies_posters/Vertigo.jpg", 
"Ikiru": "Movies_posters/Ikiru.jpg", 
"Amélie": "Movies_posters/Amélie.jpg", 
"The Apartment": "Movies_posters/The_Apartment.jpg", 
"A Clockwork Orange": "Movies_posters/A_Clockwork_Orange.jpg", 
"Double Indemnity": "Movies_posters/Double_Indemnity.jpg", 
"Full Metal Jacket": "Movies_posters/Full_Metal_Jacket.jpg", 
"Top Gun: Maverick": "Movies_posters/Top_Gun_Maverick.jpg", 
"Scarface": "Movies_posters/Scarface.jpg", 
"Hamilton": "Movies_posters/Hamilton.jpg", 
"Incendies": "Movies_posters/Incendies.jpg", 
"To Kill a Mockingbird": "Movies_posters/To_Kill_a_Mockingbird.jpg", 
"Heat": "Movies_posters/Heat.jpg", 
"The Sting": "Movies_posters/The_Sting.jpg", 
"Up": "Movies_posters/Up.jpg", 
"A Separation": "Movies_posters/A_Separation.jpg", 
"Metropolis": "Movies_posters/Metropolis.jpg",
"Taxi Driver": "Movies_posters/Taxi_Driver.jpg", 
"L.A. Confidential": "Movies_posters/L.A_Confidential.jpg", 
"Die Hard": "Movies_posters/Die_Hard.jpg", 
"Snatch": "Movies_posters/Snatch.jpg", 
"Indiana Jones and the Last Crusade": "Movies_posters/indiana_jones_and_the_last_crusade.jpg",
"Bicycle Thieves": "Movies_posters/Bicycle_Thieves.jpg", 
"Like Stars on Earth": "Movies_posters/Like_Stars_on_Earth.jpg", 
"1917": "Movies_posters/1917.jpg",
"Downfall": "Movies_posters/Downfall.jpg",
"Dangal": "Movies_posters/Dangal.jpg",
"For a Few Dollars More": "Movies_posters/For_a_Few_Dollars_More.jpg",
"Batman Begins": "Movies_posters/Batman_Begins.jpg", 
"The Kid": "Movies_posters/The_Kid.jpg", 
"Some Like It Hot": "Movies_posters/Some_Like_It_Hot.jpg", 
"The Father": "Movies_posters/The_Father.jpg", 
"All About Eve": "Movies_posters/All_About_Eve.jpg", 
"The Wolf of Wall Street": "Movies_posters/The_Wolf_of_Wall_Street.jpg", 
"Green Book": "Movies_posters/Green_Book.jpg", 
"Judgment at Nuremberg": "Movies_posters/Judgment_at_Nuremberg.jpg", 
"Casino": "Movies_posters/Casino.jpg", 
"Ran": "Movies_posters/Ran.jpg", 
"Pan's Labyrinth": "Movies_posters/Pan's_Labyrinth.jpg", 
"The Truman Show": "Movies_posters/The_Truman_Show.jpg", 
"There Will Be Blood": "Movies_posters/There_Will_Be_Blood.jpg", 
"Unforgiven": "Movies_posters/Unforgiven.jpg", 
"The Sixth Sense": "Movies_posters/The_Sixth_Sense.jpg", 
"Shutter Island": "Movies_posters/Shutter_Island.jpg",
"A Beautiful Mind": "Movies_posters/A_Beautiful_Mind.jpg",
"Jurassic Park": "Movies_posters/Jurassic_Park.jpg", 
"Yojimbo": "Movies_posters/Yojimbo.jpg",
"The Treasure of the Sierra Madre": "Movies_posters/The_Treasure_of_the_Sierra_Madre.jpg",
"Monty Python and the Holy Grail": "Movies_posters/Monty_Python_and_the_Holy_Grail.jpg",
"The Great Escape": "Movies_posters/The_Great_Escape.jpg",
"No Country for Old Men": "Movies_posters/No_Country_for_Old_Men.jpg",
"Spider-Man: No Way Home": "Movies_posters/Spider_Man_No_Way_Home.jpg",
"Kill Bill: Vol. 1": "Movies_posters/Kill_Bill_Vol_1.jpg",
"Rashomon": "Movies_posters/Rashomon.jpg",
"The Thing": "Movies_posters/The_Thing.jpg",
"Finding Nemo": "Movies_posters/Finding_Nemo.jpg",
"The Elephant Man": "Movies_posters/The_Elephant_Man.jpg",
"Chinatown": "Movies_posters/Chinatown.jpg",
"Raging Bull": "Movies_posters/Raging_Bull.jpg",
"V for Vendetta": "Movies_posters/V_for_Vendetta.jpg",
"Gone with the Wind": "Movies_posters/Gone_with_the_Wind.jpg",
"Lock, Stock and Two Smoking Barrels": "Movies_posters/Lock_Stock_and_Two_Smoking_Barrels.jpg",
"Inside Out": "Movies_posters/Inside_Out.jpg",
"Dial M for Murder": "Movies_posters/Dial_M_for_Murder.jpg",
"The Secret in Their Eyes": "Movies_posters/The_Secret_in_Their_Eyes.jpg",
"Howl's Moving Castle": "Movies_posters/Howl's_Moving_Castle.jpg",
"Three Billboards Outside Ebbing, Missouri": "Movies_posters/Three_Billboards_Outside_Ebbing_Missouri.jpg",
"The Bridge on the River Kwai": "Movies_posters/The_Bridge_on_the_River_Kwai.jpg",
"Trainspotting": "Movies_posters/Trainspotting.jpg",
"Prisoners": "Movies_posters\Prisoners.jpg",
"Warrior": "Movies_posters/Warrior.jpg",
"Fargo": "Movies_posters/Fargo.jpg",
"Gran Torino": "Movies_posters/Gran_Torino.jpg",
"My Neighbor Totoro": "Movies_posters/My_Neighbor_Totoro.jpg",
"Catch Me If You Can": "Movies_posters/Catch_Me_If_You_Can.jpg",
"Million Dollar Baby": "Movies_posters/Million_Dollar_Baby.jpg",
"Children of Heaven": "Movies_posters/Children_of_Heaven.jpg",
"Blade Runner": "Movies_posters/Blade_Runner.jpg",
"The Gold Rush": "Movies_posters/The_Gold_Rush.jpg",
"Before Sunrise": "Movies_posters/Before_Sunrise.jpg",
"12 Years a Slave": "Movies_posters/12_Years_a_Slave.jpg",
"Klaus": "Movies_posters/Klaus.jpg",
"Harry Potter and the Deathly Hallows: Part 2": "Movies_posters/Harry_Potter_and_the_Deathly_Hallows_Part_2.jpg",
"On the Waterfront": "Movies_posters/On_the_Waterfront.jpg",
"Ben-Hur": "Movies_posters/Ben_Hur.jpg",
"Gone Girl": "Movies_posters/Gone_Girl.jpg",
"The Grand Budapest Hotel": "Movies_posters/he_Grand_Budapest_Hotel.jpg",
"Wild Strawberries": "Movies_posters/Wild_Strawberries.jpg",
"The General": "Movies_posters/The_General.jpg",
"The Third Man": "Movies_posters/The_Third_Man.jpg",
"In the Name of the Father": "Movies_posters/In_the_Name_of_the_Father.jpg",
"The Deer Hunter": "Movies_posters/The_Deer_Hunter.jpg",
"Barry Lyndon": "Movies_posters/Barry_Lyndon.jpg",
"Hacksaw Ridge": "Movies_posters/Hacksaw_Ridge.jpg",
"The Wages of Fear": "Movies_posters/The_Wages",
"Memories of Murder": "Movies_posters/Memories_of_Murder.jpg",
"Sherlock Jr. ": "Movies_posters/Sherlock_Jr.jpg",
"Wild Tales": "Movies_posters/Wild_Tales.jpg",
"Mr. Smith Goes to Washington": "Movies_posters/Mr_Smith_Goes_to_Washington.jpg",
"Mad Max: Fury Road": "Movies_posters/Mad_Max_Fury_Road.jpg",
"The Seventh Seal": "Movies_posters/The_Seventh_Seal.jpg",
"Mary and Max": "Movies_posters/Mary_and_Max.jpg",
"How to Train Your Dragon": "Movies_posters/How_to_Train_Your_Dragon.jpg",
"Room": "Movies_posters/Room.jpg",
"Monsters, Inc. ": "Movies_posters/Monsters_Inc.jpg",
"Jaws": "Movies_posters/Jaws.jpg",
"Dead Poets Society": "Movies_posters/Dead_Poets_Society.jpg",
"The Big Lebowski": "Movies_posters/The_Big_Lebowski.jpg",
"Tokyo Story": "Movies_posters/Tokyo_Story.jpg",
"The Passion of Joan of Arc": "Movies_posters/The_Passion_of_Joan_of_Arc.jpg",
"Hotel Rwanda": "Movies_posters/Hotel_Rwanda.jpg",
"Ford v Ferrari": "Movies_posters/Ford_v_Ferrari.jpg",
"Rocky": "Movies_posters/Rocky.jpg",
"Platoon": "Movies_posters/Platoon.jpg",
"Ratatouille": "Movies_posters/Ratatouille.jpg",
"Spotlight": "Movies_posters/Spotlight.jpg",
"The Terminator": "Movies_posters/The_Terminator.jpg",
"Logan": "Movies_posters/Logan.jpg",
"Stand by Me": "Movies_posters/Stand_by_Me.jpg",
"Rush": "Movies_posters/Rush.jpg",
"Network": "Movies_posters/Network.jpg",
"Into the Wild": "Movies_posters/Into_the_Wild.jpg",
"Before Sunset": "Movies_posters/Before_Sunset.jpg",
"The Wizard of Oz": "Movies_posters/The_Wizard_of_Oz.jpg",
"Pather Panchali": "Movies_posters/Pather_Panchali.jpg",
"Groundhog Day": "Movies_posters/Groundhog_Day.jpg",
"The Best Years of Our Lives": "Movies_posters/The_Best_Years_of_Our_Lives.jpg",
"The Exorcist": "Movies_posters/The_Exorcist.jpg",
"The Incredibles": "Movies_posters/The_Incredibles.jpg",
"To Be or Not to Be": "Movies_posters/To_Be_or_Not_to_Be.jpg",
"La haine": "Movies_posters/La_haine.jpg",
"The Battle of Algiers": "Movies_posters/The_Battle_of_Algiers.jpg",
"Pirates of the Caribbean: The Curse of the Black Pearl": "Movies_posters/Pirates_of_the_Caribbean_The_Curse_of_the_Black_Pearl.jpg",
"Hachi: A Dog's Tale": "Movies_posters/Hachi_A_Dog's_Tale.jpg",
"The Grapes of Wrath": "Movies_posters/The_Grapes_of_Wrath.jpg",
"Jai Bhim": "Movies_posters/Jai_Bhim.jpg",
"My Father and My Son": "Movies_posters/My_Father_and_My_Son.jpg",
"Amores Perros": "Movies_posters/Amores_Perros.jpg",
"Rebecca": "Movies_posters/Rebecca.jpg",
"Cool Hand Luke": "Movies_posters/Cool_Hand_Luke.jpg",
"The Handmaiden": "Movies_posters/The_Handmaiden.jpg",
"The 400 Blows": "Movies_posters/The_400_Blows.jpg",
"The Sound of Music": "Movies_posters/The_Sound_of_Music.jpg",
"It Happened One Night": "Movies_posters/It_Happened_One_Night.jpg",
"Persona": "Movies_posters/Persona.jpg",
"Life of Brian": "Movies_posters/Life_of_Brian.jpg",
"The Iron Giant": "Movies_posters/The_Iron_Giant.jpg",
"The Help": "Movies_posters/The_Help.jpg",
"Dersu Uzala": "Movies_posters/Dersu_Uzala.jpg",
"Aladdin": "Movies_posters/Aladdin.jpg",
"Gandhi": "Movies_posters/Gandhi.jpg",
"Dances with Wolves": "Movies_posters/Dances_with_Wolves.jpg",
}


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
        recomendaciones = obtener_recomendaciones(pelicula_referencia, similarity_matrix, df)
        st.write(f"Recommendations for {pelicula_referencia}:")

        # Excluir la película de referencia de las recomendaciones
        recomendaciones = [r for r in recomendaciones if r != pelicula_referencia]

        for recomendacion in recomendaciones:
            poster_path = movies_posters.get(recomendacion)
            info_adicional = df[df['name'] == recomendacion][['name', 'year', 'run_time', 'genre']].values[0]

  
            col1, col2 = st.columns(2)  # Divide la pantalla en dos columnas

            with col1:
                # Mostrar la imagen (poster) en una columna y ajustar el tamaño con width
                st.image(poster_path, caption=recomendacion, width=100)

            with col2:
                # Mostrar la información adicional en la otra columna
                st.write("Name:", info_adicional[0])
                st.write("Year:", info_adicional[1])
                st.write("Duration:", info_adicional[2])
                st.write("Genre:", info_adicional[3])
    else:
        st.write("No recommendations found.")






