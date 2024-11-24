import pandas as pd
import unicodedata
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Cargar el dataset
dataset = pd.read_csv('instrumentos.csv')

# Entrenar el modelo de recomendación
def train_model(dataset):
    # Crear datos simulados de calificaciones
    rating_data = dataset[['Nombre_instrumento', 'Precio_real']].copy()
    rating_data['user_id'] = 1  # Simular un único usuario
    rating_data['rating'] = rating_data['Precio_real'].rank(method='dense', ascending=False)  # Rank como ratings
    
    # Convertir al formato de Surprise
    reader = Reader(rating_scale=(1, 10))
    surprise_data = Dataset.load_from_df(rating_data[['user_id', 'Nombre_instrumento', 'rating']], reader)
    
    # Dividir los datos en entrenamiento y prueba
    trainset, _ = train_test_split(surprise_data, test_size=0.2)
    
    # Entrenar un modelo basado en SVD
    model = SVD()
    model.fit(trainset)
    return model

# Normalizar texto eliminando acentos
def normalize_text(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()

# Obtener preferencias del usuario con validaciones
def get_user_preferences():
    print("Por favor, responde las siguientes preguntas para encontrar el instrumento más adecuado para ti.")
    
    def ask_question(question, valid_options):
        while True:
            answer = input(question).strip()
            normalized_answer = normalize_text(answer)
            normalized_options = [normalize_text(opt) for opt in valid_options]
            if normalized_answer in normalized_options:
                return valid_options[normalized_options.index(normalized_answer)]
            else:
                print(f"Respuesta no válida. Opciones válidas: {', '.join(valid_options)}")
    
    tipo = ask_question("¿Qué tipo de instrumento prefieres (Cuerda, Viento, Percusión, etc.)? ", 
                        dataset['Tipo'].unique())
    ruido = ask_question("¿Cuánto ruido estás dispuesto a tolerar (Bajo, Medio, Alto)? ", 
                         dataset['Ruido'].unique())
    rango_precio = ask_question("¿Cuál es tu rango de precio preferido (Económico, Medio, Alto)? ", 
                                dataset['Rango de precio'].unique())
    necesidad_cambio = ask_question("¿Es importante que no necesites cambiarlo con los años (Sí, No)? ", 
                                     dataset['Necesidad de cambiar con los años'].unique())
    complejidad = ask_question("¿Qué nivel de complejidad estás dispuesto a manejar (Bajo, Medio, Alto)? ", 
                               dataset['Nivel de complejidad'].unique())
    tipo_sonido = ask_question("¿Qué tipo de sonido prefieres (Acústico o Electrónico)? ", 
                               dataset['Tipo de sonido'].unique())
    mantenimiento = ask_question("¿Cuánto mantenimiento estás dispuesto a hacer (Bajo, Medio, Alto)? ", 
                                  dataset['Mantenimiento'].unique())
    portabilidad = ask_question("¿Qué tan importante es la portabilidad (Moderada, Difícil, Fácil)? ", 
                                 ["Moderada", "Difícil", "Fácil"])
    popularidad = ask_question("¿Prefieres un instrumento popular (Alta, Media, Baja)? ", 
                                dataset['Popularidad'].unique())
    
    return {
        "Tipo": tipo,
        "Ruido": ruido,
        "Rango de precio": rango_precio,
        "Necesidad de cambiar con los años": necesidad_cambio,
        "Nivel de complejidad": complejidad,
        "Tipo de sonido": tipo_sonido,
        "Mantenimiento": mantenimiento,
        "Portabilidad": portabilidad,
        "Popularidad": popularidad,
    }

# Recomendación basada en las preferencias del usuario
def recommend_instruments(preferences, dataset, model, top_n=5):
    filtered_data = dataset.copy()
    
    # Prioridad de los parámetros basada en la puntuación
    parameter_priority = [
        "Tipo", "Ruido", "Rango de precio", 
        "Necesidad de cambiar con los años", 
        "Nivel de complejidad", "Tipo de sonido", 
        "Mantenimiento", "Portabilidad", "Popularidad"
    ]
    
    # Filtrar exactos
    for key, value in preferences.items():
        filtered_data = filtered_data[filtered_data[key].str.contains(value, case=False, na=False)]
    
    if len(filtered_data) >= top_n:
        precise_data = filtered_data
    else:
        precise_data = dataset  # Relajado
    
    recommendations = []
    
    # Puntuación personalizada basada en coincidencias
    for _, row in precise_data.iterrows():
        score = 10  # Puntuación inicial
        for key, value in preferences.items():
            if row[key] != value:
                score -= 1  # Reducir la puntuación por cada coincidencia faltante
        recommendations.append((row['Nombre_instrumento'], row['Intruemnto'], score, row['Precio_real']))
    
    # Ordenar por puntuación
    recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)
    return recommendations[:top_n]

# Entrenar el modelo
model = train_model(dataset)

# Obtener las preferencias del usuario
user_preferences = get_user_preferences()

# Generar recomendaciones
recommended_instruments = recommend_instruments(user_preferences, dataset, model)

# Mostrar resultados
print("\nLos mejores instrumentos para ti son:")
for name, instrument, score, price in recommended_instruments:
    print(f"- {name} ({instrument}): {price} EUR, Puntuación estimada: {score:.2f}")
