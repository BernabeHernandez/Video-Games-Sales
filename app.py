from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
import numpy as np

app = Flask(__name__)

# Configurar el registr
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado, el scaler y el ordinal encoder
model = joblib.load('modelo_random_forest.pkl')
scaler = joblib.load('scaler.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')
app.logger.debug('Modelos y encoder cargados correctamente.')

# Cargar las opciones de categorías
category_options = joblib.load('category_options.pkl')
app.logger.debug('Opciones de categorías cargadas correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/get_categories')
def get_categories():
    return jsonify(category_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        rank = float(request.form['rank'])
        platform = request.form['platform']
        year = float(request.form['year'])
        genre = request.form['genre']
        publisher = request.form['publisher']

        # Verificar que las categorías existan en el encoder
        valid_platforms = ordinal_encoder.categories_[0]
        valid_genres = ordinal_encoder.categories_[1]
        valid_publishers = ordinal_encoder.categories_[2]
        if platform not in valid_platforms or genre not in valid_genres or publisher not in valid_publishers:
            raise ValueError(f"Alguna categoría no es válida. Debe estar en las listas de plataformas, géneros o publishers.")

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[rank, platform, year, genre, publisher]], 
                               columns=['Rank', 'Platform', 'Year', 'Genre', 'Publisher'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Transformar las variables categóricas con el OrdinalEncoder
        categorical_cols = ['Platform', 'Genre', 'Publisher']
        data_df[categorical_cols] = ordinal_encoder.transform(data_df[categorical_cols])
        app.logger.debug(f'DataFrame tras encoding: {data_df}')

        # Seleccionar solo las características usadas en el modelo
        X = data_df[['Rank', 'Platform', 'Year', 'Genre', 'Publisher']].values

        # Escalar los datos
        X_scaled = scaler.transform(X)
        app.logger.debug(f'Datos escalados: {X_scaled}')

        # Realizar predicciones
        prediction = model.predict(X_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'global_sales': round(prediction[0], 2)})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)