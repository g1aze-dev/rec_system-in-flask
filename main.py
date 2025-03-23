import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Загрузка данных из JSON файла
def load_data():
    with open('data/movies01.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Преобразование JSON в DataFrame
data = load_data()
df = pd.DataFrame(data)

# Вычисление среднего рейтинга для каждого фильма
df['averageRating'] = df['ratings'].apply(lambda x: sum(x) / len(x))

# Обработка жанров с помощью MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_)

# Объединение данных
df = pd.concat([df, genres_encoded], axis=1)

# Признаки для KNN
features = ['averageRating', 'year'] + list(mlb.classes_)
X = df[features]

# Обучение модели KNN
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X)

# Функция для поиска индекса фильма по названию
def find_movie_index_by_title(df, title):
    # Поиск индекса по точному совпадению названия
    match = df[df['title'].str.lower() == title.lower()]
    if not match.empty:
        return match.index[0]
    else:
        # Если точного совпадения нет, ищем частичное совпадение
        partial_match = df[df['title'].str.lower().str.contains(title.lower())]
        if not partial_match.empty:
            print("Точного совпадения не найдено. Возможно, вы имели в виду:")
            for i, row in partial_match.iterrows():
                print(f"- {row['title']} (индекс: {i})")
            return None
        else:
            print("Фильм не найден. Пожалуйста, проверьте название.")
            return None

# Функция для рекомендации фильмов
def recommend_movies(movie_index, knn_model, df):
    distances, indices = knn_model.kneighbors([X.iloc[movie_index]])
    recommendations = df.iloc[indices[0]]
    return recommendations

# Маршрут для главной страницы
@app.route('/')
def main():
    # Загружаем данные
    movies = load_data()
    # Передаем данные в шаблон
    return render_template('index.html', movies=movies)

# Маршрут для страницы рекомендаций
@app.route('/recommend')
def recommend_page():
    # Загружаем данные
    movies = load_data()
    # Передаем данные в шаблон
    return render_template('rec.html', movies=movies)

# Маршрут для получения рекомендаций
@app.route('/rec', methods=['POST'])
def recommend():
    # Получаем название фильма из формы
    movie_title = request.form['movie_title']
    
    # Поиск индекса фильма по названию
    movie_index = find_movie_index_by_title(df, movie_title)
    
    if movie_index is not None:
        # Получаем рекомендации
        recommendations = recommend_movies(movie_index, knn, df)
        # Преобразуем рекомендации в список словарей
        recommendations_list = recommendations[['title', 'averageRating', 'year']].to_dict('records')
        # Возвращаем рекомендации в формате JSON
        return jsonify(recommendations_list)
    else:
        return jsonify({"error": "Фильм не найден"}), 404

if __name__ == '__main__':
    app.run(debug=True)