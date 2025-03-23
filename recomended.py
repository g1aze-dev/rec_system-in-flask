import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

# Преобразование JSON в DataFrame
data = pd.read_json("data/movies01.json")
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

# Ввод названия фильма от пользователя
movie_title = input("Введите название фильма: ")

# Поиск индекса фильма по названию
movie_index = find_movie_index_by_title(df, movie_title)

if movie_index is not None:
    # Получение рекомендаций
    recommendations = recommend_movies(movie_index, knn, df)
    print("\nРекомендации для фильма:", df.iloc[movie_index]['title'])
    print(recommendations[['title', 'averageRating', 'year']])