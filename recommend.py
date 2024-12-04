import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных
df = pd.read_csv('amz_ca_total_products_data_processed.csv')

# Обработка пропусков в описаниях товаров (если они есть)
df['title'] = df['title'].fillna('')

# Инициализация TF-IDF векторайзера для названия товара
vectorizer = TfidfVectorizer()

# Преобразуем название товаров в векторное представление
tfidf_matrix = vectorizer.fit_transform(df['title'])

# Рассчитываем косинусное сходство между товарами
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Функция для получения рекомендаций на основе названия товара
def recommend_by_description(product_name, cosine_sim=cosine_sim):
    # Находим индекс товара по названию
    idx = df.index[df['title'] == product_name].tolist()[0]

    # Получаем оценки схожести товаров
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Сортируем товары по схожести
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Получаем индексы наиболее похожих товаров (исключаем сам товар)
    sim_scores = sim_scores[1:6]  # Топ 5 похожих товаров
    product_indices = [i[0] for i in sim_scores]

    # Возвращаем рекомендации
    return df[['title', 'price', 'categoryName', 'productURL']].iloc[product_indices]

# Функция для получения рекомендаций на основе категории товара
def recommend_by_category(product_name):
    # Находим товар по названию
    product = df[df['title'] == product_name].iloc[0]
    category = product['categoryName']
    
    # Рекомендуем товары той же категории
    recommended = df[df['categoryName'] == category]
    
    # Возвращаем топ-5 товаров в той же категории
    return recommended[['title', 'price', 'categoryName', 'productURL']].head(5)
