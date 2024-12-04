import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Загрузка данных
df = pd.read_csv('amz_ca_total_products_data_processed.csv', nrows=10000)
# Инициализация корзины в session_state, если она еще не существует
if 'cart' not in st.session_state:
    st.session_state.cart = []

# Инициализация текущей страницы в session_state
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1

# Обработка пропусков в описаниях товаров (если они есть)
df['title'] = df['title'].fillna('')

# Инициализация TF-IDF векторайзера для названия товара
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Функция для получения рекомендаций на основе одного товара
def recommend_by_description(product_name, cosine_sim=cosine_sim):
    idx = df.index[df['title'] == product_name].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Отбираем 5 товаров
    product_indices = [i[0] for i in sim_scores]
    return df[['title', 'price', 'categoryName', 'productURL', 'imgUrl']].iloc[product_indices]

# Функция для отображения товаров с загрузкой по кнопке
def show_products():
    page_number = st.session_state.page_number
    page_size = 20
    start_idx = (page_number - 1) * page_size
    end_idx = min(page_number * page_size, len(df))  # Убедиться, что не выходит за пределы
    
    # Перемешиваем DataFrame случайным образом для загрузки товаров
    random_products = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Получаем нужную страницу товаров
    products_to_show = random_products.iloc[start_idx:end_idx]

    # Отображение товаров в виде карточек в несколько колонок
    num_columns = 5  # 5 колонок
    for i in range(0, len(products_to_show), num_columns):
        cols = st.columns(num_columns)
        for j, col in enumerate(cols):
            if i + j < len(products_to_show):
                product = products_to_show.iloc[i + j]
                with col:
                    # Отображаем карточку товара с фиксированным размером картинки через CSS
                    st.markdown(
                        f"""
                        <style>
                            .product-img-{i+j} {{
                                width: 200px;
                                height: 150px;
                                object-fit: cover;
                            }}
                        </style>
                        <img class="product-img-{i+j}" src="{product['imgUrl']}" />
                        """, unsafe_allow_html=True)
                    st.write(f"**{product['title'][:50]}...**")  # Обрезаем текст, если он слишком длинный
                    st.write(f"Цена: {product['price']} $")
                    st.write(f"Рейтинг: {product['stars']} звёзд ({product['reviews']} отзывов)")
                    if st.button(f"Добавить в корзину", key=f"add_to_cart_{product['asin']}"):
                        st.session_state.cart.append(product['title'])

    # Кнопка для загрузки дополнительных товаров
    if st.button('Загрузить еще товары', key=f"load_more_{page_number}"):
        st.session_state.page_number += 1
        show_products()

# Функция для отображения товаров в корзине
def show_cart():
    if not st.session_state.cart:
        st.write("Корзина пуста")
        return  # Если корзина пуста, прекращаем выполнение функции
    
    st.subheader("Корзина")
    st.write("Ваши товары в корзине:")
    
    # Отображаем товары в корзине в виде карточек
    num_columns = 5  # 5 колонок
    for i in range(0, len(st.session_state.cart), num_columns):
        cols = st.columns(num_columns)
        for j, col in enumerate(cols):
            if i + j < len(st.session_state.cart):
                product_title = st.session_state.cart[i + j]
                product = df[df['title'] == product_title].iloc[0]
                with col:
                    # Отображаем карточку товара из корзины с фиксированным размером картинки через CSS
                    st.markdown(
                        f"""
                        <style>
                            .product-img-{i+j} {{
                                width: 200px;
                                height: 150px;
                                object-fit: cover;
                            }}
                        </style>
                        <img class="product-img-{i+j}" src="{product['imgUrl']}" />
                        """, unsafe_allow_html=True)
                    st.write(f"**{product['title'][:50]}...**")
                    st.write(f"Цена: {product['price']} ₽")
                    st.write(f"Рейтинг: {product['stars']} звёзд ({product['reviews']} отзывов)")

    # Рекомендации на основе всех товаров в корзине
    st.subheader("Рекомендуемые товары по описанию")
    
    all_recommendations = []
    
    # Генерируем рекомендации для каждого товара в корзине
    for product_title in st.session_state.cart:
        recommendations = recommend_by_description(product_title)
        if not recommendations.empty:
            all_recommendations.append(recommendations)
    
    # Если есть рекомендации, объединяем их и случайным образом перемешиваем
    if all_recommendations:
        all_recommendations_df = pd.concat(all_recommendations).drop_duplicates()
        all_recommendations_df = all_recommendations_df.sample(frac=1).reset_index(drop=True)
        
        # Показываем рекомендованные товары по 5 штук в 5 колонках
        num_columns = 5
        for i in range(0, len(all_recommendations_df), num_columns):
            cols = st.columns(num_columns)
            for j, col in enumerate(cols):
                if i + j < len(all_recommendations_df):
                    rec = all_recommendations_df.iloc[i + j]
                    with col:
                        # Отображаем карточку рекомендованного товара с фиксированным размером картинки
                        st.markdown(
                            f"""
                            <style>
                                .product-img-rec-{i+j} {{
                                    width: 200px;
                                    height: 150px;
                                    object-fit: cover;
                                }}
                            </style>
                            <img class="product-img-rec-{i+j}" src="{rec['imgUrl']}" />
                            """, unsafe_allow_html=True)
                        st.write(f"**{rec['title']}**")
                        st.write(f"Цена: {rec['price']} ₽")
                        if st.button(f"Добавить в корзину", key=f"add_to_cart_rec_{rec['imgUrl']}"):
                            st.session_state.cart.append(rec['title'])
    else:
        st.write("Нет рекомендаций для ваших товаров.")

# Главная страница с товарами
st.title("DMart - Рекомендации товаров")

# Вкладки для переключения между страницами
page = st.radio("Выберите страницу", ["Товары", "Корзина"])

if page == "Товары":
    show_products()
elif page == "Корзина":
    show_cart()
