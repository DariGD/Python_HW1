import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import aiohttp
import asyncio

def analyze_city_season(df, city_name):
    city_data = df[df['city'] == city_name]
    city_data = city_data.dropna(subset=['temperature', 'timestamp'])

    # Скользящее среднее и стандартное отклонение (30 дней) по сезону
    city_data['season'] = city_data['season'].astype(str)
    city_data['rolling_mean'] = city_data.groupby('season')['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    city_data['rolling_std'] = city_data.groupby('season')['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std()
    )


    city_data['anomaly'] = (
        (city_data['temperature'] > (city_data['rolling_mean'] + 2 * city_data['rolling_std'])) |
        (city_data['temperature'] < (city_data['rolling_mean'] - 2 * city_data['rolling_std']))
    )

    seasonal_profile = city_data.groupby('season').agg(
        avg_temp=('temperature', 'mean'),
        std_temp=('temperature', 'std')
    ).reset_index()

 
    rolling_stats = city_data.groupby('season').agg(
        rolling_mean=('rolling_mean', 'mean'),
        rolling_std=('rolling_std', 'mean')
    ).reset_index()


    city_data['timestamp'] = pd.to_datetime(city_data['timestamp'])
    city_data['days_since_start'] = (city_data['timestamp'] - city_data['timestamp'].min()).dt.days
    city_data = city_data.dropna(subset=['days_since_start'])
    X = city_data['days_since_start'].values.reshape(-1, 1)  # Количество дней с начала периода
    y = city_data['temperature'].values
    model = LinearRegression()
    model.fit(X, y)
    trend_slope = model.coef_[0]

    # 5. Среднее, min, max температуры за всё время
    avg_temp_all = city_data['temperature'].mean()
    min_temp_all = city_data['temperature'].min()
    max_temp_all = city_data['temperature'].max()

    # Собираем результаты
    result = {
        'city': city_name,
        'avg_temp_all': avg_temp_all,
        'min_temp_all': min_temp_all,
        'max_temp_all': max_temp_all,
        'seasonal_profile': seasonal_profile,
        'rolling_stats': rolling_stats,
        'trend_slope': trend_slope,
        'anomalies': city_data[city_data['anomaly'] == True][['temperature', 'anomaly']].to_dict(orient='records')
    }

    return result

def merge_with_historical_data(df, historical_data):
    merged_data = df.copy()

    # Добавляем данные из словаря
    for city_name, city_data in historical_data.items():
        seasonal_data = city_data['seasonal_profile']

        for season in seasonal_data['season']:
            season_data = seasonal_data[seasonal_data['season'] == season]
            avg_temp_season = season_data['avg_temp'].values[0]
            std_temp_season = season_data['std_temp'].values[0]

            merged_data.loc[(merged_data['city'] == city_name) & (merged_data['season'] == season),
                            'avg_temp_for_season'] = avg_temp_season
            merged_data.loc[(merged_data['city'] == city_name) & (merged_data['season'] == season),
                            'std_temp_for_season'] = std_temp_season


        merged_data.loc[merged_data['city'] == city_name, 'avg_temp_all'] = city_data['avg_temp_all']
        merged_data.loc[merged_data['city'] == city_name, 'min_temp_all'] = city_data['min_temp_all']
        merged_data.loc[merged_data['city'] == city_name, 'max_temp_all'] = city_data['max_temp_all']

    merged_data['rolling_mean'] = merged_data.groupby(['city', 'season'])['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    merged_data['rolling_std'] = merged_data.groupby(['city', 'season'])['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std()
    )

    merged_data['anomalies'] = merged_data.apply(
        lambda row: 'да' if abs(row['temperature'] - row['rolling_mean']) > 2 * row['rolling_std'] else 'нет',
        axis=1
    )

    return merged_data

st.title("Анализ температурных данных")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл с данными (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Загруженные данные:")
    st.dataframe(data.head())

    required_columns = {'city', 'timestamp', 'temperature', 'season'}
    if not required_columns.issubset(data.columns):
        st.error(f"Файл должен содержать столбцы: {', '.join(required_columns)}")
    else:
        cities = data['city'].unique()
        analysis_results = {city: analyze_city_season(data, city) for city in cities}
        st.session_state_analysis_results = analysis_results
        merged_df = merge_with_historical_data(data, analysis_results)
        
        st.write("Результирующий датафрейм с анализом:")
        st.dataframe(merged_df)
else:
    st.info("Загрузите файл для анализа.")

@st.cache_data
def plot_city_graphs(df, selected_cities):
    for city_name in selected_cities:
        city_data = df[df['city'] == city_name]

        plt.figure(figsize=(20, 10), facecolor='white')
 
        plt.plot(city_data['timestamp'], city_data['temperature'], label='Temperature', color='tab:blue', linewidth=0.6)

        plt.plot(city_data['timestamp'], city_data['avg_temp_for_season'], label='Seasonal Avg Temp', color='tab:red', linestyle='--', linewidth=2)

        plt.fill_between(city_data['timestamp'],
                         city_data['avg_temp_for_season'] - city_data['std_temp_for_season'],
                         city_data['avg_temp_for_season'] + city_data['std_temp_for_season'],
                         color='tab:orange', alpha=0.4, label='Seasonal Temp Range')

        anomalies = city_data[city_data['anomalies'] == 'да']
        plt.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', label='Anomalies', zorder=5)

        plt.title(f"Temperature Data for {city_name}")
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(False)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.gca().xaxis.set_major_locator(mdates.YearLocator()) 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  

        st.pyplot(plt)
        plt.close()

# Функция для построения сезонного графика
@st.cache_data
def plot_city_temperature(city_data, city_name):
    city = city_data[city_name]

    avg_temp_all = city['avg_temp_all']
    min_temp_all = city['min_temp_all']
    max_temp_all = city['max_temp_all']

    seasonal_data = city['seasonal_profile']

    fig, ax = plt.subplots(figsize=(10, 6))

 
    ax.plot(seasonal_data['season'], seasonal_data['avg_temp'], marker='o', label='Средняя температура сезона', color='tab:blue')
    ax.fill_between(seasonal_data['season'], seasonal_data['avg_temp'] - seasonal_data['std_temp'],
                    seasonal_data['avg_temp'] + seasonal_data['std_temp'], alpha=0.2, color='tab:blue', label='Стандартное отклонение')

    ax.axhline(avg_temp_all, color='gray', linestyle='--', label='Средняя температура за год')
    ax.axhline(min_temp_all, color='blue', linestyle=':', label='Минимальная температура за год')
    ax.axhline(max_temp_all, color='red', linestyle=':', label='Максимальная температура за год')


    ax.set_title(f'Температура для {city_name}')
    ax.set_xlabel('Сезоны')
    ax.set_ylabel('Температура (°C)')
    ax.legend()

    st.pyplot(fig)
    plt.close()


if uploaded_file:
    # После анализа данных
    st.subheader("Визуализация температурных данных")

    # Выбор городов
    all_cities = merged_df['city'].unique()
    selected_cities = st.multiselect("Выберите города для визуализации временных рядов:", all_cities, default=all_cities[:1])

    if selected_cities:
        st.write("Графики временных рядов для выбранных городов:")
        plot_city_graphs(merged_df, selected_cities)
    else:
        st.info("Выберите хотя бы один город для визуализации.")

    # Выбор города для сезонных данных
    selected_city_temp = st.selectbox("Выберите город для визуализации сезонных данных:", all_cities)

    if selected_city_temp:
        st.write(f"Сезонный график для города {selected_city_temp}:")
        plot_city_temperature(analysis_results, selected_city_temp)


st.title("Анализ температурных данных с использованием API OpenWeather")
api_key = st.text_input("Введите ваш API-ключ OpenWeather", type="password")

# Список доступных городов
cities = ['New York', 'London', 'Paris', 'Tokyo', 'Moscow', 'Sydney',
          'Berlin', 'Beijing', 'Rio de Janeiro', 'Dubai', 'Los Angeles',
          'Singapore', 'Mumbai', 'Cairo', 'Mexico City']

@st.cache_data
def get_url(city, api_key):
    return f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

async def fetch_weather(city, api_key):
    async with aiohttp.ClientSession() as session:
        url = get_url(city, api_key)
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data['main']['temp']
            else:
                st.error(f"Ошибка при получении данных для города {city}: {response.status}")
                return None

# Асинхронная функция для обработки одного города (вообще можно и без асинхронности тут)
async def get_city_temperature(city, api_key):
    temperature = await fetch_weather(city, api_key)
    if temperature is not None:
        st.success(f"Текущая температура в городе {city}: {temperature}°C")
    return temperature

@st.cache_data
def compare_city_temperature(city_name, current_temp, historical_data, current_season):

    if city_name not in historical_data:
        return f"Город {city_name} не найден в исторических данных."

    city_result = historical_data[city_name]

    max_temp = city_result['max_temp_all']
    min_temp = city_result['min_temp_all']
    avg_temp = city_result['avg_temp_all']

    seasonal_data = city_result['seasonal_profile']
    season_avg_temp = None
    season_std_temp = None


    if seasonal_data is not None:
        season_data = seasonal_data[seasonal_data['season'] == current_season]
        if not season_data.empty:
            season_avg_temp = season_data['avg_temp'].values[0]
            season_std_temp = season_data['std_temp'].values[0]


    comparison = {
        'city': city_name,
        'current_temp': current_temp,
        'avg_temp': avg_temp,
        'max_temp': max_temp,
        'min_temp': min_temp,
        'season_avg_temp': season_avg_temp,
        'season_std_temp': season_std_temp,
        'current_season': current_season,
        'comparison': {
            'above_max': current_temp > max_temp,
            'below_min': current_temp < min_temp,
            'above_avg': current_temp > avg_temp,
            'below_avg': current_temp < avg_temp,
            'above_season_avg': current_temp > season_avg_temp if season_avg_temp is not None else None,
            'below_season_avg': current_temp < season_avg_temp if season_avg_temp is not None else None,
        },
    }
    comparison_df = pd.DataFrame([{
        'Параметр': 'Текущая температура',
        'Значение': f"{current_temp}°C"
    }, {
        'Параметр': 'Средняя температура за год',
        'Значение': f"{avg_temp}°C"
    }, {
        'Параметр': 'Максимальная температура за год',
        'Значение': f"{max_temp}°C"
    }, {
        'Параметр': 'Минимальная температура за год',
        'Значение': f"{min_temp}°C"
    }, {
        'Параметр': 'Средняя температура текущего сезона',
        'Значение': f"{season_avg_temp}°C" if season_avg_temp is not None else 'Нет данных'
    }, {
        'Параметр': 'Стандартное отклонение текущего сезона',
        'Значение': f"{season_std_temp}°C" if season_std_temp is not None else 'Нет данных'
    }, {
        'Параметр': 'Текущий сезон',
        'Значение': current_season
    }, {
        'Параметр': 'Сравнение с максимальной температурой за год',
        'Значение': 'Да' if comparison['comparison']['above_max'] else 'Нет'
    }, {
        'Параметр': 'Сравнение с минимальной температурой за год',
        'Значение': 'Да' if comparison['comparison']['below_min'] else 'Нет'
    }, {
        'Параметр': 'Выше средней температуры за год',
        'Значение': 'Да' if comparison['comparison']['above_avg'] else 'Нет'
    }, {
        'Параметр': 'Ниже средней температуры за год',
        'Значение': 'Да' if comparison['comparison']['below_avg'] else 'Нет'
    }, {
        'Параметр': 'Выше средней температуры сезона',
        'Значение': 'Да' if comparison['comparison']['above_season_avg'] else 'Нет'
    }, {
        'Параметр': 'Ниже средней температуры сезона',
        'Значение': 'Да' if comparison['comparison']['below_season_avg'] else 'Нет'
    }])


    def style_negative(val):
        color = 'red' if val == 'Да' else 'green'  # Красный если "Да" (выше, ниже), зеленый если "Нет"
        return f'color: {color}'

    st.write("Результаты сравнения температуры города с историческими данными:")
    st.table(comparison_df.style.applymap(style_negative, subset=['Значение']))



# Выпадающий список для выбора города и сезона
selected_city = st.selectbox("Выберите город для получения температуры:", cities)
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
current_season = st.selectbox("Выберите текущий сезон:", seasons)

# Кнопка для запроса данных
if st.button("Получить температуру"):
    if not api_key:
        st.error("Пожалуйста, введите API-ключ.")
    else:
        
        current_temperature = asyncio.run(get_city_temperature(selected_city, api_key))
        st.session_state.current_temperature = current_temperature

if st.button("Сравнить с историческими данными"):
    if 'current_temperature' in st.session_state:
        comparison_result = compare_city_temperature(
                    selected_city, st.session_state.current_temperature, st.session_state_analysis_results, current_season
                )
