import pandas as pd
import numpy as np
import streamlit as st

from etna.datasets import TSDataset
from etna.models import (CatBoostPerSegmentModel,
                         CatBoostMultiSegmentModel)

from etna.transforms import (
    LogTransform,
    TimeSeriesImputerTransform,
    LinearTrendTransform,
    LagTransform,
    DateFlagsTransform,
    FourierTransform,
    SegmentEncoderTransform,
    MeanTransform
)

from etna.metrics import SMAPE
from etna.pipeline import Pipeline
from etna.analysis import (plot_forecast,
                           plot_backtest)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Прогнозирование потребления электроэнергии по 4 сегментам')
st.header('Исходные данные')
st.write(
    'Данные длжны быть в расплавленном, длинном виде.'
    'Столбец с метками времени должен называться **timestamp**.'
    'Столбец с сегментами должен называться **segment**.'
    'Столбец с зависимой переменной должен называться **target**.'
)

data_file = st.file_uploader('Загрузите ваш CSV-файл')
if data_file is not None:
    data_df = pd.read_csv(data_file)
else:
    st.stop()

st.dataframe(data_df.head(10))

df = TSDataset.to_dataset(data_df)
ts = TSDataset(df, freq='D')

st.header('Визуализация рядов')

visualize = st.radio('Визуализировать ряды?', ('Нет', 'Да'))
if visualize == 'Нет':
    pass
else:
    st.pyplot(ts.plot())

st.header('Горизонт прогнозирования')

HORIZON = st.number_input('Задайте горизонт прогнозирования', min_value=1, value=14)

train_ts, test_ts = ts.train_test_split(test_size=HORIZON)

st.write(f'временные рамки обучающей выборки: '
         f'{train_ts.index[0].strftime("%Y-%m-%d")} - '
         f'{train_ts.index[-1].strftime("%Y-%m-%d")}')
st.write(f'временные рамки тестовой выборки: '
         f'{test_ts.index[0].strftime("%Y-%m-%d")} - '
         f'{test_ts.index[-1].strftime("%Y-%m-%d")}')

st.header('Преобразования зависимой переменной')

tf_classes_options = ['LogTransform', 'TimeSeriesImputerTransform', 'LinearTrendTransform']

tf_classes_lst = st.sidebar.multiselect('Список классов, создающих преобразования', tf_classes_options)

tf_classes_dict = {}

if 'LogTransform' in tf_classes_lst:
    log = LogTransform(in_column="target")

    tf_classes_dict.update({'LogTransform': log})

if 'TimeSeriesImputerTransform' in tf_classes_lst:
    imputer_title = (
        '<p style="font-family:Arial; color:Black; font-size: 18px;"' +
        '>Выберите настройки для TimeSeriesImputerTransform</p>')
    st.markdown(imputer_title, unsafe_allow_html=True)

    strategy = st.selectbox(
        'Стратегия импутации пропусков',
        ['constant', 'mean', 'running_mean', 'seasonal', 'forward_fill'])

    window = st.number_input('Введите ширину скользящего окна', min_value=-1)
    seasonality = st.number_input('Введите длинну сезонности', min_value=1)

    imputer = TimeSeriesImputerTransform(
        in_column='target',
        strategy = strategy,
        window=window,
        seasonality=seasonality)

    tf_classes_dict.update({'TimeSeriesImputerTransform': imputer})

if 'LinearTrendTransform' in tf_classes_lst:
    detrend = LinearTrendTransform(in_column='target')

    tf_classes_dict.update({'LinearTrendTransform': detrend})

final_tf_classes_lst = list(tf_classes_dict.values())

st.header('Конструирование признаков')

fe_classes_options = ['LagTransform', 'MeanTransform', 'DateFlagsTransform', 'SegmentEncoderTransform']
fe_classes_lst = st.sidebar.multiselect(
    'Список классов, создающих признаки', fe_classes_options)

fe_classes_dict = {}

if 'LagTransform' in fe_classes_lst:
    lags_title = (
        '<p style="font-family:Arial; color:Black; font-size: 18px;"' +
        '>Выберите настройки для LagTransform</p>')
    st.markdown(lags_title, unsafe_allow_html=True)

    lower_limit = st.number_input('Нижняя граница порядка лага',
                                  min_value=HORIZON)
    upper_limit = st.number_input('Верхняя граница порядка лага',
                                  min_value=2 * HORIZON)
    increment = st.number_input('Шаг прироста порядка лага',
                                min_value=int(np.sqrt(HORIZON)))
    lags = LagTransform(in_column='target',
                        lags = list(range(lower_limit, upper_limit, increment)),
                        out_column='target_lag')
    fe_classes_dict.update({'LagTransform': lags})

if 'MeanTransform' in fe_classes_lst:
    means_title = (
            '<p style="font-family:Arial; color:Black; font-size: 18px;"' +
            '>Выберите настройки для MeanTransform</p>')
    st.markdown(means_title, unsafe_allow_html=True)
    means_number = st.number_input('Введите количество окон', min_value=1)
    numbers = [st.slider(f'Введите ширину {i + 1}-го окна',
                         min_value=HORIZON,
                         max_value=3*HORIZON)
               for i in range(means_number)]
    for number in numbers:
        fe_classes_dict.update({f'MeanTransform{number}': MeanTransform(
            in_column='target',
            window=number,
            out_column=f'target_mean{number}')})

if 'DateFlagsTransform' in fe_classes_lst:
    dateflags_title = (
            '<p style="font-family:Arial; color:Black; font-size: 18px;"' +
            '>Выберите настройки для DateFlagsTransform</p>')
    st.markdown(dateflags_title, unsafe_allow_html=True)

    day_number_in_week = st.checkbox(
        'Порядковый номер дня в неделе', value=False)
    day_number_in_month = st.checkbox(
        'Порядковый номер дня в месяце', value=True)
    week_number_in_month = st.checkbox(
        'Порядковый номер недели в месяце', value=False)
    month_number_in_year = st.checkbox(
        'Порядковый номер месяца в году', value=True)
    season_number = st.checkbox(
        'Порядковый номер сезона в году', value=False)
    is_weekend = st.checkbox(
        'индикатор выходного дня', value=False)

    dateflags = DateFlagsTransform(
        day_number_in_week=day_number_in_week,
        day_number_in_month=day_number_in_month,
        week_number_in_month=week_number_in_month,
        month_number_in_year=month_number_in_year,
        season_number=season_number,
        is_weekend=is_weekend,
        out_column='date_flag')

    fe_classes_dict.update({'DateFlagsTransform': dateflags})

if 'SegmentEncoderTransform' in fe_classes_lst:
    seg = SegmentEncoderTransform()
    fe_classes_dict.update({'SegmentEncoderTransform': seg})

final_fe_classes_lst = list(fe_classes_dict.values())

final_classes_lst = final_tf_classes_lst + final_fe_classes_lst

default_lags = LagTransform(
    in_column='target',
    lags = list(range(HORIZON, 3 * HORIZON, HORIZON)),
    out_column='target_lag')

default_dateflags = DateFlagsTransform(
    day_number_in_week=True,
    week_number_in_month=True,
    month_number_in_year=True,
    out_column='date_flag')

default_classes_lst = [default_lags, default_dateflags]

if len(final_fe_classes_lst) == 0:
    transforms = default_classes_lst
else:
    transforms = final_classes_lst

st.header('Обучение базовой модели Catboost')

iterations = st.number_input(
    'Введите количество деревьев',
    min_value=1, max_value=2000, value=1000)
learning_rate = st.number_input(
    'Введите темп обучения',
    min_value=0.001, max_value=1.0, value=0.03)
depth = st.number_input(
    'Введите максимальную глубину деревьев',
    min_value=1, max_value=16, value=6)

catboost_model_type = st.sidebar.selectbox(
    'Какую модель CatBoost обучить?',
    ['PerSegment', 'MultiSegment'])

if catboost_model_type == 'PerSegment':
    model = CatBoostPerSegmentModel(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth)
else:
    model=CatBoostMultiSegmentModel(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth)

run_model = st.radio(
    'Обучить базовую модель?',
    ('Нет', 'Да'))

if run_model == 'Нет':
    st.stop()
else:
    pass

pipeline = Pipeline(model=model,
                    transforms=transforms,
                    horizon=HORIZON)
pipeline.fit(train_ts)
forecast_ts = pipeline.forecast()

smape = SMAPE()
smape_values = smape(y_true=test_ts,
                     y_pred=forecast_ts)

smape_values = pd.DataFrame({'Прогнозы': smape_values})
smape_mean = smape_values['Прогнозы'].mean()

st.header('Оценка качества прогнозов базовой модели - SMAPE')

st.write(smape_values)
st.write('Среднее значение:', smape_mean)

st.header('Визуализация прогнозов базовой модели')

n_train_samples = st.slider(
    'N последних наблюдений в обучающей выборке',
    min_value=3 * HORIZON)

st.pyplot(
    plot_forecast(
        forecast_ts=forecast_ts,
        test_ts=test_ts,
        train_ts=train_ts,
        n_train_samples=n_train_samples
    )
)

st.header('Перекрестная проверка модели Catboost')

mode = st.selectbox(
    'Стратегия перекрестной проверки',
    ['expand', 'constant'])

n_folds = st.number_input(
    'Введите количество блоков перекрестной проверки',
    min_value=1, max_value=24, value=3)

run_cv = st.radio(
    'Запускать перекрестную проверку?',
    ('Нет', 'Да'))

if run_cv == 'Нет':
    st.stop()
else:
    pass

metrics_cv, forecast_cv, _ = pipeline.backtest(
    mode=mode,
    n_folds=n_folds,
    ts=ts,
    metrics=[smape],
    aggregate_metrics=True)

cv_mean_smape = metrics_cv['SMAPE'].mean()

st.header('Оценка качества прогнозов по итогам '
          'перекрестной проверки - SMAPE')

st.write(metrics_cv)
st.write('Среднее значение:', cv_mean_smape)

st.header('Визуализация прогнозов по итогам перекрестной проверки')

st.pyplot(
    plot_backtest(forecast_cv, ts, history_len=0)
)
















