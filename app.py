import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tensorflow.keras.models import load_model, Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
import pickle
from math import sqrt
import seaborn as sns
from scipy.stats import pearsonr

st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.simplefilter('ignore', ConvergenceWarning)

# Define your functions here

def corr_with_pvalues(df):
    df = df.dropna()._get_numeric_data()
    cols = df.columns
    pval_matrix = pd.DataFrame(index=cols, columns=cols)

    for i in range(len(cols)):
        for j in range(i, len(cols)):
            if i == j:
                pval_matrix.iat[i, j] = 0
            else:
                corr, pval = pearsonr(df[cols[i]], df[cols[j]])
                pval_matrix.iat[i, j] = pval_matrix.iat[j, i] = pval

    return pval_matrix.astype(float)



def filter_country_data(data, country):
    country_data = data[data['location'] == country]
    features = ['new_cases', 'new_deaths', 'new_vaccinations']
    country_data = country_data[features].dropna()
    return country_data

def train_linear_regression(country_data):
    X = country_data[['new_deaths', 'new_vaccinations']]
    y = country_data['new_cases']

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaled_x = scaler_x.fit_transform(X)
    scaled_y = scaler_y.fit_transform(y.values.reshape(-1, 1))

    model_lr = LinearRegression()
    model_lr.fit(scaled_x, scaled_y)

    return model_lr, scaler_x, scaler_y

def predict_new_cases(model_lr, scaler_x, scaler_y, new_deaths, new_vaccinations):
    input_data = np.array([[new_deaths, new_vaccinations]])
    scaled_input_data = scaler_x.transform(input_data)
    scaled_prediction = model_lr.predict(scaled_input_data)
    prediction = scaler_y.inverse_transform(scaled_prediction).flatten()[0]
    return prediction

@st.cache_data
def load_data(csv_file_path):
    data_cov = pd.read_csv(csv_file_path)
    data_cov.index = data_cov['date']
    data_cov = data_cov[data_cov['location'] == "India"]
    features = ['new_cases', 'new_cases_smoothed', 'new_deaths', 'new_vaccinations']
    data_cov = data_cov[features]
    return data_cov

@st.cache_data
def load_data2(csv_file_path):
    data_cov = pd.read_csv(csv_file_path)
    return data_cov


@st.cache_data
def create_lagged_dataset(data_cov, n_lags=14):
    dataset = series_to_supervised(data_cov, n_in=n_lags)
    return dataset

@st.cache_data
def train_models(target_col, dataset):
    X = dataset.drop(columns=[target_col])
    Y = dataset[target_col]

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaled_x = scaler_x.fit_transform(X)
    scaled_y = scaler_y.fit_transform(Y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(scaled_x, scaled_y, test_size=0.3, shuffle=False)

    y_train_inv = scaler_y.inverse_transform(y_train).flatten()
    y_test_inv = scaler_y.inverse_transform(y_test).flatten()

    # Linear Regression Model
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    y_pred_test_lr = model_lr.predict(X_test)
    y_pred_test_lr_inv = scaler_y.inverse_transform(y_pred_test_lr).flatten()

    lr_metrics = {
        'mae': mean_absolute_error(y_test_inv, y_pred_test_lr_inv),
        'mse': mean_squared_error(y_test_inv, y_pred_test_lr_inv),
        'rmse': sqrt(mean_squared_error(y_test_inv, y_pred_test_lr_inv)),
        'r2': r2_score(y_test_inv, y_pred_test_lr_inv)
    }

    # ARIMA Model
    y_train_arima = y_train.flatten()
    y_test_arima = y_test.flatten()

    auto_arima_model = auto_arima(y_train, trace=True, suppress_warnings=True)
    arima_model_313 = ARIMA(y_train_arima, order=(0, 1, 1)).fit()

    history = [x for x in y_train_arima]
    predictions_arima = []
    for t in range(len(y_test_arima)):
        model = ARIMA(history, order=(3, 1, 3))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions_arima.append(yhat)
        history.append(y_test_arima[t])

    predictions_arima = np.array(predictions_arima).reshape(-1, 1)
    predictions_arima_inv = scaler_y.inverse_transform(predictions_arima).flatten()

    arima_metrics = {
        'mae': mean_absolute_error(y_test_inv, predictions_arima_inv),
        'mse': mean_squared_error(y_test_inv, predictions_arima_inv),
        'rmse': sqrt(mean_squared_error(y_test_inv, predictions_arima_inv)),
        'r2': r2_score(y_test_inv, predictions_arima_inv)
    }

    # ANN Model
    epochs = 500
    batch_size = int(y_train.shape[0] * 0.1)
    estimator = KerasRegressor(epochs=epochs, batch_size=batch_size, verbose=1)

    model = Sequential()
    model.add(Dense(50, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    estimator.model = model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = estimator.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stopping])
    estimator.model_.save('BP_saved_model.h5')
    with open('history.pickle', 'wb') as f:
        pickle.dump(history.history_, f)

    estimator.model_ = load_model('BP_saved_model.h5')
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)

    res_ts = estimator.predict(X_test)
    res_test_ANN = scaler_y.inverse_transform(res_ts.reshape(-1, 1)).flatten()

    ann_metrics = {
        'mae': mean_absolute_error(y_test_inv, res_test_ANN),
        'mse': mean_squared_error(y_test_inv, res_test_ANN),
        'rmse': sqrt(mean_squared_error(y_test_inv, res_test_ANN)),
        'r2': r2_score(y_test_inv, res_test_ANN)
    }

    # LSTM Model
    train_x_LSTM = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    test_x_LSTM = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    batch_size=int(y_train.shape[0]*.1)
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(7, input_shape=(train_x_LSTM.shape[1], train_x_LSTM.shape[2])))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(Dense(y_train.shape[1]))
    LSTM_model.compile(loss='mse', optimizer='adam')

    fitting = True
    fitting_save = True
    epochs = 420

    if fitting:
        history = LSTM_model.fit(train_x_LSTM, y_train, epochs=epochs, batch_size=batch_size, validation_data=(test_x_LSTM, y_test), verbose=1, shuffle=False)
        if fitting_save:
            # Serialize model to JSON
            model_json = model.to_json()
            with open("LSTM_model.json", "w") as json_file:
                json_file.write(model_json)

            # Save weights to HDF5
            model.save_weights("LSTM_model.weights.h5")
            with open('history_LSTM.pickle', 'wb') as f:
                pickle.dump(history.history, f)

    json_file = open('LSTM_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("LSTM_model.weights.h5")
    with open('history_LSTM.pickle', 'rb') as f:
        history = pickle.load(f)

    lstm_test_predict = LSTM_model.predict(test_x_LSTM)
    lstm_test_predict_inv = scaler_y.inverse_transform(lstm_test_predict).flatten()

    lstm_metrics = {
        'mae': mean_absolute_error(y_test_inv, lstm_test_predict_inv),
        'mse': mean_squared_error(y_test_inv, lstm_test_predict_inv),
        'rmse': sqrt(mean_squared_error(y_test_inv, lstm_test_predict_inv)),
        'r2': r2_score(y_test_inv, lstm_test_predict_inv)
    }

    results = {
        'linear_regression': lr_metrics,
        'arima': arima_metrics,
        'ann': ann_metrics,
        'lstm': lstm_metrics,
        'predictions': {
            'linear': y_pred_test_lr_inv,
            'arima': predictions_arima_inv,
            'ann': res_test_ANN,
            'lstm': lstm_test_predict_inv,
            'actual': y_test_inv
        }
    }

    return results


def series_to_supervised(df, n_in=1, dropnan=True):
    n_vars = df.shape[1]
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    cols.append(df)
    names += [('%s(t)' % (df.columns[j])) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)

    return agg


# Function to set the sidebar with icons
def sidebar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["🏡 Home", "📈 Analysis", "📊 New Cases Modelling (India)", "💀 New Deaths Modelling (India)", "🔮 Prediction (Linear regression)"],
            icons=["house", "magic", "bar-chart", "bar-chart", "bar-chart"],
            menu_icon="cast",
            default_index=0,
        )
    return selected

# Function to create page layout
def page_layout(title, subtitle, content):
    st.markdown(
        f"""
        <style>
        .page-title {{
            font-size: 32px;
            font-weight: bold;
            padding: 20px 0;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
        }}
        .subtitle {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .content {{
            text-align: left;
            line-height: 1.6;
        }}
        </style>
        <div class="page-title">{title}</div>
        <div class="subtitle">{subtitle}</div>
        <div class="content">{content}</div>
        """,
        unsafe_allow_html=True,
    )

# Main app function
def main():
    st.set_page_config(page_title="COVID-19 Case Prediction", page_icon="🏡")
    selected = sidebar()

    # Home Page
    if selected == "🏡 Home":
        title = "🏡 Home"
        subtitle = "Project Background"
        content = """
        This project aims to predict COVID-19 cases using various machine learning models.
        We will explore different modeling techniques such as Linear Regression, ARIMA, 
        Auto ARIMA, Backpropagation Neural Networks (ANN), and Long Short-Term Memory (LSTM) networks.

        <div class="subtitle">Project Objectives</div>
        - Predict COVID-19 cases using different models.<br>
        - Compare the performance of these models.<br>
        - Visualize the predictions and evaluate their accuracy.
        """
        page_layout(title, subtitle, content)

    if selected == "📈 Analysis":
        title = "Analysis"
        subtitle = "Correlation and P-Values Matrix"
        content = """
        This section provides an analysis of the correlation between different COVID-19 metrics.
        Notable correlations include new_cases, new_deaths, new_cases_smoothed, and new_deaths_smoothed.
        The correlation involving new_vaccinations was experimental and its significance is quite low.
        """
        st.title(title)
        st.subheader(subtitle)
        st.write(content)
        csv_file_path = "st.csv"
        data_cov = load_data2(csv_file_path)
        covid_data = data_cov.copy()

        correlation_matrix = covid_data.corr(numeric_only=True)
        pval_matrix = corr_with_pvalues(covid_data)
        pval_matrix_formatted = pval_matrix.applymap(lambda x: f'{x:.4f}')

        st.markdown("### Correlation Matrix Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix Heatmap', fontsize=16)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        st.pyplot()

        st.markdown("### P-Values Matrix")
        st.dataframe(pval_matrix_formatted)

    elif selected == "📊 New Cases Modelling (India)":
        title = "New_cases models"
        subtitle = "Model Comparisons"

        csv_file_path = "st.csv"
        data_cov = pd.read_csv(csv_file_path)
        data_cov.index = data_cov['date']
        data_cov = data_cov[data_cov['location'] == "India"]

        features = ['new_cases', 'new_cases_smoothed', 'new_deaths', 'new_vaccinations']
        data_cov = data_cov[features]

        # Create lagged dataset
        n_lags = 14
        dataset = series_to_supervised(data_cov, n_in=n_lags)
        target_col = 'new_cases(t)'

        try:
            with st.spinner('Training models...'):
                results = train_models(target_col, dataset)

            # Linear Regression Model Evaluation
            st.markdown("### Linear Regression Model Evaluation")
            st.write("MAE (Mean Absolute Error):", results['linear_regression']['mae'])
            st.write("MSE (Mean Squared Error):", results['linear_regression']['mse'])
            st.write("RMSE (Root Mean Squared Error):", results['linear_regression']['rmse'])
            st.write("R2 Score:", results['linear_regression']['r2'])

            # Plot Linear Regression Prediction
            st.markdown("### Linear Regression Model Prediction")
            plt.figure(figsize=(14, 8))
            plt.plot(results['predictions']['actual'], label='Actual test')
            plt.plot(results['predictions']['linear'], label='Linear Model', linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('New Cases')
            plt.title('Linear Regression Model Prediction')
            plt.legend()
            st.pyplot()

            # ARIMA Model Evaluation
            st.markdown("### ARIMA Model Evaluation")
            st.write("MAE (Mean Absolute Error):", results['arima']['mae'])
            st.write("MSE (Mean Squared Error):", results['arima']['mse'])
            st.write("RMSE (Root Mean Squared Error):", results['arima']['rmse'])
            st.write("R2 Score:", results['arima']['r2'])

            # Plot ARIMA Prediction
            st.markdown("### ARIMA Model Prediction")
            plt.figure(figsize=(14, 8))
            plt.plot(results['predictions']['actual'], label='Actual test')
            plt.plot(results['predictions']['arima'], label='ARIMA Model', linestyle='-')
            plt.xlabel('Date')
            plt.ylabel('New Cases')
            plt.title('ARIMA Model Prediction')
            plt.legend()
            st.pyplot()

            # ANN Model Evaluation
            st.markdown("### ANN Model Evaluation")
            st.write("MAE (Mean Absolute Error):", results['ann']['mae'])
            st.write("MSE (Mean Squared Error):", results['ann']['mse'])
            st.write("RMSE (Root Mean Squared Error):", results['ann']['rmse'])
            st.write("R2 Score:", results['ann']['r2'])

            # Plot ANN Prediction
            st.markdown("### ANN Model Prediction")
            plt.figure(figsize=(14, 8))
            plt.plot(results['predictions']['actual'], label='Actual test')
            plt.plot(results['predictions']['ann'], label='ANN Model', linestyle=':')
            plt.xlabel('Date')
            plt.ylabel('New Cases')
            plt.title('ANN Model Prediction')
            plt.legend()
            st.pyplot()

            # LSTM Model Evaluation
            st.markdown("### LSTM Model Evaluation")
            st.write("MAE (Mean Absolute Error):", results['lstm']['mae'])
            st.write("MSE (Mean Squared Error):", results['lstm']['mse'])
            st.write("RMSE (Root Mean Squared Error):", results['lstm']['rmse'])
            st.write("R2 Score:", results['lstm']['r2'])

            # Plot LSTM Prediction
            st.markdown("### LSTM Model Prediction")
            plt.figure(figsize=(14, 8))
            plt.plot(results['predictions']['actual'], label='Actual test')
            plt.plot(results['predictions']['lstm'], label='LSTM Model', linestyle='-.')
            plt.xlabel('Date')
            plt.ylabel('New Cases')
            plt.title('LSTM Model Prediction')
            plt.legend()
            st.pyplot()

        except BrokenPipeError:
            st.error("The connection was lost. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    elif selected == "💀 New Deaths Modelling (India)":
        title = "New Deaths Models"
        subtitle = "Model Comparisons"

        csv_file_path = "st.csv"
        data_cov = pd.read_csv(csv_file_path)
        data_cov.index = data_cov['date']
        data_cov = data_cov[data_cov['location'] == "India"]

        features = ['new_deaths', 'new_cases', 'new_deaths_smoothed', 'new_vaccinations']
        data_cov = data_cov[features]

        # Create lagged dataset
        n_lags = 14
        dataset = series_to_supervised(data_cov, n_in=n_lags)
        target_col = 'new_deaths(t)'

        try:
            with st.spinner('Training models...'):
                results = train_models(target_col, dataset)

            # Linear Regression Model Evaluation
            st.markdown("### Linear Regression Model Evaluation")
            st.write("MAE (Mean Absolute Error):", results['linear_regression']['mae'])
            st.write("MSE (Mean Squared Error):", results['linear_regression']['mse'])
            st.write("RMSE (Root Mean Squared Error):", results['linear_regression']['rmse'])
            st.write("R2 Score:", results['linear_regression']['r2'])

            # Plot Linear Regression Prediction
            st.markdown("### Linear Regression Model Prediction")
            plt.figure(figsize=(14, 8))
            plt.plot(results['predictions']['actual'], label='Actual test')
            plt.plot(results['predictions']['linear'], label='Linear Model', linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('New Deaths')
            plt.title('Linear Regression Model Prediction')
            plt.legend()
            st.pyplot()

            # ARIMA Model Evaluation
            st.markdown("### ARIMA Model Evaluation")
            st.write("MAE (Mean Absolute Error):", results['arima']['mae'])
            st.write("MSE (Mean Squared Error):", results['arima']['mse'])
            st.write("RMSE (Root Mean Squared Error):", results['arima']['rmse'])
            st.write("R2 Score:", results['arima']['r2'])

            # Plot ARIMA Prediction
            st.markdown("### ARIMA Model Prediction")
            plt.figure(figsize=(14, 8))
            plt.plot(results['predictions']['actual'], label='Actual test')
            plt.plot(results['predictions']['arima'], label='ARIMA Model', linestyle='-')
            plt.xlabel('Date')
            plt.ylabel('New Deaths')
            plt.title('ARIMA Model Prediction')
            plt.legend()
            st.pyplot()

            # ANN Model Evaluation
            st.markdown("### ANN Model Evaluation")
            st.write("MAE (Mean Absolute Error):", results['ann']['mae'])
            st.write("MSE (Mean Squared Error):", results['ann']['mse'])
            st.write("RMSE (Root Mean Squared Error):", results['ann']['rmse'])
            st.write("R2 Score:", results['ann']['r2'])

            # Plot ANN Prediction
            st.markdown("### ANN Model Prediction")
            plt.figure(figsize=(14, 8))
            plt.plot(results['predictions']['actual'], label='Actual test')
            plt.plot(results['predictions']['ann'], label='ANN Model', linestyle=':')
            plt.xlabel('Date')
            plt.ylabel('New Deaths')
            plt.title('ANN Model Prediction')
            plt.legend()
            st.pyplot()

            # LSTM Model Evaluation
            st.markdown("### LSTM Model Evaluation")
            st.write("MAE (Mean Absolute Error):", results['lstm']['mae'])
            st.write("MSE (Mean Squared Error):", results['lstm']['mse'])
            st.write("RMSE (Root Mean Squared Error):", results['lstm']['rmse'])
            st.write("R2 Score:", results['lstm']['r2'])

            # Plot LSTM Prediction
            st.markdown("### LSTM Model Prediction")
            plt.figure(figsize=(14, 8))
            plt.plot(results['predictions']['actual'], label='Actual test')
            plt.plot(results['predictions']['lstm'], label='LSTM Model', linestyle='-.')
            plt.xlabel('Date')
            plt.ylabel('New Deaths')
            plt.title('LSTM Model Prediction')
            plt.legend()
            st.pyplot()

        except BrokenPipeError:
            st.error("The connection was lost. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    elif selected == "🔮 Prediction (Linear regression)":
        title = "Linear Regression Prediction"
        subtitle = "Predict COVID-19 cases using Linear Regression"
        st.title(title)
        st.subheader(subtitle)
        st.write("""
        Upload your data to predict COVID-19 cases using a pre-trained Linear Regression model.
        """)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:", data.head())

            # Ensure the correct column names are present
            required_columns = ['new_cases', 'new_cases_smoothed', 'new_deaths', 'new_vaccinations']
            if not all(column in data.columns for column in required_columns):
                st.error(f"The uploaded CSV file must contain the following columns: {required_columns}")
            else:
                # Create lagged dataset
                n_lags = 14
                dataset = series_to_supervised(data, n_in=n_lags)

                try:
                    with st.spinner('Training the Linear Regression model...'):
                        target_col = 'new_cases(t)'
                        X = dataset.drop(columns=[target_col])
                        y = dataset[target_col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = sqrt(mse)
                        r2 = r2_score(y_test, y_pred)

                        st.markdown("### Linear Regression Model Evaluation")
                        st.write("MAE (Mean Absolute Error):", mae)
                        st.write("MSE (Mean Squared Error):", mse)
                        st.write("RMSE (Root Mean Squared Error):", rmse)
                        st.write("R2 Score:", r2)

                        # Plot Prediction
                        st.markdown("### Linear Regression Model Prediction")
                        plt.figure(figsize=(14, 8))
                        plt.plot(y_test.values, label='Actual test')
                        plt.plot(y_pred, label='Predicted', linestyle='--')
                        plt.xlabel('Index')
                        plt.ylabel('New Cases')
                        plt.title('Linear Regression Model Prediction')
                        plt.legend()
                        st.pyplot()

                except BrokenPipeError:
                    st.error("The connection was lost. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
