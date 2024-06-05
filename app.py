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
    print("Saved model and training history to disk")

    # Load model and training history
    estimator.model_ = load_model('BP_saved_model.h5')
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)
    print("Loaded model and training history from disk")

    res_ts = estimator.predict(X_test)
    res_test_ANN = scaler_y.inverse_transform(res_ts.reshape(-1, 1)).flatten()
    y_test_inv = scaler_y.inverse_transform(y_test).flatten()
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
    model = Sequential()
    model.add(LSTM(7, input_shape=(train_x_LSTM.shape[1], train_x_LSTM.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1])) #activation='sigmoid'
    model.compile(loss='mean_squared_error', optimizer='adam')

    fitting = True
    fitting_save = True
    epochs = 420

    if fitting:
        history = model.fit(train_x_LSTM, y_train, epochs=epochs, batch_size=batch_size, validation_data=(test_x_LSTM, y_test), verbose=1, shuffle=False)
        if fitting_save:
            # Serialize model to JSON
            model_json = model.to_json()
            with open("LSTM_model.json", "w") as json_file:
                json_file.write(model_json)

            # Save weights to HDF5
            model.save_weights("LSTM_model.weights.h5")  # Change filename here
            print("Saved model to disk")

            # Save training history
            with open('history_LSTM.pickle', 'wb') as f:
                pickle.dump(history.history, f)

    # Load model
    from keras.models import model_from_json
    json_file = open('LSTM_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)

    # Load weights into the model
    model.load_weights("LSTM_model.weights.h5")  # Adjust filename here

    # Load training history
    with open('history_LSTM.pickle', 'rb') as f:
        history = pickle.load(f)

    print("Loaded model from disk")

    lstm_test_predict = model.predict(test_x_LSTM)
    lstm_test_predict_inv = scaler_y.inverse_transform(lstm_test_predict).flatten()
    y_test_inv = scaler_y.inverse_transform(y_test).flatten()
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


# Function to set the sidebar with icons
def sidebar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["🏡 Home", "📈 Analysis", "📊 New Cases Modelling", "💀 New Deaths Modelling", "🔮 Prediction"],
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

    # Call sidebar function
    selected = sidebar()

    # Split dataset into inputs (X) and target (Y)



    # Home Page
    if selected == "🏡 Home":
        title = "🏡 Home"
        subtitle = "Project Background"
        content = """
        <p>At the end of 2019, the COVID-19 pandemic suddenly swept across 
        the globe without warning, affecting not only public health but also 
        having a ripple effect on politics, economy, education, and other fields. 
        As a result, countries worldwide sounded the highest level of alarm to jointly 
        address what is considered the greatest crisis of modern humanity. For our data 
        science project, our targeted audience is mainly public health officials, healthcare 
        professionals, policymakers, and also researchers. These professionals serve as the 
        government's strongest bulwark when facing sudden public health emergencies.</p>
        
        <p>As leaders responsible for protecting public health, other than conducting regular health 
        education and also formulating good health policies to ensure the health standards of the people 
        are met daily. A more important role played by these professionals is during the happening of 
        crises like COVID-19. People need these experts to provide the most professional guidance to 
        control the outbreak to save their lives and protect their families. These experts fulfilled 
        their responsibilities by utilizing professional knowledge and experience related to formulating 
        effective control measures, timely releasing accurate information, and coordinating various response 
        efforts. Therefore, when the existence of technology can better realize this mission, we should adopt 
        an attitude of actively using technology by combining the convenience of technology with their professional 
        abilities, to achieve a synergy effect of 1+1 greater than 2.</p>
        
        <p>In our projects, with the data on COVID-19 cases, fatalities, vaccination coverages, and also advanced 
        machine learning algorithms and statistical models, potential surges or declines in cases can be predicted. 
        These results will be presented in clear, understandable formats, which is a user-friendly website. In other words, 
        our project aims to provide powerful tools to empower the government with key insight to make informed decisions. 
        The objectives of this project cannot be looked down upon because accurate forecasting of the pandemic's progression 
        is essential for predicting medical resources for future demands, optimizing vaccination strategies, and also 
        implementing effective public health measures which is meaningful because it is protecting life and maintaining 
        the stability of society.</p>

        <div class="subtitle">Project Objectives</div>
        <p>Data Collection and Preprocessing:</p>
        <ul>
            <li>Collecting COVID-19 case data is the initial step.</li>
            <li>Clean and preprocess the data: handle missing values and outliers and ensure that the dataset does not have duplicates and all data rows are configured for analysis.</li>
            <li>Perform exploratory data analysis (EDA) to understand the data distribution and identify any trends or patterns.</li>
        </ul>

        <p>Model Implementation:</p>
        <ol>
            <li>Linear Regression:
                <ul>
                    <li>Implement a linear regression model since there are patterns that suggest increasing values through a period, in a systematic order. Also, Linear regression is easy to maintain, and it is always good to start with.</li>
                    <li>Evaluate its performance on training and test datasets.</li>
                </ul>
            </li>
            <li>ARIMA:
                <ul>
                    <li>Develop an ARIMA model taking into account the order of autoregression (p), differencing (d), and moving average (q).</li>
                    <li>We tune the parameters of ARIMA so that it always has the best fit for each dataset. However, the default ARIMA will be created, excluding any boosters.</li>
                </ul>
            </li>
            <li>Artificial Neural Networks (ANN) and Long Short-Term Memory (LSTM):
                <ul>
                    <li>Design and train a neural network for predicting COVID-19 cases.</li>
                    <li>We will experiment with different architectures and hyperparameters to enhance model accuracy. However, any boosters still wouldn't be applied, which means only the default model's parameters will be changed to obtain results.</li>
                </ul>
            </li>
        </ol>

        <p>Model Comparison:</p>
        <ul>
            <li>Define evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) to assess model accuracy and compare them in terms of performance.</li>
        </ul>
        """
        page_layout(title, subtitle, content)




    if selected == "📈 Analysis":
        title = "Analysis"
        subtitle = "Correlation and P-Values Matrix"
        content = """
        This analysis page is included to demonstrate the relationships between 
        various data fields within our dataset. The primary focus is on identifying 
        significant correlations between metrics such as new_cases, new_deaths, new_cases_smoothed, 
        and new_deaths_smoothed. Additionally, we have provided p-values to indicate the statistical 
        significance of these correlations.
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

        st.markdown("### Correlation Matrix and Significance")
        content = """
        In the correlation matrix (refer to Figure 1), we observe notable 
        correlations between new_cases and new_deaths, as well as between new_cases
        _smoothed and new_deaths_smoothed. These correlations suggest a strong relationship between these metrics.
        """
        st.write(content)
        
        st.markdown("### Correlation Matrix Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix Heatmap', fontsize=16)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        st.pyplot()
        
        st.markdown("### P-Values Matrix")
        st.dataframe(pval_matrix_formatted)
        
        content = """
        To support these findings, we have included a p-values matrix 
        (refer to Figure 2). For almost all data relationships, such as between new_deaths 
        and new_cases, the p-value is less than 0.05. This indicates that the correlations are 
        statistically significant and not due to random chance.

        However, it is important to note that some metrics, such as new_tests, may show 
        correlations with other data fields that are not statistically significant. In these cases, the p-values are higher, 
        suggesting that any observed correlations might be coincidental and should be interpreted with caution.
        """
        st.write(content)
        
        content = """
        ### Experimental Data and Model Selection
        While we also examined the correlation involving new_vaccinations, the significance of 
        this metric is quite low. As a result, we decided not to include new_vaccinations in our primary analysis.
        """
        st.write(content)
        
        content = """
        ### Conclusion
        Based on the analysis of the correlation matrix and p-values, we concluded that the most reliable
        metrics for further modeling are new_cases, new_deaths, new_cases_smoothed, and new_deaths_smoothed. 
        These metrics demonstrate strong and statistically significant correlations, making them suitable for 
        training and modeling our dataset.

        In summary, the purpose of this analysis was to identify the most relevant data fields to work with. The 
        correlation matrix and p-values were essential in guiding our decision to focus on new_cases, new_deaths, 
        new_cases_smoothed, and new_deaths_smoothed for our subsequent modeling efforts.
        """
        st.write(content)

    if selected == "📊 New Cases Modelling":
        title = "New_cases models"
        subtitle = "Model Comparisons"
        
        st.title(title)
        st.subheader(subtitle)
        
        # Introduction text
        intro_text = """
        Now that we know which fields we want to use for training, we have developed four models. Here are the results.
        """
        st.write(intro_text)
        
        csv_file_path = "st.csv"
        data_cov = pd.read_csv(csv_file_path)
        
        data_cov.index = data_cov['date']
        data_cov = data_cov[data_cov['location'] == "India"]
        
        features = ['new_cases', 'new_cases_smoothed','new_deaths','new_vaccinations']
        data_cov = data_cov[features]
        
        # Create lagged dataset
        n_lags = 14
        dataset = series_to_supervised(data_cov, n_in=n_lags)
        
        target_col = 'new_cases(t)'
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
        plt.plot(results['predictions']['ann'], label='ANN Model', linestyle='-.')
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
        plt.plot(results['predictions']['lstm'], label='LSTM Model', linestyle=':')
        plt.xlabel('Date')
        plt.ylabel('New Cases')
        plt.title('LSTM Model Prediction')
        plt.legend()
        st.pyplot()
        
        # Conclusion text
        conclusion_text = """
        Linear regression showed the best results for predicting new_cases based on new deaths and new vaccinations.
        """
        st.write(conclusion_text)



    if selected == "💀 New Deaths Modelling":
        title = "New_deaths models"
        subtitle = "Model Comparisons"
        
        st.title(title)
        st.subheader(subtitle)
        
        # Introduction text
        intro_text = """
        This page is an evaluation of new_deaths. We used the same parameters to train our models. Let's see the results.
        """
        st.write(intro_text)
        
        csv_file_path = "st.csv"
        data_cov = pd.read_csv(csv_file_path)
        
        data_cov.index = data_cov['date']
        data_cov = data_cov[data_cov['location'] == "India"]
        
        features = ['new_cases', 'new_deaths_smoothed','new_deaths']
        data_cov = data_cov[features]
        
        # Create lagged dataset
        n_lags = 14
        dataset = series_to_supervised(data_cov, n_in=n_lags)
        
        target_col = 'new_deaths(t)'
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
        plt.plot(results['predictions']['ann'], label='ANN Model', linestyle='-.')
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
        plt.plot(results['predictions']['lstm'], label='LSTM Model', linestyle=':')
        plt.xlabel('Date')
        plt.ylabel('New Deaths')
        plt.title('LSTM Model Prediction')
        plt.legend()
        st.pyplot()
        
        # Conclusion text
        conclusion_text = """
        Most of the models performed poorly except for the linear regression model. This poor performance is most likely due to the noise created by new_vaccinations. Even though linear regression showed some promising results, we decided to reject the idea of predicting new_deaths for this project. The other models showed significantly worse performance, which might indicate that our dataset has a lot of zero values, possibly introduced during data cleaning.
        """
        st.write(conclusion_text)



    if selected == "🔮 Prediction":
        title = "Prediction"
        subtitle = "Predict Future COVID-19 Cases"
        content = """
        This section will allow you to predict future COVID-19 cases using the Linear Regression model.

        Due to the results of our previous analysis, we decided to use linear regression as the predictive tool for our COVID-19 project. 
        As noted earlier, we decided not to predict new_deaths due to the poor performance of models. However, predicting new_cases showed 
        promising results. Therefore, this model will be predicting new_cases based on new_vaccinations and new_deaths.
        """
        page_layout(title, subtitle, content)
        csv_file_path = "st.csv"
        data_cov = load_data2(csv_file_path)
        countries = data_cov['location'].unique()
        selected_country = st.selectbox('Select a Country', countries)

        if selected_country:
            country_data = filter_country_data(data_cov, selected_country)
            if not country_data.empty:
                model_lr, scaler_x, scaler_y = train_linear_regression(country_data)

                st.markdown("### Enter New Values for Prediction")
                new_deaths = st.number_input('New Deaths', min_value=0)
                new_vaccinations = st.number_input('New Vaccinations', min_value=0)

                if st.button('Predict'):
                    prediction = predict_new_cases(model_lr, scaler_x, scaler_y, new_deaths, new_vaccinations)
                    st.write(f"Predicted New Cases: {prediction:.2f}")
                    
                st.markdown(f"### Distribution of New Deaths to New Cases in {selected_country}")
                plt.figure(figsize=(10, 6))
                plt.scatter(country_data['new_cases'], country_data['new_deaths'], alpha=0.7)
                plt.xlabel('New Cases')
                plt.ylabel('New Deaths')
                plt.title('Distribution of New Deaths to New Cases')
                plt.grid(True)
                st.pyplot()
                
                st.write("""
                This graph shows the distribution in the dataset of new_deaths to new_cases. 
                It helps verify that the model performs correctly by providing a visual representation of the relationship between these variables.
                """)
            else:
                st.write("No data available for the selected country.")



        

if __name__ == '__main__':
    main()

#