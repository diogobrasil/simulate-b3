from classes.linear_regression import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib
import json
import os

class TrainingPipeline:
  def __init__(self, target: str, window_size: int = 3, version: str = '1.0'):
      """
      Inicializa o pipeline de treinamento com o alvo e tamanho da janela.
      
      :param target: Nome da coluna alvo no DataFrame.
      :param window_size: Tamanho da janela para as features de lag.
      """
      self.target = target
      self.window_size = window_size
      self.version = version
     
  #--Funções--

  # Função para carregar os dados do csv
  @staticmethod
  def load_data():
      """
      Carrega os dados do CSV e prepara o DataFrame.
      
      """
      df = pd.read_csv("./data/stocks.csv")
      df['Date'] = pd.to_datetime(df['Date'])
      df.set_index('Date', inplace=True)
      return df

  # Função para janelamento dos dados
  def create_window_data(self, df: pd.DataFrame):
        """
        Gera X e y onde X contém os últimos 'window_size' preços (terminando em P_t),
        e y é o preço do dia seguinte (P_{t+1}).
        
        Esta versão inclui o preço do dia atual (P_t) nas features.
        """
        # Muda o range para ir de 'window_size-1' até 0
        # Ex: window=3 -> range(2, -1, -1) -> shifts de 2, 1, 0
        lags = [df[self.target].shift(lag).rename(f'lag_{lag}') for lag in range(self.window_size - 1, -1, -1)]
        
        y = df[self.target].shift(-1).rename('y_next')

        df_lagged = pd.concat(lags + [y], axis=1).dropna()
        
        # Atualiza a lista de colunas para extrair X
        X = df_lagged[[f'lag_{lag}' for lag in range(self.window_size - 1, -1, -1)]].values
        y = df_lagged['y_next'].values

        return X, y

  # Função para separar os dados em treino, validação e teste
  def split_data_by_date(self, df: pd.DataFrame, train_end: str = '2017-12-31', val_end: str = '2018-12-31'):
      """
      Divide em treino, validação e teste com base em datas.
      """
      train_df = df.loc[:train_end]
      val_df   = df.loc[train_end:val_end]
      test_df  = df.loc[val_end:]

      X_train, y_train = self.create_window_data(train_df)
      X_val,   y_val   = self.create_window_data(val_df)
      X_test,  y_test  = self.create_window_data(test_df)

      return X_train, X_val, X_test, y_train, y_val, y_test

  # Função para normalizar os dados
  @staticmethod
  def normalize_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
      """
      Ajusta MinMaxScaler em X_train e y_train, e transforma val/test.
      """
      scaler_X = MinMaxScaler()
      scaler_y = MinMaxScaler()

      X_train_norm = scaler_X.fit_transform(X_train)
      X_val_norm   = scaler_X.transform(X_val)
      X_test_norm  = scaler_X.transform(X_test)

      y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
      y_val_norm   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
      y_test_norm  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

      return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, scaler_X, scaler_y

  # Função para avaliar o modelo
  @staticmethod
  def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray):
      """
      Calcula métricas de regressão.
      """
      mse = mean_squared_error(y_true, y_pred)
      rmse = np.sqrt(mse)
      mae = mean_absolute_error(y_true, y_pred)
      r2 = r2_score(y_true, y_pred)

      return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

  # Função para treinar o modelo
  def train_and_evaluate(self):
      """
      Pipeline de treinamento de regressão linear:
        - carrega dados
        - divide treino/val/test
        - normaliza
        - treina pela equação normal
        - avalia e salva artefatos
      """
      df = self.load_data()

      if self.target not in df.columns:
        raise ValueError(f"Target column '{self.target}' not found. Available: {list(df.columns)}")

      X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_by_date(df)
      
      
      X_train_n, X_val_n, X_test_n, y_train_n, y_val_n, y_test_n, scaler_X, scaler_y = self.normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)

      model = LinearRegression()
    
      theta = model.normal_equation(X_train_n, y_train_n)

      # previsões (normalizadas)
      y_train_pred_n = model.predict(X_train_n)
      y_val_pred_n   = model.predict(X_val_n)
      y_test_pred_n  = model.predict(X_test_n)

      
      y_train_pred = scaler_y.inverse_transform(y_train_pred_n.reshape(-1, 1)).ravel()
      y_val_pred   = scaler_y.inverse_transform(y_val_pred_n.reshape(-1, 1)).ravel()
      y_test_pred  = scaler_y.inverse_transform(y_test_pred_n.reshape(-1, 1)).ravel()

      
      train_metrics = self.evaluate_model(y_train, y_train_pred)
      val_metrics   = self.evaluate_model(y_val,   y_val_pred)
      test_metrics  = self.evaluate_model(y_test,  y_test_pred)

      if not os.path.exists("models") or not os.path.exists("utils"): # Verifica se o diretório não existe
        os.makedirs("models", exist_ok=True) # Cria o diretório (e subdiretórios se necessário)
        os.makedirs("utils/scalers", exist_ok=True)
        os.makedirs("utils/metrics", exist_ok=True)
        os.makedirs("utils/arrays", exist_ok=True)

      # salva modelo e scalers (com compressão)
      joblib.dump(model, f"models/{self.target}_model_v{self.version}.pkl", compress=('gzip', 3))
      joblib.dump(scaler_X, f"utils/scalers/{self.target}_scaler_X_v{self.version}.pkl", compress=('gzip', 3))
      joblib.dump(scaler_y, f"utils/scalers/{self.target}_scaler_y_v{self.version}.pkl", compress=('gzip', 3))
      
      # salva métricas em JSON legível
      metrics_path = f"utils/metrics/{self.target}_metrics_v{self.version}.json"
      with open(metrics_path, 'w', encoding='utf-8') as f:
          json.dump({
              'train': train_metrics,
              'val':   val_metrics,
              'test':  test_metrics
          }, f, indent=4, ensure_ascii=False)

      # salva arrays para análises posteriores
      np.save(f"utils/arrays/{self.target}_y_test_v{self.version}.npy",   y_test)
      np.save(f"utils/arrays/{self.target}_y_pred_v{self.version}.npy",   y_test_pred)
      np.save(f"utils/arrays/{self.target}_X_test_norm_v{self.version}.npy", X_test_n)

      return {
          'model': model,
          'theta': theta,
          'metrics': {
              'train': train_metrics,
              'val': val_metrics,
              'test': test_metrics
          },
          'y_test': y_test,
          'y_pred': y_test_pred
      }
