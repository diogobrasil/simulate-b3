from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

class LRModelInput:
  def __init__(self, model: joblib.load, scaler: MinMaxScaler = MinMaxScaler(), window_size: int = 3):
    """
    Inicializa o modelo de entrada para regressão linear.
      
    :param model: Modelo de regressão linear treinado.
    :param scaler: Scaler para normalização dos dados.
    :param window_size: Tamanho da janela para as features de lag.
    """
    self.window_size = window_size
    self.scaler = scaler
    self.model = model

  def create_window_data(self, df: pd.DataFrame):
      """
      Gera data usando lag features de forma vetorizada.
      """
      
      df_lags = [df.shift(lag).rename(f'lag_{lag}') for lag in range(self.window_size - 1, -1, -1)]
      data = df_lags.dropna().values
  
      return data

  def predict(self, df: pd.DataFrame):
      """
      Faz previsões usando o modelo de regressão linear.
      
      :param df: DataFrame com os dados de entrada.
      :return: Previsões do modelo.
      """
      # Cria as features de janela
      X = self.create_window_data(df)
      
      # Normaliza os dados
      X_scaled = self.scaler.fit_transform(X)
      
      # Faz a previsão
      predictions = self.model.predict(X_scaled)

      # Desnormaliza as previsões
      predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
      
      return predictions  
