from trainning_linear_regression import TrainingPipeline

def print_metrics(results: dict):
    """
    Função para imprimir as métricas de avaliação do modelo.
    """
    print("Métricas de Avaliação:")
    for metric, value in results['metrics'].items():
        print(f"{metric}:")
        for key, val in value.items():
            print(f"{key}: {val:.4f}")

def main(window_size: int =3, version: str ='1.0'):

  #Lista com as ações a serem treinadas
  tickers = ['ITUB4','BBAS3','CYRE3','TEND3','DIRR3','ELET3','EQTL3','CMIG4','PETR3','VALE3','BRAP3']

  # Itera sobre cada ticker e treina o modelo
  for ticker in tickers:
      print(f"Treinando modelo para {ticker}...")

      # Cria o pipeline de treinamento
      pipeline = TrainingPipeline(target=ticker, window_size=window_size, version=version)

      # Treina e avalia o modelo
      results = pipeline.train_and_evaluate()

      print(f"Modelo treinado para {ticker} com sucesso!")
      print_metrics(results)

if __name__ == "__main__":
  # Executa o pipeline de treinamento
  try:
    main()
  except Exception as e:
    print(f"Ocorreu um erro durante o treinamento: {e}")
    raise
