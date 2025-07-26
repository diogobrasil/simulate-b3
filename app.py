import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from datetime import date, timedelta
import calendar
from trading_strategy import ActionPredictionTrading

st.set_page_config(layout="wide", page_title="Simulação Dinâmica de Ativos", page_icon="📈")

st.title("📈 Simulador de Desempenho de Ativos Financeiros")
st.write("Visualização de valores reais e previstos para ativos da Bolsa.")

# --- Funções de Ajuda ---

@st.cache_data # Cache para carregar o DataFrame apenas uma vez
def load_all_stock_data_simplified():
    """
    Carrega o DataFrame completo onde cada coluna é uma ação.
    A primeira coluna é a 'Date'.
    """
    try:
        df = pd.read_csv('./data/stocks.csv', parse_dates=['Date'], index_col='Date')
        df = df.sort_index() # Garante que as datas estão ordenadas
        return df
    except FileNotFoundError:
        st.error("Arquivo 'stocks.csv' não encontrado. Por favor, coloque-o na mesma pasta do script.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar stocks.csv: {e}")
        st.stop()

@st.cache_data # Cache para carregar os arquivos NPY completos do conjunto de teste
def load_full_test_set_npy(asset_ticker):
    """Carrega os arquivos NPY para o conjunto de teste completo do ativo."""
    real_path = f'./utils/arrays/{asset_ticker}_y_test_v1.0.npy'
    predicted_path = f'./utils/arrays/{asset_ticker}_y_pred_v1.0.npy'
    try:
        real_data = np.load(real_path)
        predicted_data = np.load(predicted_path)
        if len(real_data) != len(predicted_data):
            st.error(f"Erro: Os arquivos NPY para {asset_ticker} têm tamanhos diferentes. Real: {len(real_data)}, Predito: {len(predicted_data)}.")
            st.stop()
        return real_data, predicted_data
    except FileNotFoundError:
        st.error(f"Arquivos NPY de conjunto de teste para {asset_ticker} não encontrados. Esperado: '{real_path}' e '{predicted_path}'.")
        st.error("Por favor, verifique se seus arquivos NPY estão nomeados corretamente e na pasta './utils/arrays/'.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar arquivos NPY para {asset_ticker}: {e}")
        st.stop()

def get_seven_dates_for_week(df_asset_index, year, month_name, week_number_str):
    """
    Retorna uma lista de 7 pd.Timestamps para a semana selecionada
    a partir do índice de datas do DataFrame de um ativo.
    """
    month_map = {
        "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4, "Maio": 5, "Junho": 6,
        "Julho": 7, "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
    }
    month_num = month_map[month_name]
    week_num = int(week_number_str.split('ª')[0])

    # Filtrar datas para o ano e mês selecionados
    dates_in_month_timestamps = df_asset_index[(df_asset_index.year == year) &
                                                 (df_asset_index.month == month_num)].tolist()
    dates_in_month_timestamps.sort() # Garante ordem cronológica

    if not dates_in_month_timestamps:
        return None, "Não há dados para o mês e ano selecionados no histórico disponível."

    # Agrupar datas em "semanas" de até 7 dias, baseadas nos dias disponíveis
    weeks_list_of_timestamps = [dates_in_month_timestamps[i:i + 7] for i in range(0, len(dates_in_month_timestamps), 7)]

    if week_num > 0 and week_num <= len(weeks_list_of_timestamps):
        selected_week_timestamps = weeks_list_of_timestamps[week_num - 1]
        
        # Opcional: validar se a semana tem exatamente 7 dias úteis/negociados
        if len(selected_week_timestamps) != 7: # ESSA VALIDAÇÃO É CRÍTICA PARA A CONSISTÊNCIA DA SIMULAÇÃO DE 7 DIAS
            return None, f"A {week_number_str} de {month_name} em {year} tem {len(selected_week_timestamps)} dias negociados, mas esperamos 7 para a simulação de trading. Escolha outra semana."
        
        return selected_week_timestamps, None
    else:
        return None, "Semana selecionada fora do intervalo de dados disponíveis para o mês."


# --- Carregar o DataFrame geral (para as datas) ---
df_all_stocks = load_all_stock_data_simplified()

# --- Sidebar para Seleção ---
st.sidebar.header("Configurações da Simulação")

# Os nomes das colunas (exceto 'Date') são os tickers das ações
available_assets = df_all_stocks.columns.tolist()
selected_asset = st.sidebar.selectbox("Selecione o Ativo:", available_assets)

# Obter anos disponíveis a partir do índice (coluna 'Date') do DataFrame do ativo
# É mais seguro obter os anos apenas do ativo selecionado
df_selected_asset_series = df_all_stocks[selected_asset]
available_years = sorted(df_selected_asset_series.index.year.unique().tolist(), reverse=True)
selected_year = st.sidebar.selectbox("Selecione o Ano:", available_years)

available_months = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
selected_month = st.sidebar.selectbox("Selecione o Mês:", available_months)

available_weeks = ["1ª Semana", "2ª Semana", "3ª Semana", "4ª Semana", "5ª Semana"] # Inclui 5ª semana caso o mês tenha
selected_week = st.sidebar.selectbox("Selecione a Semana:", available_weeks)

# Opções de simulação de trading
st.sidebar.subheader("Parâmetros da Estratégia")
initial_capital = st.sidebar.number_input("Capital Inicial:", min_value=1000.0, value=100000.0, step=1000.0)
shares_per_trade = st.sidebar.number_input("Ações por Trade:", min_value=1, value=100, step=10)
enable_stop_loss = st.sidebar.checkbox("Habilitar Stop Loss?", value=True)
stop_value_percent = st.sidebar.slider("Valor Stop Loss (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.5) / 100.0


# --- Lógica Principal da Simulação ---
if st.sidebar.button("Rodar Simulação"):
    st.info(f"Preparando simulação para **{selected_asset}** na **{selected_week}** de **{selected_month}/{selected_year}**...")

    # 1. Carregar os conjuntos de teste completos para o ativo selecionado
    real_full_test, predicted_full_test = load_full_test_set_npy(selected_asset)
    
    # 2. Obter as 7 datas correspondentes à semana selecionada para o ativo específico
    seven_dates_for_week_timestamps, error_msg = \
        get_seven_dates_for_week(df_selected_asset_series.index, selected_year, selected_month, selected_week)

    if error_msg:
        st.error(error_msg)
        st.stop()
    
    # 3. Encontrar a fatia correta nos arrays NPY usando as datas
    # Assumimos que os NPYs correspondem aos últimos N dias do histórico do ativo no stocks.csv.
    num_test_points = len(real_full_test)
    
    # As datas no DataFrame que correspondem ao período de teste dos NPYs
    # É fundamental que o índice de df_selected_asset_series esteja ordenado e completo para o período de teste
    test_set_dates_in_df = df_selected_asset_series.index[-num_test_points:]

    # Encontra o índice da primeira data da semana selecionada DENTRO do período de teste NPY
    try:
        # Pega a posição da primeira data da semana selecionada no índice do período de teste
        start_idx_in_test_set = test_set_dates_in_df.get_loc(seven_dates_for_week_timestamps[0])
    except KeyError:
        st.error(f"Erro: A primeira data da semana selecionada ({seven_dates_for_week_timestamps[0].strftime('%Y-%m-%d')}) não foi encontrada no período de teste NPY para {selected_asset}. Verifique a consistência das datas e do conjunto de teste.")
        st.stop()
    
    # Verifica se há dados suficientes nos NPYs a partir do start_idx_in_test_set para cobrir 7 dias
    if (start_idx_in_test_set + 7) > num_test_points:
        st.error(f"Dados NPY insuficientes para a semana selecionada. O conjunto de teste NPY não cobre o período até {seven_dates_for_week_timestamps[-1].strftime('%Y-%m-%d')}.")
        st.stop()

    # Fatia os arrays NPY para os 7 dias da semana selecionada
    real_values_for_plot = real_full_test[start_idx_in_test_set : start_idx_in_test_set + 7]
    predicted_values_for_plot = predicted_full_test[start_idx_in_test_set : start_idx_in_test_set + 7]
    
    # As datas para plotagem são as Timestamps que já obtivemos
    dates_for_plot = seven_dates_for_week_timestamps

    # Validação final: garantir que os tamanhos das listas são consistentes
    if not (len(dates_for_plot) == len(real_values_for_plot) == len(predicted_values_for_plot) == 7):
        st.error("Erro interno: Inconsistência no número de pontos para plotagem. Deveriam ser 7 para a semana.")
        st.stop()

    # --- Preparar DataFrame para a Classe de Trading ---
    # Este DataFrame terá 6 linhas (D1-D6) para 6 trades (um por dia útil)
    # cada linha representa a informação DISPONÍVEL no início do dia
    
    # date: D1 a D6
    trading_dates = dates_for_plot[0:6] 
    # actual: R1 a R6 (preço de "hoje" para tomar decisão)
    trading_actual = real_values_for_plot[0:6]
    # predicted: Pred_R2 a Pred_R7 (previsão para "amanhã", feita "hoje")
    # ATENÇÃO: predicted_values_for_plot[i] é a previsão para real_values_for_plot[i]
    # Então, se queremos a previsão para R2 (que acontece em D2), precisamos de predicted_values_for_plot[1]
    trading_predicted = predicted_values_for_plot[1:7]
    # actual_next: R2 a R7 (preço real de "amanhã" para calcular PnL)
    trading_actual_next = real_values_for_plot[1:7]

    df_for_trading_class = pd.DataFrame({
        'date': trading_dates,
        'actual': trading_actual,
        'predicted': trading_predicted,
        'actual_next': trading_actual_next
    })

    if df_for_trading_class.empty:
        st.warning("Nenhum dado de trade preparado para a semana selecionada.")
        st.stop()

    # --- Executar Simulações de Trading ---
    trading_simulator = ActionPredictionTrading(df_for_trading_class, ticker=selected_asset)

    model_strategy_results = trading_simulator.simulate_trading(
        stop_loss=enable_stop_loss,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        stop_value=stop_value_percent,
        stop_type='percent'
    )

    buy_and_hold_results = trading_simulator.simulate_buy_and_hold(
        initial_capital=initial_capital,
        shares=shares_per_trade # Assumimos que comprou o mesmo número de ações para manter a base de comparação
    )

    # --- Início da Plotagem Dinâmica (Gráfico) ---
    st.subheader(f"Simulação Dinâmica para {selected_asset}")

    initial_data_display = pd.DataFrame({
        'Data': pd.to_datetime([]),
        'Preço Real': [],
        'Preço Previsto': []
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Preço Real', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Preço Previsto', line=dict(color='red', dash='dot', width=2)))

    fig.update_layout(
        title='Preços Reais vs. Preços Previstos (Simulação Semanal)',
        xaxis_title='Data',
        yaxis_title='Preço de Fechamento',
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        xaxis_tickformat='%d/%m/%Y'
    )

    chart_placeholder = st.empty()
    table_placeholder = st.empty()
    
    st.info("Simulação de gráfico em andamento... Aguarde os pontos serem plotados. ⏳")

    display_df = initial_data_display.copy()

    for i in range(len(dates_for_plot)): # Itera sobre os 7 dias para o gráfico
        new_row = pd.DataFrame({
            'Data': [dates_for_plot[i]],
            'Preço Real': [real_values_for_plot[i]],
            'Preço Previsto': [predicted_values_for_plot[i]]
        })
        display_df = pd.concat([display_df, new_row], ignore_index=True)
        
        with fig.batch_update():
            fig.data[0].x = display_df['Data']
            fig.data[0].y = display_df['Preço Real']
            fig.data[1].x = display_df['Data']
            fig.data[1].y = display_df['Preço Previsto']
        
        with chart_placeholder:
            st.plotly_chart(fig, use_container_width=True)
        
        with table_placeholder:
            # Mostra apenas as primeiras 6 linhas do df_for_trading_class, se relevante
            # display_df agora contém os 7 dias de dados para o gráfico.
            # Se quiser mostrar a tabela de dados de trade, pode ser:
            st.dataframe(display_df.set_index('Data').style.format(precision=2))
        
        time.sleep(0.7) # Delay para o efeito dinâmico

    st.success("Simulação de gráfico concluída! ✅")

    # --- Exibir Resultados das Estratégias (Tabela) ---
    st.subheader("📊 Comparativo de Estratégias de Trade")

    # Preparar dados para a tabela de resultados
    results_data = {
        "Métrica": [
            "Retorno Total",
            "Capital Final",
            "Taxa de Acerto (Hit Rate)",
            "Sharpe Ratio",
            "Max Drawdown",
            "Total de Trades",
            "Stop Loss Acionado"
        ],
        "Estratégia do Modelo": [
            f"{model_strategy_results['total_return']:.5%}",
            f"R$ {model_strategy_results['final_capital']:,.2f}",
            f"{model_strategy_results['hit_rate']:.2%}",
            f"{model_strategy_results['sharpe_ratio']:.2f}",
            f"{model_strategy_results['max_drawdown']:.2%}",
            f"{model_strategy_results['total_trades']}",
            f"{model_strategy_results['stop_triggered']}"
        ],
        "Buy-and-Hold": [
            f"{buy_and_hold_results['total_return']:.5%}",
            f"R$ {buy_and_hold_results['final_capital']:,.2f}",
            "-", # Não aplicável para B&H
            "-", # Não aplicável para B&H
            "-", # Max Drawdown para B&H precisaria de mais lógica na classe, se quiser calcular
            "-", # Não aplicável para B&H
            "-"  # Não aplicável para B&H
        ]
    }

    df_results = pd.DataFrame(results_data)
    st.table(df_results.set_index("Métrica"))

    st.markdown("""
    **Observações:**
    * **Retorno Total:** Lucro/Prejuízo percentual em relação ao capital inicial.
    * **Capital Final:** Valor final do capital após a simulação.
    * **Taxa de Acerto (Hit Rate):** Percentual de trades lucrativos.
    * **Sharpe Ratio:** Mede o retorno da estratégia ajustado ao risco. Valores maiores são melhores.
    * **Max Drawdown:** Maior queda percentual do capital a partir de um pico.
    """)
