import numpy as np
import pandas as pd


class ActionPredictionTrading:
    """
    Classe para simular operações de trading e comparar com buy-and-hold,
    recebendo os valores reais e preditos para o período de simulação.
    """

    def __init__(self, simulation_df: pd.DataFrame, ticker: str):
        """
        Args:
            simulation_df: DataFrame já preparado com as colunas
                           ['date', 'actual', 'predicted', 'actual_next']
                           para a semana de simulação.
            ticker: O ticker da ação que está sendo simulada.
        """
        if not all(col in simulation_df.columns for col in ['date', 'actual', 'predicted', 'actual_next']):
            raise ValueError("O DataFrame de simulação deve conter as colunas 'date', 'actual', 'predicted', 'actual_next'.")
        
        self.df = simulation_df.copy()
        self.full_df = simulation_df.copy() # full_df também será o período da simulação para B&H
        self.ticker = ticker
        
        # O self.df['actual_next'] pode ter NaN na última linha se o input original tinha.
        # Para simulate_trading, ele itera len(self.df) - 1, o que geralmente evita o NaN no final.
        # No entanto, para garantir, vamos remover NaNs em actual_next se houver,
        # ou garantir que o df passado tenha o formato correto.
        self.df.dropna(subset=['actual_next'], inplace=True)
        self.full_df.dropna(subset=['actual_next'], inplace=True) # Para B&H, garantindo consistência


    def simulate_trading(
        self,
        stop_loss: bool = False,
        initial_capital: float = 100000,
        shares_per_trade: int = 100,
        stop_type: str = 'percent',
        stop_value: float = 0.03,
        dead_zone_pct: float = 0.005 # Mantido, mas comentado no loop se não for usar
    ) -> dict:
        capital = initial_capital
        capital_history = [capital]
        hits = 0
        total_trades = 0
        profits = []
        stop_triggered = 0

        # O loop agora itera sobre as linhas do DataFrame de simulação já preparado
        # O len(self.df) já é 4 se você preparou para 4 trades (D1-D4)
        for i in range(len(self.df)): # Loop diretamente sobre as 4 linhas
            price_today = self.df.iloc[i]['actual']
            price_tomorrow = self.df.iloc[i]['actual_next'] # 'actual_next' já está na linha atual
            predicted_price = self.df.iloc[i]['predicted'] # 'predicted' já é a previsão para 'actual_next'

            # Removido dead_zone_pct do loop para simplificar e focar no que você deseja usar
            # diff_pct = abs(predicted_price - price_today) / price_today
            # if diff_pct < dead_zone_pct:
            #     continue 

            limit = stop_value * price_today if stop_type == 'percent' else stop_value
            limit_amt = limit * shares_per_trade

            position = 'long' if predicted_price > price_today else 'short'
            pnl = ((price_tomorrow - price_today) if position == 'long'
                   else (price_today - price_tomorrow)) * shares_per_trade

            if stop_loss and pnl < -limit_amt:
                pnl = -limit_amt
                stop_triggered += 1

            capital += pnl
            profits.append(pnl)
            total_trades += 1
            if pnl > 0:
                hits += 1
            capital_history.append(capital)
        
        # Se não houver trades (len(self.df) é 0), evita divisão por zero
        if total_trades == 0:
            return {
                'total_return': 0.0, 'hit_rate': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown': 0.0, 'final_capital': initial_capital,
                'total_trades': 0, 'stop_triggered': 0,
                'predicted_prices': [], 'today_prices': [], 'tomorrow_prices': [], 'dates': []
            }


        hit_rate = hits / total_trades if total_trades else 0
        total_return = (capital - initial_capital) / initial_capital
        sharpe_ratio = (np.mean(profits) / np.std(profits)
                        if len(profits) > 1 and np.std(profits) != 0 else 0) # Adicionado np.std != 0
        peak = np.maximum.accumulate(capital_history)
        max_drawdown = np.max((peak - capital_history) / peak)

        return {
            'total_return': total_return,
            'hit_rate': hit_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': capital,
            'total_trades': total_trades,
            'stop_triggered': stop_triggered,
            'predicted_prices': self.df['predicted'].tolist(),
            'today_prices': self.df['actual'].tolist(),
            'tomorrow_prices': self.df['actual_next'].tolist(),
            'dates': self.df['date'].tolist()
        }

    def simulate_buy_and_hold(
        self,
        initial_capital: float = 100000,
        shares: int = 100
    ) -> dict:
        # full_df agora é o DataFrame da semana de simulação, já que é o que foi passado
        df_bh = self.full_df 
        if df_bh.empty:
            raise ValueError("DataFrame for Buy-and-Hold is empty. Ensure the data was loaded correctly.")
        
        # Se for uma simulação de 4 trades, df_bh terá 4 ou 5 linhas.
        # price_buy é o preço do primeiro dia da semana (df.iloc[0]['actual'])
        # price_sell é o preço do último dia da simulação (o "tomorrow_price" da última linha)
        price_buy = df_bh.iloc[0]['actual']
        # O último preço para Buy-and-Hold deve ser o 'actual_next' da última linha.
        price_sell = df_bh.iloc[-1]['actual_next'] if not df_bh['actual_next'].isnull().all() else df_bh.iloc[-1]['actual'] # Fallback if actual_next is NaN for some reason

        profit = (price_sell - price_buy) * shares
        final_capital = initial_capital + profit
        total_return = profit / initial_capital

        # Para o histórico de capital do Buy-and-Hold durante a semana
        capital_history = [initial_capital]
        for i in range(len(df_bh)):
            # Capital no dia atual (i), usando o preço atual e o preço de compra
            current_capital = initial_capital + (df_bh.iloc[i]['actual'] - price_buy) * shares
            capital_history.append(current_capital)
        # Adiciona o capital final baseado no 'price_sell'
        if len(df_bh) > 0 and 'actual_next' in df_bh.columns and not pd.isna(df_bh.iloc[-1]['actual_next']):
             capital_history.append(initial_capital + (df_bh.iloc[-1]['actual_next'] - price_buy) * shares)


        return {
            'total_return': total_return,
            'initial_price': price_buy,
            'final_price': price_sell,
            'final_capital': final_capital,
            'shares_held': shares,
            'days_held': len(df_bh) + 1, # Se df_bh tem 4 dias, são 5 dias de hold (do 1o ao 5o)
            'capital_history': capital_history # Adicionado para possível visualização futura
        }
