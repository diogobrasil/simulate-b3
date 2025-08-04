# 📈 Simulador de Desempenho de Ativos Financeiros

Este projeto oferece uma aplicação interativa construída com Streamlit para simular o comportamento de ativos financeiros. Ele permite visualizar dinamicamente os preços reais e as previsões de um modelo de Regressão Linear, além de comparar o desempenho de uma estratégia de trading baseada nas previsões com a estratégia de Buy-and-Hold.    
Neste projeto você pode consultar [README.ipynb](./README.ipynb) para mais entendimento do código principal do app!!!

## Funcionalidades

* **Visualização Dinâmica:** Apresenta os preços reais e previstos para uma ação selecionada, com os pontos sendo plotados em "tempo real" para uma semana específica.
* **Seleção Flexível:** O usuário pode escolher a ação (ticker), o ano, o mês e a semana para a simulação.
* **Simulação de Estratégias:** Calcula e exibe métricas de desempenho para uma estratégia de trading baseada no modelo de previsão (com opção de Stop Loss) e compara com a estratégia de Buy-and-Hold para a mesma semana.

## Requisitos

Certifique-se de ter o Python 3.8+ instalado em sua máquina.

## Configuração do Ambiente

Siga os passos abaixo para configurar o ambiente e rodar a aplicação:

1.  **Crie um Ambiente Virtual (`venv`):**
    É altamente recomendado usar um ambiente virtual para gerenciar as dependências do projeto.
    ```bash
    python -m venv venv
    ```

2.  **Ative o Ambiente Virtual:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Instale as Dependências:**
    Todas as bibliotecas necessárias estão listadas no arquivo `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
*Nota: caso seja solicitado _upgrade_ do _pip_, rode o comando abaixo:*
```bash
    python -m pip install --upgrade pip
```

## Estrutura do Projeto   
```
   
.
├── app.py
├── data/
|   └──stocks.csv
├── requirements.txt
├── scripts/
|   └── train_linear_models.py
├── classes/
|   ├── trading_strategy.py  # Contém a classe ActionPredictionTrading
|   ├── pipeline_linear_regression.py
|   ├── model_input_linear_regression.py
|   └── linear_regression.py
├── models/               # Pasta criada automaticamente ao rodar o script train_linear_models.py
│   ├── VALE3_model.pkl
│   ├── PETR4_model.pkl
│   └── ... (outros modelos para suas 11 ações)
└── utils/                # Pasta criada automaticamente ao rodar o script train_linear_models.py
    ├──metrics/
    ├──scalers/
    └──arrays/
      ├── VALE3_y_test_v1.0.npy
      ├── VALE3_y_pred_v1.0.npy
      ├── PETR4_y_test_v1.0.npy
      ├── PETR4_y_pred_v1.0.npy
      └── ... (arquivos NPY para suas 11 ações)
   
```
## **Explicação dos Arquivos e Pastas:**
**APP**   
* `app.py`: O script principal da aplicação Streamlit.      

**Dados**   
* `stocks.csv`: Contém os dados históricos de todas as suas ações. **Cada coluna deve ser um ticker de ação (ex: VALE3, PETR4) e o índice/primeira coluna deve ser a data (`Date`).**

**Ambiente de desenvolvimento**   
* `requirements.txt`: Lista de todas as bibliotecas Python necessárias.

**Scripts**   
* `train_linear_models.py`: Script para treinar os modelos de regressão linear para cada ação e gerar os arquivos `.pkl` dos modelos/scalers (se aplicável) e os arquivos `.npy` do conjunto de teste.

**Classes**   
* `trading_strategy.py`: Módulo contendo a classe `ActionPredictionTrading`.   
* `linear_regression.py`: Módulo que contém a classe `LinearRegression` que usa a equação normal para encontrar os parâmetros do modelo linear.   
* `pipeline_linear_regression`: Módulo que contém a classe `TrainingPipeline`, responsável pelo treinamento e avalição das métricas do modelo linear.   
* `model_input.py`: Módulo que contém a classe `LRModelInput` responsavel pelo pipeline de _input_ dos modelos lineares.   

**Modelos**
* `models/`: Pasta para armazenar os modelos treinados (arquivos `.pkl` ou similar) para cada ação. **Embora o `app.py` não os carregue para a simulação de trade, o `train_linear_models.py` os gerará e eles são importantes para o processo de treinamento.**   

**Utils**   
* `metrics/`: Contem as métricas dos modelos para fins de consulta.   
* `scalers/`: Pasta contendo os _scalers_ dos modelos treinados.   
* `arrays/`: Pasta para armazenar os arquivos `.npy` contendo os valores reais e previstos do **conjunto de teste completo** para cada ação.

## Preparação dos Dados e Modelos

Antes de rodar o aplicativo, você deve preparar os dados e treinar os modelos:

1.  **Garanta o `stocks.csv`:** Certifique-se de que seu arquivo `stocks.csv` está na pasta `data/` na raiz do projeto com o formato correto (`Date` como índice/primeira coluna e tickers como nomes das colunas de dados).

2.  **Treine os Modelos e Gere os Arquivos `.npy`:**
    O script `train_linear_models.py` é responsável por:
    * Treinar os modelos de Regressão Linear para cada uma das suas 11 ações.
    * Salvar cada modelo treinado em `models/{TICKER}_model_v1.0.pkl`.
    * Gerar e salvar os valores reais do conjunto de teste em `utils/arrays/{TICKER}_y_test_v1.0.npy`.
    * Gerar e salvar as previsões do modelo para o conjunto de teste em `utils/arrays/{TICKER}_y_pred_v1.0.npy`.

    **É crucial que os valores nos arquivos `.npy` estejam em ordem cronológica e correspondam aos *últimos N dias* de dados da respectiva ação no `stocks.csv` (onde N é o tamanho do seu conjunto de teste).**

    Execute o script de treinamento:
    ```bash
    python scripts/train_linear_models.py
    ```
    (Certifique-se de que `train_linear_models.py` está configurado para salvar os `.npy`s com a nomenclatura e na pasta corretas.)

## Como Rodar o Aplicativo

Após configurar o ambiente e preparar os dados/modelos, inicie o aplicativo Streamlit:

1.  **Certifique-se de que seu ambiente virtual ainda está ativo.**
2.  **Execute o comando:**
    ```bash
    streamlit run app.py
    ```

O aplicativo será aberto automaticamente em seu navegador padrão.

## Notas Importantes

* A simulação de trade assume que os arquivos `.npy` de teste contêm os dados na mesma ordem cronológica que as datas no `stocks.csv`.
* Você pode ajustar os parâmetros da simulação de trading (Capital Inicial, Ações por Trade, Stop Loss) na barra lateral do aplicativo Streamlit.   
* Consulte [README.ipynb](./README.ipynb) para mais entendimento do código principal do app.

---
# Reconhecimentos e Direitos Autorais   

@autor: Diogo Brasil Da Silva, Emanuel Lopes Silva e Matheus Costa Alves

@contato: diogobrasildasilva@gmail.com, emanuelsilva.slz@gmail.com e mathii10costa@gmail.com

@data última versão: 25/07/2025

@versão: 1.0

@outros repositórios: https://github.com/diogobrasil, https://github.com/EmanuelSilva69 e https://github.com/matheus2049alves

@Agradecimentos: Universidade Federal do Maranhão (UFMA), Professor Doutor Thales Levi Azevedo Valente, e colegas de curso.

Copyright/License Este material é resultado de um trabalho acadêmico para a disciplina TCC-TRABALHO DE CONCLUSÃO DE CURSO, sob a orientação do professor Dr. THALES LEVI AZEVEDO VALENTE, semestre letivo 2025.1, curso Engenharia da Computação, na Universidade Federal do Maranhão (UFMA). Todo o material sob esta licença é software livre: pode ser usado para fins acadêmicos e comerciais sem nenhum custo. Não há papelada, nem royalties, nem restrições de "copyleft" do tipo GNU. Ele é licenciado sob os termos da Licença MIT, conforme descrito abaixo, e, portanto, é compatível com a GPL e também se qualifica como software de código aberto. É de domínio público. Os detalhes legais estão abaixo. O espírito desta licença é que você é livre para usar este material para qualquer finalidade, sem nenhum custo. O único requisito é que, se você usá-los, nos dê crédito. Licenciado sob a Licença MIT. Permissão é concedida, gratuitamente, a qualquer pessoa que obtenha uma cópia deste software e dos arquivos de documentação associados (o "Software"), para lidar no Software sem restrição, incluindo sem limitação os direitos de usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do Software, e permitir pessoas a quem o Software é fornecido a fazê-lo, sujeito às seguintes condições: Este aviso de direitos autorais e este aviso de permissão devem ser incluídos em todas as cópias ou partes substanciais do Software. O SOFTWARE É FORNECIDO "COMO ESTÁ", SEM GARANTIA DE QUALQUER TIPO, EXPRESSA OU IMPLÍCITA, INCLUINDO MAS NÃO SE LIMITANDO ÀS GARANTIAS DE COMERCIALIZAÇÃO, ADEQUAÇÃO A UM DETERMINADO FIM E NÃO INFRINGÊNCIA. EM NENHUM CASO OS AUTORES OU DETENTORES DE DIREITOS AUTORAIS SERÃO RESPONSÁVEIS POR QUALQUER RECLAMAÇÃO, DANOS OU OUTRA RESPONSABILIDADE, SEJA EM AÇÃO DE CONTRATO, TORT OU OUTRA FORMA, DECORRENTE DE, FORA DE OU EM CONEXÃO COM O SOFTWARE OU O USO OU OUTRAS NEGOCIAÇÕES NO SOFTWARE. Para mais informações sobre a Licença MIT: https://opensource.org/licenses/MIT
