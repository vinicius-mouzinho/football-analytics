import soccerdata as sd
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplsoccer
import socceraction
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from io import StringIO
import warnings
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from mplsoccer import Pitch

warnings.simplefilter(action='ignore', category=FutureWarning)

def distance_to_goal(x, y):
    goal_x, goal_y = 105, 34 # Coordenadas do gol adversário
    return math.sqrt((x - goal_x) ** 2 + (y - goal_y)**2)


def is_deep_completion(x, y):
    return distance_to_goal(x, y) <= 25


def calculate_total_xT(group):
    return group['xT'].sum()


def play_left_to_right_away_games(df): 
    def transform_coordinates(row):
        if row['team'] == row['away_team_schedule']:  # Aplica a transformação apenas se o time for o time de fora
            row['start_x'] = 105 - row['start_x']
            row['end_x'] = 105 - row['end_x']
            row['start_y'] = 68 - row['start_y']
            row['end_y'] = 68 - row['end_y']
        return row
        
    return df.apply(transform_coordinates, axis=1)
 

def extract_brazilian_league_players_ws(chromedriver_path, url):
    # Caminho para o ChromeDriver
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service)
    
    # URL da página
    driver.get(url)
    
    # Espera até que a tabela seja carregada
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.ID, 'stage-top-player-stats')))
    
    # Inicializando uma lista para armazenar os dados de todas as páginas
    all_data = []

    # Função para extrair dados da tabela na página atual
    def extract_table_data():
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        # Procurando a tabela dentro do contêiner com id 'stage-top-player-stats'
        table_container = soup.find('div', {'id': 'stage-top-player-stats'})
        table = table_container.find('table') if table_container else None
        
        if table:
            table_html = str(table)
            return pd.read_html(StringIO(table_html))[0]
        return pd.DataFrame()

    # Extraindo dados de todas as páginas
    while True:
        # Extraindo dados da tabela na página atual
        data = extract_table_data()
        if not data.empty:
            all_data.append(data)
        
        # Tentando clicar no botão 'next' para ir para a próxima página
        try:
            next_button = driver.find_element(By.LINK_TEXT, 'next')
            if 'disabled' in next_button.get_attribute('class'):
                break  # Se o botão 'next' estiver desativado, saia do loop
            next_button.click()
            time.sleep(0.75)  # Aguardando um tempo para a próxima página carregar
        except Exception as e:
            print(f"Erro ao tentar ir para a próxima página: {e}")
            break

    # Fechando o navegador
    driver.quit()

    # Concatenando todos os dados em um único DataFrame, se houver dados
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print('Sucesso.')
        return final_df
    else:
        print("Nenhuma tabela encontrada em todas as páginas.")
        return pd.DataFrame()


def correct_ws_names(df):
    # Filtrando apenas as colunas desejadas
    df_filtered = df[['Player.1', 'Mins']].copy()

    # Lista dos clubes
    clubes = ['Flamengo', 'Palmeiras', 'Botafogo RJ', 'Athletico Paranaense', 'Sao Paulo', 'Internacional', 'Bahia', 'Cruzeiro', 'Juventude', 'Atletico MG', 'Red Bull Bragantino', 'Fortaleza', 'Vasco da Gama',
              'Criciuma', 'Cuiaba', 'Corinthians', 'Vitoria', 'Atletico GO', 'Gremio', 'Fluminense']

    # Função para separar jogador e equipe
    def separar_jogador_equipe(player):
        for clube in clubes:
            if clube in player:
                nome_completo, resto = player.split(clube, 1)
                return nome_completo.strip(), clube
        return player, 'Desconhecido'

    # Aplicando a função ao DataFrame
    df_filtered[['Jogador', 'Equipe']] = df_filtered['Player.1'].apply(lambda x: pd.Series(separar_jogador_equipe(x)))

    # Removendo a coluna 'Player.1' agora que temos 'Jogador' e 'Equipe'
    df_final = df_filtered[['Jogador', 'Equipe', 'Mins']]

    return df_final


def corrigir_nomes_repetidos_ws(df):
    # Contar a ocorrência de cada nome de jogador
    jogador_counts = df['Jogador'].value_counts()
    
    # Filtrar apenas os jogadores com nomes duplicados
    jogadores_duplicados = jogador_counts[jogador_counts > 1].index
    
    # Função para adicionar as três primeiras letras da equipe aos jogadores duplicados
    def adicionar_sigla_equipe(row):
        if row['Jogador'] in jogadores_duplicados:
            sigla_equipe = row['Equipe'][:3].upper()
            return f"{row['Jogador']} ({sigla_equipe})"
        return row['Jogador']
    
    # Aplicando a função ao DataFrame
    df['Jogador'] = df.apply(adicionar_sigla_equipe, axis=1)
    
    return df


def get_brasileirao2024_events(retry_missing=True):
    # Instanciar o scraper 
    ws = sd.WhoScored(leagues="BRA-Brasileirao", seasons='2024')

    # Fazer uma lista dos ids de todas as partidas
    schedule = ws.read_schedule()
    br2024_game_ids = schedule['game_id'].tolist()

    # Baixar o dataframe de todas as partidas no campeonato
    br2024_df = ws.read_events(br2024_game_ids, output_fmt='spadl', retry_missing=retry_missing)

    # Renomear colunas do schedule
    schedule_renamed = schedule.rename(columns={'home_team': 'home_team_schedule', 'away_team': 'away_team_schedule'})

    # Mesclar dataframes
    br2024_df_transformed = br2024_df.merge(schedule_renamed[['game_id', 'home_team_schedule', 'away_team_schedule']], on='game_id', how='left')

    # Aplicar a função play_left_to_right_away_games ao dataframe inteiro
    br2024_df_casa = play_left_to_right_away_games(br2024_df_transformed)

    # Filtrando as linhas onde a distância entre start_y e end_y seja maior que 70
    br2024_df_casa = br2024_df_casa[abs(br2024_df_casa['start_x'] - br2024_df_casa['end_x']) <= 70]

    return br2024_df_casa

def correct_ws_event_names(df):
    # Criar uma cópia do dataframe para evitar modificar o original
    df_copy = df.copy()
    
    # Identificar jogadores duplicados com nomes iguais mas times diferentes
    player_team_combinations = df_copy.groupby(['player', 'team']).size().reset_index().rename(columns={0: 'count'})
    
    # Dicionário para armazenar jogadores que precisam ser desambiguados
    disambiguation_dict = {}
    
    for index, row in player_team_combinations.iterrows():
        player = row['player']
        team = row['team']
        
        # Se o jogador já está no dicionário, significa que já existe uma entrada para ele
        if player in disambiguation_dict:
            disambiguation_dict[player].append(team)
        else:
            disambiguation_dict[player] = [team]
    
    # Renomear jogadores que têm o mesmo nome mas times diferentes
    for player, teams in disambiguation_dict.items():
        if len(teams) > 1:  # Só precisa renomear se o jogador estiver em mais de um time
            for team in teams:
                team_abbr = team[:3].upper()  # Pega as três primeiras letras do nome do time em maiúsculas
                df_copy.loc[(df_copy['player'] == player) & (df_copy['team'] == team), 'player'] = f"{player} ({team_abbr})"

    df_copy.to_csv('C:/Users/Anderson/soccerdataDfs/Brasileirão 2024.csv', encoding='utf-8-sig')
    
    return df_copy


def find_top_player_correct_passes(df):
    # Filtrar apenas passes bem-sucedidos
    successful_passes = df[df['result_id'] == 1]
    
    # Contar o número de passes certos por jogador
    correct_pass_count = successful_passes.groupby('player').size()
    
    # Ordenar a contagem de passes certos em ordem decrescente
    sorted_correct_pass_count = correct_pass_count.sort_values(ascending=False)
    
    # Descobrir o jogador com mais passes corretos
    top_player_correct_passes = sorted_correct_pass_count.idxmax()
    top_pass_count = sorted_correct_pass_count.max()
    
    return top_player_correct_passes, top_pass_count, sorted_correct_pass_count


def print_top_players_correct_passes(sorted_correct_pass_count, players_quantity):
    # Imprimir o número de passes certos de cada jogador
    print(f'NÚMERO DE PASSES CERTOS DE CADA JOGADOR:')
    print('')
    top_players = sorted_correct_pass_count.head(players_quantity)
    for player, pass_count in top_players.items():
        print(f"{player}: {pass_count} passes certos")
    print('')
    print('*' * 50)
    print('')


def find_top_player_progressive_passes(df):
    # Calcular a distância ao gol antes e depois do passe
    df['dist_to_goal_before'] = df.apply(lambda row: distance_to_goal(row['start_x'], row['start_y']), axis=1)
    df['dist_to_goal_after'] = df.apply(lambda row: distance_to_goal(row['end_x'], row['end_y']), axis=1)

    # Determinar se o passe é progressivo (mais próximo do gol em pelo menos 25%)
    progressive_threshold = 0.75  # Passe é progressivo se dist_to_goal_after < 75% * dist_to_goal_before

    df['progressive_pass'] = (
        ((df['dist_to_goal_after'] < df['dist_to_goal_before'] * progressive_threshold) & (df['end_x'] > df['start_x'])) | 
        ((df['end_x'].between(88, 105)) & (df['end_y'].between(14, 54)))
    )

    # Agrupar por jogador e contar o número de passes progressivos
    correct_progressive_pass = df[(df['progressive_pass'] == True) & (df['result_id'] == 1)]
    progressive_pass_count = correct_progressive_pass.groupby('player')['progressive_pass'].sum()

    # Ordenar os resultados do maior para o menor número de passes progressivos
    sorted_progressive_pass_count = progressive_pass_count.sort_values(ascending=False)

    # Ver o jogador com mais passes progressivos
    top_player_progressive_passes = sorted_progressive_pass_count.idxmax()
    top_progressive_passes_count = sorted_progressive_pass_count.max()

    return top_player_progressive_passes, top_progressive_passes_count, sorted_progressive_pass_count

def print_top_players_progressive_passes(sorted_progressive_pass_count, players_quantity):
    # Imprimir o número total de passes progressivos de cada jogador
    print(f'NÚMERO DE PASSES PROGRESSIVOS DE CADA JOGADOR:')
    print('')
    top_players = sorted_progressive_pass_count.head(players_quantity)
    for player, pass_count in top_players.items():
        print(f"{player}: {pass_count} passes progressivos")
    print('')
    print('*' * 50)
    print('')

def find_top_player_deep_completions(df):
    # Aplicar a função para verificar deep completions
    df['is_deep_completion'] = df.apply(lambda row: is_deep_completion(row['end_x'], row['end_y']), axis=1)

    # Filtrar apenas as deep completions bem-sucedidas
    successful_deep_completions = df[(df['result_id'] == 1) & (df['is_deep_completion'])]

    # Agrupar pelo jogador e contar o número de deep completions
    deep_completion_count = successful_deep_completions.groupby('player').size()

    # Ordenar pela contagem de deep completions em ordem decrescente
    sorted_deep_completion_count = deep_completion_count.sort_values(ascending=False)

    # Imprimir o nome do jogador com mais deep completions
    top_player_deep_completions = sorted_deep_completion_count.idxmax()
    top_deep_completion_count = sorted_deep_completion_count.max()

    return top_player_deep_completions, top_deep_completion_count, sorted_deep_completion_count

def print_top_players_deep_completions(sorted_deep_completion_count, players_quantity):
    # Imprimir o número de deep completions de cada jogador
    print('DEEP COMPLETIONS:')
    print('')
    top_players = sorted_deep_completion_count.head(players_quantity)
    for player, completion_count in top_players.items():
        print(f"{player}: {completion_count} deep completions")
    print('')
    print('*' * 50)
    print('')


def find_top_player_xT_pass(df, xT_grid_path='xT_grid.csv'):
    # Carregar a matriz xT
    xT = pd.read_csv(xT_grid_path, header=None)
    xT = np.array(xT)
    xT_rows, xT_cols = xT.shape

    # Filtrar apenas passes bem-sucedidos
    successful_passes = df[df['result_id'] == 1]

    # Mapear as coordenadas dos passes para as zonas da matriz xT
    successful_passes['x1_bin'] = pd.cut(successful_passes['start_x'], bins=xT_cols, labels=False)
    successful_passes['y1_bin'] = pd.cut(successful_passes['start_y'], bins=xT_rows, labels=False)
    successful_passes['x2_bin'] = pd.cut(successful_passes['end_x'], bins=xT_cols, labels=False)
    successful_passes['y2_bin'] = pd.cut(successful_passes['end_y'], bins=xT_rows, labels=False)

    # Calcular os valores xT para cada passe
    successful_passes['start_zone_value'] = successful_passes[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    successful_passes['end_zone_value'] = successful_passes[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    successful_passes['xT'] = successful_passes['end_zone_value'] - successful_passes['start_zone_value']
    successful_passes['xT'] = successful_passes['xT'].apply(lambda x: 0 if x < 0 else x)

    # Agrupar por jogador e calcular a soma de xT
    df_xT = successful_passes.groupby('player')['xT'].sum().reset_index(name='Total xT')

    # Ordenar pela soma de xT em ordem decrescente
    df_xT = df_xT.sort_values(by='Total xT', ascending=False)

    # Encontrar o jogador com o maior Total xT
    top_player_xT = df_xT.loc[df_xT['Total xT'].idxmax()]

    return top_player_xT, df_xT

def print_top_players_xT_pass(df_xT, players_quantity):
    # Imprimir o número total de xT de cada jogador
    print('LÍDERES EM xT por passe:')
    print('')
    
    top_players = df_xT.head(players_quantity)
    
    for index, row in top_players.iterrows():
        print(f'{row["player"]}: {row["Total xT"]:.2f}')
    
    print('')
    print('*' * 50)
    print('')

def find_top_players_progressive_dribbles(df, progressive_threshold=0.75):
    # Calcular a distância ao gol antes e depois da condução
    df['dist_to_goal_before'] = df.apply(lambda row: distance_to_goal(row['start_x'], row['start_y']), axis=1)
    df['dist_to_goal_after'] = df.apply(lambda row: distance_to_goal(row['end_x'], row['end_y']), axis=1)

    # Determinar se a condução é progressiva (mais próxima do gol em pelo menos 25%)
    df['progressive_dribble'] = (df['dist_to_goal_after'] < df['dist_to_goal_before'] * progressive_threshold)

    # Agrupar por jogador e contar o número de conduções progressivas
    correct_progressive_dribbles = df[(df['progressive_dribble'] == True) & (df['result_id'] == 1)]
    progressive_dribble_count = correct_progressive_dribbles.groupby('player')['progressive_dribble'].sum()

    # Ordenar os resultados do maior para o menor número de conduções progressivas
    sorted_progressive_dribble_count = progressive_dribble_count.sort_values(ascending=False)

    # Ver o jogador com mais conduções progressivas
    top_player_progressive_dribbles = sorted_progressive_dribble_count.idxmax()
    top_progressive_dribble_count = sorted_progressive_dribble_count.max()

    return top_player_progressive_dribbles, top_progressive_dribble_count, sorted_progressive_dribble_count

def print_top_players_progressive_dribbles(sorted_progressive_dribble_count, players_quantity):
    # Imprimir o número total de conduções progressivas de cada jogador
    print(f'NÚMERO DE CONDUÇÕES PROGRESSIVAS DE CADA JOGADOR:')
    print('')
    
    top_players = sorted_progressive_dribble_count.head(players_quantity)
    
    for player, dribble_count in top_players.items():
        print(f"{player}: {dribble_count} conduções progressivas")
    
    print('')
    print('*' * 50)
    print('')


def find_top_players_xT_dribbles(df, xT_grid_path='xT_grid.csv'):
    # Carregar a matriz xT
    xT = pd.read_csv(xT_grid_path, header=None)
    xT = np.array(xT)
    xT_rows, xT_cols = xT.shape

    # Filtrar apenas conduções bem-sucedidas
    successful_dribbles = df[df['result_id'] == 1]

    # Mapear as coordenadas das conduções para as zonas da matriz xT
    successful_dribbles['x1_bin'] = pd.cut(successful_dribbles['start_x'], bins=xT_cols, labels=False)
    successful_dribbles['y1_bin'] = pd.cut(successful_dribbles['start_y'], bins=xT_rows, labels=False)
    successful_dribbles['x2_bin'] = pd.cut(successful_dribbles['end_x'], bins=xT_cols, labels=False)
    successful_dribbles['y2_bin'] = pd.cut(successful_dribbles['end_y'], bins=xT_rows, labels=False)

    # Calcular os valores xT para cada condução
    successful_dribbles['start_zone_value'] = successful_dribbles[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    successful_dribbles['end_zone_value'] = successful_dribbles[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    successful_dribbles['xT'] = successful_dribbles['end_zone_value'] - successful_dribbles['start_zone_value']
    successful_dribbles['xT'] = successful_dribbles['xT'].apply(lambda x: 0 if x < 0 else x)

    # Agrupar por jogador e calcular a soma de xT
    df_xT_dribbles = successful_dribbles.groupby('player')['xT'].sum().reset_index(name='Total xT')

    # Ordenar pela soma de xT em ordem decrescente
    df_xT_dribbles = df_xT_dribbles.sort_values(by='Total xT', ascending=False)

    # Encontrar o jogador com o maior Total xT
    top_player_xT_dribbles = df_xT_dribbles.loc[df_xT_dribbles['Total xT'].idxmax()]

    return top_player_xT_dribbles, df_xT_dribbles

def print_top_players_xT_dribbles(df_xT_dribbles, players_quantity):
    # Imprimir o número total de xT de cada jogador
    print('LÍDERES EM xT por conduções:')
    print('')
    
    top_players = df_xT_dribbles.head(players_quantity)
    
    for index, row in top_players.iterrows():
        print(f'{row["player"]}: {row["Total xT"]:.2f}')
    
    print('')
    print('*' * 50)
    print('')


def find_top_players_recoveries(df):
    # Filtrar apenas os desarmes e interceptações bem-sucedidos
    successful_recoveries = df[df['result_id'] == 1]

    # Contar o número de desarmes e interceptações por jogador
    recovery_count = successful_recoveries.groupby('player').size()

    # Ordenar os jogadores pelo número de desarmes e interceptações em ordem decrescente
    sorted_recovery_count = recovery_count.sort_values(ascending=False)

    # Descobrir o jogador com mais desarmes e interceptações
    top_player_recoveries = sorted_recovery_count.idxmax()
    top_recovery_count = sorted_recovery_count.max()

    return sorted_recovery_count, top_player_recoveries, top_recovery_count


def print_top_players_recoveries(sorted_recovery_count, top_player_recoveries, top_recovery_count, num_players=10):
    # Imprimir o número de desarmes e interceptações de cada jogador
    print(f'NÚMERO DE DESARMES E INTERCEPTAÇÕES DE CADA JOGADOR:')
    print('')
    print(f"O jogador com mais desarmes e interceptações é {top_player_recoveries} com {top_recovery_count} recuperações.")
    print('')
    count = 0
    for player, recovery_count in sorted_recovery_count.items():
        if count >= num_players:
            break
        print(f"{player}: {recovery_count} recuperações")
        count += 1
    print('')
    print('*' * 50)
    print('')


def combine_and_print_top_xT(df_xT_pass, df_xT_dribbles, top_n=10):
    # Combinar os valores de xT de passes e dribles com base nos jogadores
    combined_df = pd.merge(df_xT_pass, df_xT_dribbles, on='player', how='outer', suffixes=('_pass', '_dribble'))

    # Substituir valores NaN por 0 para cálculos corretos
    combined_df.fillna(0, inplace=True)

    # Calcular o total de xT para cada jogador
    combined_df['Total xT'] = combined_df['Total xT_pass'] + combined_df['Total xT_dribble']

    # Ordenar os resultados pelo total de xT em ordem decrescente
    combined_df = combined_df.sort_values(by='Total xT', ascending=False)

    # Selecionar os top n jogadores
    top_players = combined_df.head(top_n)

    # Imprimir os top n jogadores
    print('LÍDERES EM xT COMBINADOS (PASSES E CONDUÇÕES):')
    print('')
    for index, row in top_players.iterrows():
        print(f'{row["player"]}: {row["Total xT"]:.2f} (Passes: {row["Total xT_pass"]:.2f}, Conduções: {row["Total xT_dribble"]:.2f})')
    print('')
    print('*' * 50)
    print('')

def combine_and_analyze_player_stats(nomes_brasileirao_sem_rep, sorted_progressive_pass_count, sorted_progressive_dribble_count, df_xT_Pass, df_xT_dribbles, sorted_recovery_count, min_minutes=720, sort_by='xT Total por 90 Minutos'):
    # Criar o DataFrame de ações progressivas combinando passes progressivos e conduções progressivas
    progressive_actions = pd.DataFrame({
        'Jogador': sorted_progressive_pass_count.index,
        'Passes Progressivos': sorted_progressive_pass_count.values,
        'Conduções Progressivas': sorted_progressive_dribble_count.reindex(sorted_progressive_pass_count.index, fill_value=0).values
    })

    # Juntar com os DataFrames df_xT_Pass e df_xT_dribbles para incluir xT
    xT_passes = df_xT_Pass.set_index('player')['Total xT'].reindex(sorted_progressive_pass_count.index, fill_value=0)
    xT_dribbles = df_xT_dribbles.set_index('player')['Total xT'].reindex(sorted_progressive_pass_count.index, fill_value=0)

    # Adicionar as colunas de xT
    progressive_actions['xT Passes'] = xT_passes.values
    progressive_actions['xT Conduções'] = xT_dribbles.values

    # Calcular as ações progressivas totais
    progressive_actions['Ações Progressivas Totais'] = progressive_actions['Passes Progressivos'] + progressive_actions['Conduções Progressivas']

    # Juntar com o DataFrame sorted_recovery_count para incluir recuperações
    recovery_count = sorted_recovery_count.reindex(sorted_progressive_pass_count.index, fill_value=0)
    progressive_actions['Recuperações'] = recovery_count.values

    # Juntar com o DataFrame nomes_brasileirao_sem_rep para obter os minutos jogados
    result_df = nomes_brasileirao_sem_rep.merge(progressive_actions, how='left', left_on='Jogador', right_on='Jogador')

    # Calcular as ações progressivas e xT por 90 minutos
    result_df['Passes Progressivos por 90 Minutos'] = (result_df['Passes Progressivos'] / result_df['Mins']) * 90
    result_df['Conduções Progressivas por 90 Minutos'] = (result_df['Conduções Progressivas'] / result_df['Mins']) * 90
    result_df['Ações Progressivas por 90 Minutos'] = (result_df['Ações Progressivas Totais'] / result_df['Mins']) * 90

    # Normalizando xT para 90 minutos
    result_df['xT Passes por 90 Minutos'] = (result_df['xT Passes'] / result_df['Mins']) * 90
    result_df['xT Conduções por 90 Minutos'] = (result_df['xT Conduções'] / result_df['Mins']) * 90

    # Calcular as recuperações por 90 minutos
    result_df['Recuperações por 90 Minutos'] = (result_df['Recuperações'] / result_df['Mins']) * 90

    # Preencher valores NaN (jogadores sem ações progressivas ou recuperações) com 0
    result_df['Ações Progressivas por 90 Minutos'] = result_df['Ações Progressivas por 90 Minutos'].fillna(0)
    result_df['Recuperações'] = result_df['Recuperações'].fillna(0)
    result_df['Recuperações por 90 Minutos'] = result_df['Recuperações por 90 Minutos'].fillna(0)

    result_df['xT Total'] = result_df['xT Passes'] + result_df['xT Conduções']
    result_df['xT Total por 90 Minutos'] = result_df['xT Passes por 90 Minutos'] + result_df['xT Conduções por 90 Minutos'] 

    # Filtrar e classificar os resultados
    result_df_filtered = result_df[result_df['Mins'] > min_minutes].sort_values(by=sort_by, ascending=False)

    # Retornar o DataFrame resultante
    return result_df_filtered


def plot_2_progressive_actions(dfPass, dfDribble, result_df, jogador1, jogador2, title):
    # Dados do jogador 1
    num_passesCJ1 = dfPass[(dfPass['player'] == jogador1) & (dfPass['result_id'] == 1)].shape[0]
    num_passesJ1 = dfPass[(dfPass['player'] == jogador1)].shape[0]
    num_passesPJ1 = len(dfPass[(dfPass['player'] == jogador1) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)])
    taxa_acertoJ1 = round(((num_passesCJ1 / num_passesJ1) * 100), 1)
    dfPass1 = dfPass[(dfPass['player'] == jogador1) & (dfPass['result_id'] == 1) & (~dfPass['progressive_pass'])]
    dfPass1P = dfPass[(dfPass['player'] == jogador1) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)]
    num_CP_J1 = len(dfDribble[(dfDribble['player'] == jogador1) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)])
    dfDribble1P = dfDribble[(dfDribble['player'] == jogador1) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)]
    num_passesP90J1 = result_df.loc[result_df['Jogador'] == jogador1, 'Passes Progressivos por 90 Minutos'].values[0]
    num_conducoesP90J1 = result_df.loc[result_df['Jogador'] == jogador1, 'Conduções Progressivas por 90 Minutos'].values[0]

    # Dados do jogador 2
    num_passesCJ2 = dfPass[(dfPass['player'] == jogador2) & (dfPass['result_id'] == 1)].shape[0]
    num_passesJ2 = dfPass[(dfPass['player'] == jogador2)].shape[0]
    num_passesPJ2 = dfPass[(dfPass['player'] == jogador2) & (dfPass['progressive_pass'] == True) & (dfPass['result_id'] == 1)].shape[0]
    taxa_acertoJ2 = round(((num_passesCJ2 / num_passesJ2) * 100), 1)
    dfPass2 = dfPass[(dfPass['player'] == jogador2) & (dfPass['result_id'] == 1) & (~dfPass['progressive_pass'])]
    dfPass2P = dfPass[(dfPass['player'] == jogador2) & (dfPass['result_id'] == 1) & (dfPass['progressive_pass'])]
    num_CP_J2 = len(dfDribble[(dfDribble['player'] == jogador2) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)])
    dfDribble2P = dfDribble[(dfDribble['player'] == jogador2) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)]
    num_passesP90J2 = result_df.loc[result_df['Jogador'] == jogador2, 'Passes Progressivos por 90 Minutos'].values[0]
    num_conducoesP90J2 = result_df.loc[result_df['Jogador'] == jogador2, 'Conduções Progressivas por 90 Minutos'].values[0]

    # Criar o campo
    pitch = VerticalPitch(positional=False, pitch_type='uefa', pitch_color='#222222', line_color='white', line_zorder=2)
    fig, axs = pitch.grid(ncols=2, axis=False, figheight=10, endnote_height=0, endnote_space=0, title_height=0.01, grid_height=0.75)
    
    # Gráfico do jogador 1
    pitch.lines(dfPass1P['start_x'], dfPass1P['start_y'], dfPass1P['end_x'], dfPass1P['end_y'], color='#97c1e7',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'][0])
    pitch.scatter(dfPass1P['end_x'], dfPass1P['end_y'], color='black', edgecolor='#97c1e7', ax=axs['pitch'][0], s=50, lw=1, zorder=2)
    pitch.lines(dfDribble1P['start_x'], dfDribble1P['start_y'], dfDribble1P['end_x'], dfDribble1P['end_y'], color='#ff6666',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'][0])
    pitch.scatter(dfDribble1P['end_x'], dfDribble1P['end_y'], color='#ff6666', edgecolor='#343334', ax=axs['pitch'][0], s=50, lw=2, zorder=3)
    
    # Gráfico do jogador 2
    pitch.lines(dfPass2P['start_x'], dfPass2P['start_y'], dfPass2P['end_x'], dfPass2P['end_y'], color='#97c1e7',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'][1])
    pitch.scatter(dfPass2P['end_x'], dfPass2P['end_y'], color='black', edgecolor='#97c1e7', ax=axs['pitch'][1], s=50, lw=1, zorder=2)
    pitch.lines(dfDribble2P['start_x'], dfDribble2P['start_y'], dfDribble2P['end_x'], dfDribble2P['end_y'], color='#ff6666',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'][1])
    pitch.scatter(dfDribble2P['end_x'], dfDribble2P['end_y'], color='#ff6666', edgecolor='#343334', ax=axs['pitch'][1], s=50, lw=2, zorder=3)
    
    # Títulos e textos
    fig.suptitle(title, fontsize=20, fontweight='bold', color='white')
    axs['pitch'][0].set_title(f'Ações progressivas de {jogador1}', fontsize=18, color='white', pad=1)
    axs['pitch'][1].set_title(f'Ações progressivas de {jogador2}', fontsize=18, color='white', pad=1)

    axs['pitch'][0].text(0.05, -0.02, f'Passes: {num_passesCJ1}/{num_passesJ1} ({taxa_acertoJ1}% de acerto)', fontsize=14, color='white', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'][0].transAxes)
    axs['pitch'][0].text(0.05, -0.05, f'{num_passesPJ1} passes progressivos ({num_passesP90J1:.2f} por 90 min.)', fontsize=12, color='#97c1e7', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'][0].transAxes)
    axs['pitch'][0].text(0.05, -0.08, f'{num_CP_J1} conduções progressivas ({num_conducoesP90J1:.2f} por 90 min.)', fontsize=12, color='#ff6666', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'][0].transAxes)

    axs['pitch'][1].text(0.05, -0.02, f'Passes: {num_passesCJ2}/{num_passesJ2} ({taxa_acertoJ2}% de acerto)', fontsize=14, color='white', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'][1].transAxes)
    axs['pitch'][1].text(0.05, -0.05, f'{num_passesPJ2} passes progressivos ({num_passesP90J2:.2f} por 90 min.)', fontsize=12, color='#97c1e7', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'][1].transAxes)
    axs['pitch'][1].text(0.05, -0.08, f'{num_CP_J2} conduções progressivas ({num_conducoesP90J2:.2f} por 90 min.)', fontsize=12, color='#ff6666', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'][1].transAxes)

    # Nota de rodapé
    note_text = "@Vasco_Analytics | Dados: Opta via WhoScored"
    note_text_2 = "Ação progressiva: ponto final está ao menos 25% mais próximo do gol do que o inicial."
    fig.text(0.02, 0.01, note_text, fontsize=16, color='#ababab', ha='left', va='center', weight='300')
    fig.text(0.02, -0.02, note_text_2, fontsize=12, color='#ababab', ha='left', va='center', weight='300')

    # Alterar a cor de fundo da figura e subplots para um tom de cinza escuro
    fig.patch.set_facecolor('#222222')  # Cor de fundo da figura

    # Salvar a figura
    fig.savefig('Mapa de passes.png', dpi=300, bbox_inches='tight', facecolor='#222222')

    plt.show()


def plot_progressive_actions(dfPass, dfDribble, result_df, jogador, title):
    # Dados do jogador
    num_passesCJ = dfPass[(dfPass['player'] == jogador) & (dfPass['result_id'] == 1)].shape[0]
    num_passesJ = dfPass[(dfPass['player'] == jogador)].shape[0]
    num_passesPJ = len(dfPass[(dfPass['player'] == jogador) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)])
    taxa_acertoJ = round(((num_passesCJ / num_passesJ) * 100), 1)
    dfPassJ = dfPass[(dfPass['player'] == jogador) & (dfPass['result_id'] == 1) & (~dfPass['progressive_pass'])]
    dfPassJP = dfPass[(dfPass['player'] == jogador) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)]
    num_CP_J = len(dfDribble[(dfDribble['player'] == jogador) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)])
    dfDribbleJP = dfDribble[(dfDribble['player'] == jogador) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)]
    num_passesP90J = result_df.loc[result_df['Jogador'] == jogador, 'Passes Progressivos por 90 Minutos'].values[0]
    num_conducoesP90J = result_df.loc[result_df['Jogador'] == jogador, 'Conduções Progressivas por 90 Minutos'].values[0]

    # Criar o campo
    pitch = VerticalPitch(positional=False, pitch_type='uefa', pitch_color='#222222', line_color='white', line_zorder=2)
    fig, axs = pitch.grid(ncols=1, axis=False, figheight=10, endnote_height=0, endnote_space=0, title_height=0.01, grid_height=0.75)
    
    # Gráfico com passes progressivos
    pitch.lines(dfPassJP['start_x'], dfPassJP['start_y'], dfPassJP['end_x'], dfPassJP['end_y'], color='#97c1e7',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'])
    pitch.scatter(dfPassJP['end_x'], dfPassJP['end_y'], color='black', edgecolor='#97c1e7', ax=axs['pitch'], s=50, lw=1, zorder=2)
    
    # Gráfico com conduções progressivas
    pitch.lines(dfDribbleJP['start_x'], dfDribbleJP['start_y'], dfDribbleJP['end_x'], dfDribbleJP['end_y'], color='#ff6666',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'])
    pitch.scatter(dfDribbleJP['end_x'], dfDribbleJP['end_y'], color='#ff6666', edgecolor='#343334', ax=axs['pitch'], s=50, lw=2, zorder=3)
    
    # Título
    fig.suptitle(title, fontsize=22, fontweight='bold', color='white')
    axs['pitch'].set_title(f'Ações progressivas de {jogador}', fontsize=18, color='white', pad=10)

    # Adicionar texto abaixo do subplot
    axs['pitch'].text(0.05, -0.02, f'Passes: {num_passesCJ}/{num_passesJ} ({taxa_acertoJ}% de acerto)', fontsize=14, color='white', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)
    axs['pitch'].text(0.05, -0.05, f'{num_passesPJ} passes progressivos ({num_passesP90J:.2f} por 90 min.)', fontsize=12, color='#97c1e7', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)
    axs['pitch'].text(0.05, -0.08, f'{num_CP_J} conduções progressivas ({num_conducoesP90J:.2f} por 90 min.)', fontsize=12, color='#ff6666', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)

    # Nota de rodapé
    note_text = "@Vasco_Analytics | Dados: Opta via WhoScored"
    note_text_2 = f'Ação progressiva: ponto final no mínimo 25% mais próximo do gol que o inicial'
    fig.text(-0.07, 0.01, note_text, fontsize=13, color='gray', ha='left', va='center', weight='bold')
    fig.text(-0.07, -0.015, note_text_2, fontsize=11, color='gray', ha='left', va='center')

    # Alterar a cor de fundo da figura e subplots para um tom de cinza escuro
    fig.patch.set_facecolor('#222222')  # Cor de fundo da figura

    # Salvar a figura
    fig.savefig('Mapa de ações progressivas.png', dpi=300, bbox_inches='tight', facecolor='#222222')

    plt.show()

def plot_all_actions(dfPass, dfDribble, result_df, jogador, title):
    # Dados do jogador
    num_passesCJ = dfPass[(dfPass['player'] == jogador) & (dfPass['result_id'] == 1)].shape[0]
    num_passesJ = dfPass[(dfPass['player'] == jogador)].shape[0]
    num_passesPJ = len(dfPass[(dfPass['player'] == jogador) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)])
    taxa_acertoJ = round(((num_passesCJ / num_passesJ) * 100), 1)
    dfPassJ = dfPass[(dfPass['player'] == jogador) & (dfPass['result_id'] == 1) & (~dfPass['progressive_pass'])]
    dfPassJP = dfPass[(dfPass['player'] == jogador) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)]
    num_CP_J = len(dfDribble[(dfDribble['player'] == jogador) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)])
    dfDribbleJP = dfDribble[(dfDribble['player'] == jogador) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)]
    num_passesP90J = result_df.loc[result_df['Jogador'] == jogador, 'Passes Progressivos por 90 Minutos'].values[0]
    num_conducoesP90J = result_df.loc[result_df['Jogador'] == jogador, 'Conduções Progressivas por 90 Minutos'].values[0]

    # Criar o campo
    pitch = VerticalPitch(positional=False, pitch_type='uefa', pitch_color='#222222', line_color='white', line_zorder=2)
    fig, axs = pitch.grid(ncols=1, axis=False, figheight=10, endnote_height=0, endnote_space=0, title_height=0.01, grid_height=0.75)
    
    # Gráfico com passes não progressivos
    pitch.lines(dfPassJ['start_x'], dfPassJ['start_y'],
            dfPassJ['end_x'], dfPassJ['end_y'], color='gray',
            comet=True,
            transparent=True, alpha_start=0.2, alpha_end=0.8,
            zorder=2,
            ax=axs['pitch'])

    pitch.scatter(dfPassJ['end_x'], dfPassJ['end_y'],
              color='black', edgecolor='gray', ax=axs['pitch'],
              s=50, lw=1, zorder=2)
    
    # Gráfico com passes progressivos
    pitch.lines(dfPassJP['start_x'], dfPassJP['start_y'], dfPassJP['end_x'], dfPassJP['end_y'], color='#97c1e7',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'])
    pitch.scatter(dfPassJP['end_x'], dfPassJP['end_y'], color='black', edgecolor='#97c1e7', ax=axs['pitch'], s=50, lw=1, zorder=2)
    
    # Gráfico com conduções progressivas
    pitch.lines(dfDribbleJP['start_x'], dfDribbleJP['start_y'], dfDribbleJP['end_x'], dfDribbleJP['end_y'], color='#ff6666',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'])
    pitch.scatter(dfDribbleJP['end_x'], dfDribbleJP['end_y'], color='#ff6666', edgecolor='#343334', ax=axs['pitch'], s=50, lw=2, zorder=3)
    
    # Título
    fig.suptitle(title, fontsize=22, fontweight='bold', color='white')
    axs['pitch'].set_title(f'Ações progressivas de {jogador}', fontsize=18, color='white', pad=10)

    # Adicionar texto abaixo do subplot
    axs['pitch'].text(0.05, -0.02, f'Passes: {num_passesCJ}/{num_passesJ} ({taxa_acertoJ}% de acerto)', fontsize=14, color='white', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)
    axs['pitch'].text(0.05, -0.05, f'{num_passesPJ} passes progressivos ({num_passesP90J:.2f} por 90 min.)', fontsize=12, color='#97c1e7', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)
    axs['pitch'].text(0.05, -0.08, f'{num_CP_J} conduções progressivas ({num_conducoesP90J:.2f} por 90 min.)', fontsize=12, color='#ff6666', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)

    # Nota de rodapé
    note_text = "@Vasco_Analytics | Dados: Opta via WhoScored"
    note_text_2 = f'Ação progressiva: ponto final no mínimo 25% mais próximo do gol que o inicial'
    fig.text(-0.07, 0.01, note_text, fontsize=13, color='gray', ha='left', va='center', weight='bold')
    fig.text(-0.07, -0.015, note_text_2, fontsize=11, color='gray', ha='left', va='center')

    # Alterar a cor de fundo da figura e subplots para um tom de cinza escuro
    fig.patch.set_facecolor('#222222')  # Cor de fundo da figura

    # Salvar a figura
    fig.savefig('Mapa de ações progressivas.png', dpi=300, bbox_inches='tight', facecolor='#222222')

    plt.show()


def plot_recoveries(dfRecoveries, nomes_brasileirao_sem_rep, jogador, title):
    # Filtrar apenas as recuperações bem-sucedidas do jogador
    successful_recoveries = dfRecoveries[(dfRecoveries['player'] == jogador) & (dfRecoveries['result_id'] == 1)]
    
    # Obter os minutos jogados do jogador
    mins_jogados = nomes_brasileirao_sem_rep[nomes_brasileirao_sem_rep['Jogador'] == jogador]['Mins'].values[0]
    
    # Calcular recuperações por 90 minutos
    num_recoveries = successful_recoveries.shape[0]
    recoveries_per_90 = (num_recoveries / mins_jogados) * 90

    # Criar o campo
    pitch = VerticalPitch(positional=False, pitch_type='uefa', pitch_color='#222222', line_color='white', line_zorder=2)
    fig, axs = pitch.grid(ncols=1, axis=False, figheight=10, endnote_height=0, endnote_space=0, title_height=0.01, grid_height=0.75)
    
    # Gráfico com as recuperações (desarmes e interceptações) representadas por 'X' com design melhorado
    pitch.scatter(successful_recoveries['end_x'], successful_recoveries['end_y'], 
                  marker='X', color='#0E82B5', edgecolor='#219ED6', ax=axs['pitch'], s=150, lw=1.5, zorder=3, alpha=1)
    
    # Título
    fig.suptitle(title, fontsize=18, fontweight='bold', color='white')
    axs['pitch'].set_title(f'Desarmes/interceptações do {jogador}', fontsize=18, color='white', pad=10)

    # Adicionar texto abaixo do subplot
    axs['pitch'].text(0.05, -0.02, f'{num_recoveries} recuperações ({recoveries_per_90:.2f} por 90 min)', fontsize=14, color='#ADD8E6', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)

    # Nota de rodapé
    note_text = "@Vasco_Analytics | Dados: Opta via WhoScored"
    note_text_2 = "Recuperação: desarme ou interceptação bem-sucedida"
    fig.text(0.02, 0.01, note_text, fontsize=13, color='gray', ha='left', va='center', weight='bold')
    #fig.text(0.02, -0.02, note_text_2, fontsize=11, color='gray', ha='left', va='center')

    # Alterar a cor de fundo da figura e subplots para um tom de cinza escuro
    fig.patch.set_facecolor('#222222')  # Cor de fundo da figura

    # Salvar a figura
    fig.savefig('Mapa de recuperações.png', dpi=300, bbox_inches='tight', facecolor='#222222')

    plt.show()


def plot_2_recoveries(dfRecoveries, nomes_brasileirao_sem_rep, jogador1, jogador2, title):
    # Filtrar apenas as recuperações bem-sucedidas dos jogadores
    successful_recoveriesJ1 = dfRecoveries[(dfRecoveries['player'] == jogador1) & (dfRecoveries['result_id'] == 1)]
    successful_recoveriesJ2 = dfRecoveries[(dfRecoveries['player'] == jogador2) & (dfRecoveries['result_id'] == 1)]
    
    # Obter os minutos jogados dos jogadores
    mins_jogadosJ1 = nomes_brasileirao_sem_rep[nomes_brasileirao_sem_rep['Jogador'] == jogador1]['Mins'].values[0]
    mins_jogadosJ2 = nomes_brasileirao_sem_rep[nomes_brasileirao_sem_rep['Jogador'] == jogador2]['Mins'].values[0]
    
    # Calcular recuperações por 90 minutos
    num_recoveriesJ1 = successful_recoveriesJ1.shape[0]
    recoveries_per_90J1 = (num_recoveriesJ1 / mins_jogadosJ1) * 90
    
    num_recoveriesJ2 = successful_recoveriesJ2.shape[0]
    recoveries_per_90J2 = (num_recoveriesJ2 / mins_jogadosJ2) * 90

    # Criar o campo
    pitch = VerticalPitch(positional=False, pitch_type='uefa', pitch_color='#222222', line_color='white', line_zorder=2)
    fig, axs = pitch.grid(ncols=2, axis=False, figheight=10, endnote_height=0, endnote_space=0, title_height=0.01, grid_height=0.75)
    
    # Gráfico do jogador 1
    pitch.scatter(successful_recoveriesJ1['end_x'], successful_recoveriesJ1['end_y'], 
                  marker='X', color='#0E82B5', edgecolor='#219ED6', ax=axs['pitch'][0], s=150, lw=1.5, zorder=3, alpha=1)
    
    # Gráfico do jogador 2
    pitch.scatter(successful_recoveriesJ2['end_x'], successful_recoveriesJ2['end_y'], 
                  marker='X', color='#0E82B5', edgecolor='#219ED6', ax=axs['pitch'][1], s=150, lw=1.5, zorder=3, alpha=1)
    
    # Títulos e textos
    fig.suptitle(title, fontsize=20, fontweight='bold', color='white')
    axs['pitch'][0].set_title(f'Desarmes/Interceptações do {jogador1}', fontsize=16, color='white', pad=1)
    axs['pitch'][1].set_title(f'Desarmes/Interceptações do {jogador2}', fontsize=16, color='white', pad=1)

    axs['pitch'][0].text(0.05, -0.02, f'{num_recoveriesJ1} recuperações ({recoveries_per_90J1:.2f} por 90 min)', fontsize=14, color='#ADD8E6', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'][0].transAxes)
    axs['pitch'][1].text(0.05, -0.02, f'{num_recoveriesJ2} recuperações ({recoveries_per_90J2:.2f} por 90 min)', fontsize=14, color='#ADD8E6', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'][1].transAxes)

    # Nota de rodapé
    note_text = "@Vasco_Analytics | Dados: Opta via WhoScored"
    fig.text(0.02, 0.01, note_text, fontsize=16, color='#ababab', ha='left', va='center', weight='300')

    # Alterar a cor de fundo da figura e subplots para um tom de cinza escuro
    fig.patch.set_facecolor('#222222')  # Cor de fundo da figura

    # Salvar a figura
    fig.savefig('Mapa de desarmes interceptações.png', dpi=300, bbox_inches='tight', facecolor='#222222')

    plt.show()

def plot_progressive_passes(dfPass, result_df, jogador, title):
    # Dados do jogador
    num_passesCJ = dfPass[(dfPass['player'] == jogador) & (dfPass['result_id'] == 1)].shape[0]
    num_passesJ = dfPass[(dfPass['player'] == jogador)].shape[0]
    num_passesPJ = len(dfPass[(dfPass['player'] == jogador) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)])
    taxa_acertoJ = round(((num_passesCJ / num_passesJ) * 100), 1)
    dfPassJ = dfPass[(dfPass['player'] == jogador) & (dfPass['result_id'] == 1) & (~dfPass['progressive_pass'])]
    dfPassJP = dfPass[(dfPass['player'] == jogador) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)]    
    num_passesP90J = result_df.loc[result_df['Jogador'] == jogador, 'Passes Progressivos por 90 Minutos'].values[0]
    
    # Criar o campo
    pitch = VerticalPitch(positional=False, pitch_type='uefa', pitch_color='#222222', line_color='white', line_zorder=2)
    fig, axs = pitch.grid(ncols=1, axis=False, figheight=10, endnote_height=0, endnote_space=0, title_height=0.01, grid_height=0.75)
    
    # Gráfico com passes progressivos
    pitch.lines(dfPassJP['start_x'], dfPassJP['start_y'], dfPassJP['end_x'], dfPassJP['end_y'], color='#97c1e7',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'])
    pitch.scatter(dfPassJP['end_x'], dfPassJP['end_y'], color='black', edgecolor='#97c1e7', ax=axs['pitch'], s=50, lw=1, zorder=2)
    
    # Título
    fig.suptitle(title, fontsize=22, fontweight='bold', color='white')
    axs['pitch'].set_title(f'Passes progressivos do {jogador}', fontsize=18, color='white', pad=10)

    # Adicionar texto abaixo do subplot
    axs['pitch'].text(0.05, -0.02, f'Passes: {num_passesCJ}/{num_passesJ} ({taxa_acertoJ}% de acerto)', fontsize=14, color='white', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)
    axs['pitch'].text(0.05, -0.05, f'{num_passesPJ} passes progressivos ({num_passesP90J:.2f} por 90 min.)', fontsize=12, color='#97c1e7', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)

    # Nota de rodapé
    note_text = "@Vasco_Analytics | Dados: Opta via WhoScored"
    note_text_2 = f'Ação progressiva: ponto final no mínimo 25% mais próximo do gol que o inicial'
    fig.text(-0.07, 0.01, note_text, fontsize=13, color='gray', ha='left', va='center', weight='bold')
    fig.text(-0.07, -0.015, note_text_2, fontsize=11, color='gray', ha='left', va='center')

    # Alterar a cor de fundo da figura e subplots para um tom de cinza escuro
    fig.patch.set_facecolor('#222222')  # Cor de fundo da figura

    # Salvar a figura
    fig.savefig('Mapa de passes progressivos.png', dpi=300, bbox_inches='tight', facecolor='#222222')

    plt.show()


def plot_progressive_dribbles(dfDribble, result_df, jogador, title):
    # Dados do jogador
    
    num_CP_J = len(dfDribble[(dfDribble['player'] == jogador) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)])
    dfDribbleJP = dfDribble[(dfDribble['player'] == jogador) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)]
    num_conducoesP90J = result_df.loc[result_df['Jogador'] == jogador, 'Conduções Progressivas por 90 Minutos'].values[0]

    # Criar o campo
    pitch = VerticalPitch(positional=False, pitch_type='uefa', pitch_color='#222222', line_color='white', line_zorder=2)
    fig, axs = pitch.grid(ncols=1, axis=False, figheight=10, endnote_height=0, endnote_space=0, title_height=0.01, grid_height=0.75)
    
    # Gráfico com conduções progressivas
    pitch.lines(dfDribbleJP['start_x'], dfDribbleJP['start_y'], dfDribbleJP['end_x'], dfDribbleJP['end_y'], color='#ff6666',
                comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=axs['pitch'])
    pitch.scatter(dfDribbleJP['end_x'], dfDribbleJP['end_y'], color='#ff6666', edgecolor='#343334', ax=axs['pitch'], s=50, lw=2, zorder=3)
    
    # Título
    fig.suptitle(title, fontsize=22, fontweight='bold', color='white')
    axs['pitch'].set_title(f'Conduções progressivas do {jogador}', fontsize=18, color='white', pad=10)

    # Adicionar texto abaixo do subplot
    axs['pitch'].text(0.05, -0.02, f'{num_CP_J} conduções progressivas ({num_conducoesP90J:.2f} por 90 min.)', fontsize=12, color='#ff6666', ha='left', va='bottom', fontweight='bold', transform=axs['pitch'].transAxes)

    # Nota de rodapé
    note_text = "@Vasco_Analytics | Dados: Opta via WhoScored"
    note_text_2 = f'Ação progressiva: ponto final no mínimo 25% mais próximo do gol que o inicial'
    fig.text(-0.07, 0.01, note_text, fontsize=13, color='gray', ha='left', va='center', weight='bold')
    fig.text(-0.07, -0.015, note_text_2, fontsize=11, color='gray', ha='left', va='center')

    # Alterar a cor de fundo da figura e subplots para um tom de cinza escuro
    fig.patch.set_facecolor('#222222')  # Cor de fundo da figura

    # Salvar a figura
    fig.savefig('Mapa de conduções progressivas.png', dpi=300, bbox_inches='tight', facecolor='#222222')

    plt.show()


def plot_multiple_progressive_actions(dfPass, dfDribble, result_df, jogadores, title):
    assert len(jogadores) == 12, "A lista de jogadores deve conter exatamente 12 jogadores."

    # Criar o campo
    pitch = Pitch(positional=False, pitch_type='uefa', pitch_color='#222222', line_color='white', line_zorder=2)
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(23, 18))
    axs = axs.flatten()

    for i, jogador in enumerate(jogadores):
        # Dados do jogador
        num_passesCJ = dfPass[(dfPass['player'] == jogador) & (dfPass['result_id'] == 1)].shape[0]
        num_passesJ = dfPass[(dfPass['player'] == jogador)].shape[0]
        num_passesPJ = len(dfPass[(dfPass['player'] == jogador) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)])
        taxa_acertoJ = round(((num_passesCJ / num_passesJ) * 100), 1)
        dfPassJ = dfPass[(dfPass['player'] == jogador) & (dfPass['result_id'] == 1) & (~dfPass['progressive_pass'])]
        dfPassJP = dfPass[(dfPass['player'] == jogador) & (dfPass['progressive_pass']) & (dfPass['result_id'] == 1)]
        num_CP_J = len(dfDribble[(dfDribble['player'] == jogador) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)])
        dfDribbleJP = dfDribble[(dfDribble['player'] == jogador) & (dfDribble['progressive_dribble']) & (dfDribble['result_id'] == 1)]
        num_passesP90J = result_df.loc[result_df['Jogador'] == jogador, 'Passes Progressivos por 90 Minutos'].values[0]
        num_conducoesP90J = result_df.loc[result_df['Jogador'] == jogador, 'Conduções Progressivas por 90 Minutos'].values[0]

        ax = axs[i]
        pitch.draw(ax=ax)

        # Gráfico com passes progressivos
        pitch.lines(dfPassJP['start_x'], dfPassJP['start_y'], dfPassJP['end_x'], dfPassJP['end_y'], color='#97c1e7',
                    comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=ax)
        pitch.scatter(dfPassJP['end_x'], dfPassJP['end_y'], color='black', edgecolor='#97c1e7', ax=ax, s=50, lw=1, zorder=2)
        
        # Gráfico com conduções progressivas
        pitch.lines(dfDribbleJP['start_x'], dfDribbleJP['start_y'], dfDribbleJP['end_x'], dfDribbleJP['end_y'], color='#ff6666',
                    comet=True, transparent=True, alpha_start=0.2, alpha_end=0.8, zorder=2, ax=ax)
        pitch.scatter(dfDribbleJP['end_x'], dfDribbleJP['end_y'], color='#ff6666', edgecolor='#343334', ax=ax, s=50, lw=2, zorder=3)

        ax.set_title(f'{jogador}', fontsize=23, color='white', pad=10, weight='bold')

        # Adicionar texto abaixo de cada subplot
        ax.text(0.5, -0.05, f'{(num_passesP90J + num_conducoesP90J):.2f} Ações Progressivas por 90\'', fontsize=17, color='#ADD8E6', ha='center', va='center', fontweight='bold', transform=ax.transAxes)

    # Título
    fig.suptitle(title, fontsize=29, fontweight='bold', color='white', x=0.02, ha='left')

    # Subtítulo colorido
    fig.text(0.02, 0.930, "Passes Progressivos", fontsize=20, color='#97c1e7', ha='left', weight='bold')
    fig.text(0.161, 0.930, "e", fontsize=20, color='white', ha='left', weight='bold')
    fig.text(0.174, 0.930, "Conduções Progressivas", fontsize=20, color='#ff6666', ha='left', weight='bold')
    fig.text(0.346, 0.930, "| Apenas jogadores com + de 900 minutos disputados", fontsize=20, color='white', ha='left', weight='bold')

    # Nota de rodapé
    note_text = "@Vasco_Analytics | Dados: Opta via WhoScored | Ação progressiva: ponto final no mínimo 25% mais próximo do gol adversário ou passe para a área"
    fig.text(0.02, 0.01, note_text, fontsize=18, color='white', ha='left', va='center', weight='bold')

    # Alterar a cor de fundo da figura e subplots para um tom de cinza escuro
    fig.patch.set_facecolor('#222222')  # Cor de fundo da figura

    # Ajustar layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Salvar a figura
    fig.savefig('Mapa de ações progressivas múltiplas.png', dpi=300, bbox_inches='tight', facecolor='#222222')

    plt.show()


def get_top_players(df, sort_column, players_quantity):
    """
    Retorna uma lista com os 12 melhores jogadores com base na coluna especificada para ordenação.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados dos jogadores.
    sort_column (str): Nome da coluna usada para ordenar os jogadores.

    Retorna:
    list: Lista contendo os nomes dos 12 melhores jogadores.
    """
    # Ordenar o DataFrame pela coluna especificada
    sorted_df = df.sort_values(by=sort_column, ascending=False)
    
    # Obter os nomes dos 12 melhores jogadores
    top_players = sorted_df['Jogador'].head(players_quantity).tolist()
    
    return top_players