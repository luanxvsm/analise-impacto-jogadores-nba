import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise de Impacto NBA",
    page_icon="🏀",
    layout="wide"
)

# --- Dicionário de Tradução de Colunas ---
nomes_colunas = {
    'Player': 'Jogador',
    'Posicao': 'Posição', 
    'Time': 'Time',
    'Jogos': 'Jogos',
    'MP': 'Minutos por Jogo',
    'Pontos': 'Pontos por Jogo', 
    'TRB': 'Rebotes por Jogo',
    'AST': 'Assistências por Jogo',
    'STL': 'Roubos de Bola',
    'BLK': 'Tocos',
    'FGA': 'Arremessos Tentados',
    'eFG%': 'Aproveitamento Efetivo (%)'
}

# Dicionário para traduzir as posições
traducao_posicoes = {
    'C': 'Pivô',
    'PF': 'Ala-Pivô',
    'SF': 'Ala',
    'SG': 'Ala-Armador',
    'PG': 'Armador'
}

# --- Carregamento dos Dados ---
@st.cache_data
def carregar_dados():
    df = pd.read_csv('nba_dados_limpos.csv')
    df_renomeado = df.rename(columns=nomes_colunas)
    df_renomeado['Posição'] = df_renomeado['Posição'].map(traducao_posicoes)
    return df_renomeado

df = carregar_dados()

# --- Barra Lateral (Sidebar) com Filtros ---
st.sidebar.header("Filtros")

# Filtro de Time (mantido)
times = sorted(df['Time'].unique())
time_selecionado = st.sidebar.selectbox("Selecione um Time:", options=['Todos'] + times)

# Lógica de filtragem (agora apenas por time)
df_filtrado = df.copy() 

if time_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Time'] == time_selecionado]

# --- Título do Dashboard ---
st.title("🏀 Análise de Impacto de Jogadores da NBA")
st.markdown("Dashboard interativo para explorar o impacto dos principais jogadores, combinando análise guiada com a interatividade do Plotly.")

# --- Seção 1: Identificando a Elite da Liga ---
st.header("1. Quem são os Líderes da Liga?")

opcoes_stats = ['Pontos por Jogo', 'Rebotes por Jogo', 'Assistências por Jogo', 'Roubos de Bola', 'Tocos']
stat_selecionada = st.selectbox(
    "Selecione uma estatística para ver os líderes:",
    options=opcoes_stats
)

if not df_filtrado.empty:
    st.subheader(f"Top 10 Jogadores em {stat_selecionada}")
    top_10 = df_filtrado.nlargest(10, stat_selecionada)

    # Este gráfico agora será filtrado apenas por time
    fig1 = px.bar(
        top_10,
        x=stat_selecionada,
        y='Jogador',
        orientation='h',
        title=f"Top 10 em {stat_selecionada}",
        text=stat_selecionada
    )
    fig1.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Nenhum jogador encontrado com os filtros selecionados.")


# --- Seção 2: O Dilema da Eficiência vs. Volume ---
st.header("2. Análise de Eficiência vs. Volume")
st.markdown("""
Este gráfico é crucial para nossa tese. Ele mostra que nem todo grande pontuador é eficiente.
- **Quadrante Superior Direito (Superestrelas):** Alto volume com alta eficiência.
- **Quadrante Inferior Direito (Ineficientes):** Alto volume, mas com baixo aproveitamento.
- **Quadrante Superior Esquerdo (Especialistas):** Baixo volume, mas muito eficientes.
**Clique na legenda à direita para filtrar as posições diretamente no gráfico!**
""")

if not df_filtrado.empty:
    df_relevante = df_filtrado[df_filtrado['Arremessos Tentados'] > df_filtrado['Arremessos Tentados'].quantile(0.25)] 

    if not df_relevante.empty:
        fig2 = px.scatter(
            df_relevante,
            x='Arremessos Tentados',
            y='Aproveitamento Efetivo (%)',
            color='Posição',
            size='Pontos por Jogo',
            hover_name='Jogador',
            hover_data=['Time', 'Pontos por Jogo', 'Jogos'],
            title='Eficiência de Arremesso vs. Volume de Tentativas'
        )
        median_x = df_relevante['Arremessos Tentados'].median()
        median_y = df_relevante['Aproveitamento Efetivo (%)'].median()
        fig2.add_vline(x=median_x, line_dash="dash", line_color="red", annotation_text="Mediana de Volume")
        fig2.add_hline(y=median_y, line_dash="dash", line_color="green", annotation_text="Mediana de Eficiência")
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Não há jogadores suficientes para exibir o gráfico de eficiência com os filtros atuais.")


# --- Seção 3: O Perfil do Jogador de Impacto (Gráfico de Radar) ---
st.header("3. Perfil de Versatilidade do Jogador")

if not df_filtrado.empty:
    # A lista de jogadores agora será filtrada apenas por time
    jogadores = sorted(df_filtrado['Jogador'].unique())
    jogador_selecionado = st.selectbox("Selecione um jogador para uma análise detalhada:", options=jogadores)

    if jogador_selecionado:
        dados_jogador = df_filtrado[df_filtrado['Jogador'] == jogador_selecionado].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pontos por Jogo", f"{dados_jogador['Pontos por Jogo']:.1f}")
        col2.metric("Rebotes por Jogo", f"{dados_jogador['Rebotes por Jogo']:.1f}")
        col3.metric("Assistências por Jogo", f"{dados_jogador['Assistências por Jogo']:.1f}")
        col4.metric("Aproveitamento Efetivo (%)", f"{dados_jogador['Aproveitamento Efetivo (%)']:.3f}")

        stats_radar = ['Pontos por Jogo', 'Rebotes por Jogo', 'Assistências por Jogo', 'Roubos de Bola', 'Tocos']
        scaler = MinMaxScaler()
        df_radar_base = df[stats_radar]
        df_radar_norm = pd.DataFrame(scaler.fit_transform(df_radar_base), columns=stats_radar)
        df_radar_norm['Jogador'] = df['Jogador']

        valores_jogador_norm = df_radar_norm[df_radar_norm['Jogador'] == jogador_selecionado].iloc[0][stats_radar].values
        valores_jogador_norm = np.concatenate((valores_jogador_norm, [valores_jogador_norm[0]]))

        labels = np.array(stats_radar)
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        fig3 = go.Figure()
        fig3.add_trace(go.Scatterpolar(
              r=valores_jogador_norm,
              theta=stats_radar,
              fill='toself',
              name=jogador_selecionado
        ))
        fig3.update_layout(
          polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
          showlegend=False,
          title=f'Perfil de Versatilidade de {jogador_selecionado}'
        )
        st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.write("Desenvolvido com base em análises de dados da NBA.")