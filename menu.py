import sqlite3
import os
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
from gerar_tabela_rostos import gerar_tabela_rostos
from gerar_planilha_presenca import exibir_planilha_em_janela
from visualizacao_modelo import visualizar_analise_modelo
from db import garantir_estrutura_rostos

DB_NAME = 'rostos.db'

COLUNAS_EDITAVEIS = ['nome', 'numero_imagem', 'turma', 'data_imagem']
COLUNAS_EXCLUIVEIS = ['id_rosto', 'nome', 'numero_imagem', 'turma', 'data_imagem']

def exibir_rosto_blob(rosto_blob, titulo):
    """Mostra um rosto armazenado em BLOB para facilitar a identificação manual."""
    if rosto_blob is None:
        print('Imagem do rosto indisponível para este registro.')
        return False

    img_array = np.frombuffer(rosto_blob, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print('Não foi possível decodificar a imagem deste rosto.')
        return False

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    # plt.show() é bloqueante: só retorna quando a janela for fechada.
    plt.show()
    return True


def pesquisar():
    termo = input('Digite o termo para pesquisar: ')
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Colunas relevantes para pesquisa (ignorando rosto/BLOB)
    colunas = ['ord_id', 'id_rosto', 'nome', 'numero_imagem', 'turma', 'data_imagem']
    query = f"SELECT {', '.join(colunas)} FROM rostos"
    cursor.execute(query)
    linhas = cursor.fetchall()
    # Pesquisa apenas nas colunas textuais/relevantes
    resultados = [linha for linha in linhas if any(termo.lower() in str(valor).lower() for valor in linha if valor is not None)]
    print('\nResultados encontrados:')
    if resultados:
        # Exibir como tabela simples
        print(' | '.join(colunas))
        for linha in resultados:
            print(' | '.join(str(valor) if valor is not None else '' for valor in linha))
    else:
        print('Nenhum resultado encontrado.')
    conn.close()


def editar_dado():
    ord_id = input('Digite o ord_id da linha para editar: ')
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query('SELECT * FROM rostos WHERE ord_id = ?', conn, params=(ord_id,))
    if df.empty:
        print('ord_id não encontrado.')
        conn.close()
        return
    print('Linha atual:')
    print(df)
    print('Colunas editáveis:', COLUNAS_EDITAVEIS)
    coluna = input('Qual coluna deseja editar? ')
    if coluna not in COLUNAS_EDITAVEIS:
        print('Coluna não pode ser editada.')
        conn.close()
        return
    novo_valor = input(f'Novo valor para {coluna}: ')
    cursor = conn.cursor()
    cursor.execute(f'UPDATE rostos SET {coluna} = ? WHERE ord_id = ?', (novo_valor, ord_id))
    conn.commit()
    conn.close()
    print('Dado atualizado com sucesso.')


def excluir_memoria():
    print('Opções de exclusão:')
    print('1 - Excluir item por valor (id_rosto, nome, numero_imagem, turma, data_imagem)')
    print('2 - Excluir tudo')
    opcao = input('Escolha a opção: ')
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if opcao == '1':
        coluna = input('Qual campo deseja usar para exclusão? ')
        if coluna not in COLUNAS_EXCLUIVEIS:
            print('Campo não pode ser usado para exclusão.')
            conn.close()
            return
        valor = input(f'Valor exato do item ({coluna}) para exclusão: ')
        df = pd.read_sql_query(f'SELECT * FROM rostos WHERE {coluna} = ?', conn, params=(valor,))
        print('Linha(s) que será(ão) excluída(s):')
        print(df)
        confirm = input('Confirmar exclusão? (s/n): ')
        if confirm.lower() == 's':
            cursor.execute(f'DELETE FROM rostos WHERE {coluna} = ?', (valor,))
            conn.commit()
            print('Exclusão realizada.')
    elif opcao == '2':
        df = pd.read_sql_query('SELECT * FROM rostos', conn)
        print('Todas as linhas que serão excluídas:')
        print(df)
        confirm = input('Confirmar exclusão de tudo? (s/n): ')
        if confirm.lower() == 's':
            cursor.execute('DELETE FROM rostos')
            conn.commit()
            print('Todas as memórias foram excluídas.')
    else:
        print('Opção inválida.')
    conn.close()


def dar_nome_ao_rosto():
    conn = sqlite3.connect(DB_NAME)
    garantir_estrutura_rostos(conn)
    cursor = conn.cursor()

    query = '''
        SELECT ord_id, rosto_embeddings, id_rosto, numero_imagem, turma, data_imagem
        FROM rostos
        WHERE nome IS NULL OR TRIM(nome) = ''
        ORDER BY ord_id ASC
    '''
    cursor.execute(query)
    rostos_sem_nome = cursor.fetchall()

    if not rostos_sem_nome:
        print('Não há rostos sem nome para identificar.')
        conn.close()
        return

    print('\nRostos sem nome (ordem por ID):')
    print('ord_id | id_rosto | numero_imagem | turma | data_imagem')

    for rosto in rostos_sem_nome:
        ord_id, rosto_blob, id_rosto, numero_imagem, turma, data_imagem = rosto
        print(f'{ord_id} | {id_rosto} | {numero_imagem} | {turma} | {data_imagem}')
        titulo_janela = f'ID: {id_rosto} | Turma: {turma} | Data: {data_imagem}'
        exibir_rosto_blob(rosto_blob, titulo_janela)

        novo_nome = input('Digite o nome do rosto: ').strip()
        if not novo_nome:
            print('Nome vazio. Registro mantido sem nome.')
        else:
            cursor.execute(
                'UPDATE rostos SET nome = ?, origem_nome = ? WHERE ord_id = ?',
                (novo_nome, 'manual', ord_id),
            )

            conn.commit()
            print(f'nome {novo_nome} adicionado ao rosto {id_rosto}')

        continuar = input('Deseja ir para o próximo rosto sem nome? (s/n): ').strip().lower()
        if continuar not in {'s', 'sim', 'y', 'yes'}:
            print('Operação encerrada pelo usuário.')
            break

    conn.close()
    print('Processo de nomeação concluído.')


def exibir_tabela_rostos():
    tabela = gerar_tabela_rostos()
    if tabela.empty:
        print('Nenhum rosto cadastrado na tabela.')
        return
    print('\nTabela de rostos:')
    print(tabela.to_string(index=False))


def exibir_tabela_consolidada_por_nome():
    conn = sqlite3.connect(DB_NAME)
    garantir_estrutura_rostos(conn)
    query = '''
        SELECT
            CASE
                WHEN nome IS NULL OR TRIM(nome) = '' THEN 'Sem nome'
                ELSE TRIM(nome)
            END AS nome,
            COUNT(*) AS total_registros,
            COUNT(DISTINCT numero_imagem) AS total_imagens,
            SUM(CASE WHEN origem_nome = 'automatico' THEN 1 ELSE 0 END) AS reconhecidos_automaticamente,
            SUM(CASE WHEN origem_nome = 'manual' THEN 1 ELSE 0 END) AS nomeados_manualmente,
            SUM(CASE WHEN origem_nome IS NULL OR TRIM(origem_nome) = '' THEN 1 ELSE 0 END) AS origem_desconhecida,
            ROUND((100.0 * SUM(CASE WHEN origem_nome = 'automatico' THEN 1 ELSE 0 END)) / COUNT(*), 1) AS taxa_auto_percentual,
            MIN(data_imagem) AS primeira_ocorrencia,
            MAX(data_imagem) AS ultima_ocorrencia
        FROM rostos
        GROUP BY
            CASE
                WHEN nome IS NULL OR TRIM(nome) = '' THEN 'Sem nome'
                ELSE TRIM(nome)
            END
        ORDER BY total_registros DESC, nome ASC
    '''
    tabela = pd.read_sql_query(query, conn)
    conn.close()

    if tabela.empty:
        print('Nenhum rosto cadastrado para consolidar.')
        return

    pasta_planilhas = os.path.join(os.path.dirname(__file__), 'planilhas')
    os.makedirs(pasta_planilhas, exist_ok=True)

    caminho_csv = os.path.join(pasta_planilhas, 'tabela_consolidada_por_nome.csv')
    tabela.to_csv(caminho_csv, index=False, encoding='utf-8-sig')
    print(f'Planilha CSV consolidada salva em: {caminho_csv}')

    caminho_xlsx = os.path.join(pasta_planilhas, 'tabela_consolidada_por_nome.xlsx')
    try:
        tabela.to_excel(caminho_xlsx, index=False)
        print(f'Planilha XLSX consolidada salva em: {caminho_xlsx}')
    except Exception as exc:
        print(
            'Aviso: não foi possível gerar XLSX '
            f'({exc}). O CSV foi salvo normalmente.'
        )

    exibir_planilha_em_janela(tabela, titulo='Tabela Consolidada por Nome')

    print('\nTabela consolidada por nome:')
    print(tabela.to_string(index=False))


def criar_planilha():
    """Submenu para gerar diferentes tipos de planilhas."""
    while True:
        print('\n--- Criar Planilha ---')
        print('1 - Gerar planilha de presença')
        print('2 - Gerar tabela de presença')
        print('3 - Gerar tabela de rostos')
        print('4 - Análise técnica do modelo')
        print('5 - Voltar ao menu principal')
        opcao = input('Escolha a opção: ').strip()
        
        if opcao == '1':
            try:
                import gerar_planilha_presenca
                print('\nExecutando gerar_planilha_presenca.py...')
                if hasattr(gerar_planilha_presenca, 'main'):
                    gerar_planilha_presenca.main()
                else:
                    print('Função main() não encontrada em gerar_planilha_presenca.py')
            except Exception as e:
                print(f'Erro ao executar gerar_planilha_presenca.py: {e}')
        
        elif opcao == '2':
            try:
                import gerar_tabela_presença
                print('\nExecutando gerar_tabela_presença.py...')
                if hasattr(gerar_tabela_presença, 'main'):
                    gerar_tabela_presença.main()
                else:
                    print('Função main() não encontrada em gerar_tabela_presença.py')
            except Exception as e:
                print(f'Erro ao executar gerar_tabela_presença.py: {e}')
        
        elif opcao == '3':
            try:
                print('\nExecutando gerar_tabela_rostos.py...')
                tabela = gerar_tabela_rostos()
                if not tabela.empty:
                    print('\nTabela de rostos gerada:')
                    print(tabela.to_string(index=False))
                else:
                    print('Nenhum rosto encontrado.')
            except Exception as e:
                print(f'Erro ao executar gerar_tabela_rostos.py: {e}')
        
        elif opcao == '4':
            try:
                import analise_tecnica_modelo
                print('\nExecutando analise_tecnica_modelo.py...')
                if hasattr(analise_tecnica_modelo, 'main'):
                    analise_tecnica_modelo.main()
                else:
                    print('Função main() não encontrada em analise_tecnica_modelo.py')
            except Exception as e:
                print(f'Erro ao executar analise_tecnica_modelo.py: {e}')
        
        elif opcao == '5':
            print('Voltando ao menu principal.')
            break
        
        else:
            print('Opção inválida.')


def visualizar_imagem_cadastrada():
    pasta_imagens = 'imagens_salvas'
    extensoes_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if not os.path.isdir(pasta_imagens):
        print(f'Pasta não encontrada: {pasta_imagens}')
        return

    imagens = sorted(
        [
            nome for nome in os.listdir(pasta_imagens)
            if os.path.isfile(os.path.join(pasta_imagens, nome))
            and os.path.splitext(nome)[1].lower() in extensoes_validas
        ]
    )

    if not imagens:
        print('Nenhuma imagem cadastrada encontrada em imagens_salvas.')
        return

    print('\nImagens cadastradas em imagens_salvas:')
    for i, nome_arquivo in enumerate(imagens, start=1):
        print(f'{i} - {nome_arquivo}')

    escolha = input('Digite o número da imagem que deseja visualizar: ').strip()
    if not escolha.isdigit():
        print('Seleção inválida. Digite apenas o número da imagem.')
        return

    indice = int(escolha)
    if indice < 1 or indice > len(imagens):
        print('Número fora da lista de imagens cadastradas.')
        return

    imagem_escolhida = imagens[indice - 1]
    caminho_imagem = os.path.join(pasta_imagens, imagem_escolhida)
    print(f'Visualizando imagem: {imagem_escolhida}')
    visualizar_analise_modelo(caminho_imagem, modelo='opencv')


def main():
    while True:
        print('\n--- Menu ---')
        print('1 - Pesquisar')
        print('2 - Editar dado')
        print('3 - Excluir memória')
        print('4 - Dar nome ao rosto')
        print('5 - Exibir tabela dos rostos')
        print('6 - Exibir tabela consolidada por nome')
        print('7 - Criar planilha')
        print('8 - Visualizar imagem cadastrada (análise do modelo)')
        print('9 - Concluir')
        opcao = input('Escolha a opção: ')
        if opcao == '1':
            pesquisar()
        elif opcao == '2':
            editar_dado()
        elif opcao == '3':
            excluir_memoria()
        elif opcao == '4':
            dar_nome_ao_rosto()
        elif opcao == '5':
            exibir_tabela_rostos()
        elif opcao == '6':
            exibir_tabela_consolidada_por_nome()
        elif opcao == '7':
            criar_planilha()
        elif opcao == '8':
            visualizar_imagem_cadastrada()
        elif opcao == '9':
            print('Encerrando Menu.')
            break
        else:
            print('Opção inválida.')

if __name__ == '__main__':
    main()
