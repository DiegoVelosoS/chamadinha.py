"""
Análise Técnica do Modelo de Reconhecimento Facial

Este script realiza uma análise completa do modelo de reconhecimento facial,
incluindo métricas, gráficos e relatórios detalhados.
"""

import sqlite3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
)
from collections import Counter, defaultdict
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import face_recognition
except ImportError:
    face_recognition = None
    print("ERRO: Pacote 'face_recognition' não instalado. Execute: pip install face-recognition")
    exit(1)

DB_NAME = 'rostos.db'
EMBEDDING_TOLERANCIA = 0.55
EMBEDDING_MARGEM_AMBIGUIDADE = 0.03


def _blob_para_imagem(rosto_blob):
    """Converte BLOB do banco de dados em imagem numpy."""
    if rosto_blob is None:
        return None
    arr = np.frombuffer(rosto_blob, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def calcular_embedding_rosto(rosto_bgr):
    """Gera embedding facial (128D) para comparação entre rostos."""
    if rosto_bgr is None or rosto_bgr.size == 0:
        return None

    h, w = rosto_bgr.shape[:2]
    variantes = [rosto_bgr]

    if min(h, w) < 160:
        ampliada = cv2.resize(rosto_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variantes.append(ampliada)

    ycrcb = cv2.cvtColor(rosto_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y_channel)
    eq_bgr = cv2.cvtColor(cv2.merge([y_eq, cr_channel, cb_channel]), cv2.COLOR_YCrCb2BGR)
    variantes.append(eq_bgr)

    for variante in variantes:
        rosto_rgb = cv2.cvtColor(variante, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rosto_rgb, model='small')
        if encodings:
            return encodings[0].astype(np.float32)

    return None


def reconhecer_nome_por_embedding(embedding, referencias, tolerancia=EMBEDDING_TOLERANCIA):
    """Retorna o nome mais próximo quando a distância estiver abaixo da tolerância."""
    if embedding is None or not referencias:
        return None, None, None

    melhores_por_nome = {}
    for ref in referencias:
        distancia = float(np.linalg.norm(ref['embedding'] - embedding))
        nome = ref['nome']
        atual = melhores_por_nome.get(nome)
        if atual is None or distancia < atual:
            melhores_por_nome[nome] = distancia

    ranking = sorted((dist, nome) for nome, dist in melhores_por_nome.items())
    menor_distancia, melhor_nome = ranking[0]
    segunda_distancia = ranking[1][0] if len(ranking) > 1 else None

    # Verificar ambiguidade
    ambiguo = False
    if segunda_distancia is not None:
        margem = segunda_distancia - menor_distancia
        if margem < EMBEDDING_MARGEM_AMBIGUIDADE:
            ambiguo = True

    if menor_distancia <= tolerancia and not ambiguo:
        return melhor_nome, menor_distancia, False  # reconhecido, não ambíguo
    elif ambiguo:
        return None, menor_distancia, True  # não reconhecido por ambiguidade
    else:
        return None, menor_distancia, False  # não reconhecido por distância


def carregar_todos_rostos_nomeados():
    """Carrega todos os rostos com nome do banco de dados."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT ord_id, nome, rosto_embeddings, id_rosto, turma, data_imagem
        FROM rostos
        WHERE nome IS NOT NULL
          AND TRIM(nome) <> ''
          AND rosto_embeddings IS NOT NULL
        ORDER BY nome, ord_id
    ''')
    
    resultados = cursor.fetchall()
    conn.close()
    
    rostos_por_pessoa = defaultdict(list)
    
    for ord_id, nome, rosto_blob, id_rosto, turma, data_imagem in resultados:
        rosto_img = _blob_para_imagem(rosto_blob)
        embedding = calcular_embedding_rosto(rosto_img)
        
        if embedding is not None and embedding.size == 128:
            rostos_por_pessoa[nome].append({
                'ord_id': ord_id,
                'nome': nome,
                'embedding': embedding,
                'id_rosto': id_rosto,
                'turma': turma,
                'data_imagem': data_imagem
            })
    
    return rostos_por_pessoa


def executar_validacao_cruzada():
    """
    Executa validação leave-one-out: para cada rosto de uma pessoa,
    tenta reconhecê-lo usando os outros rostos como base.
    """
    print("=" * 80)
    print("ANÁLISE TÉCNICA DO MODELO DE RECONHECIMENTO FACIAL")
    print("=" * 80)
    print("\nCarregando dados do banco...", end=" ")
    
    rostos_por_pessoa = carregar_todos_rostos_nomeados()
    
    # Filtrar pessoas com pelo menos 2 rostos
    rostos_por_pessoa_filtrado = {
        nome: rostos for nome, rostos in rostos_por_pessoa.items() 
        if len(rostos) >= 2
    }
    
    # Também considerar pessoas com apenas 1 rosto (para testar rejeição)
    pessoas_unico_rosto = {
        nome: rostos for nome, rostos in rostos_por_pessoa.items() 
        if len(rostos) == 1
    }
    
    total_pessoas = len(rostos_por_pessoa)
    pessoas_com_multiplos = len(rostos_por_pessoa_filtrado)
    pessoas_rosto_unico = len(pessoas_unico_rosto)
    total_rostos = sum(len(rostos) for rostos in rostos_por_pessoa.values())
    
    print(f"✓")
    print(f"\nEstatísticas do banco de dados:")
    print(f"  - Total de pessoas cadastradas: {total_pessoas}")
    print(f"  - Pessoas com 2+ rostos (testáveis): {pessoas_com_multiplos}")
    print(f"  - Pessoas com 1 rosto apenas: {pessoas_rosto_unico}")
    print(f"  - Total de rostos cadastrados: {total_rostos}")
    
    if pessoas_com_multiplos == 0:
        print("\n⚠️  AVISO: Nenhuma pessoa tem 2 ou mais rostos cadastrados.")
        print("    A análise será limitada. Recomenda-se cadastrar mais rostos por pessoa.")
        return None
    
    print(f"\n{'─' * 80}")
    print("Iniciando validação cruzada (leave-one-out)...")
    print(f"{'─' * 80}\n")
    
    y_true = []
    y_pred = []
    distancias = []
    resultados_detalhados = []
    
    # Para cada pessoa com múltiplos rostos
    for nome_verdadeiro, rostos_pessoa in rostos_por_pessoa_filtrado.items():
        print(f"Testando: {nome_verdadeiro} ({len(rostos_pessoa)} rostos)")
        
        # Leave-one-out: cada rosto será testado
        for idx_teste, rosto_teste in enumerate(rostos_pessoa):
            # Base de referência: todos os rostos EXCETO o que está sendo testado
            referencias = []
            for idx_ref, rosto_ref in enumerate(rostos_pessoa):
                if idx_ref != idx_teste:
                    referencias.append(rosto_ref)
            
            # Adicionar rostos de todas as outras pessoas
            for outro_nome, outros_rostos in rostos_por_pessoa_filtrado.items():
                if outro_nome != nome_verdadeiro:
                    referencias.extend(outros_rostos)
            
            # Tentar reconhecer
            nome_reconhecido, distancia, ambiguo = reconhecer_nome_por_embedding(
                rosto_teste['embedding'], 
                referencias
            )
            
            # Registrar resultado
            y_true.append(nome_verdadeiro)
            y_pred.append(nome_reconhecido if nome_reconhecido else 'DESCONHECIDO')
            distancias.append(distancia if distancia is not None else 1.0)
            
            resultado = {
                'nome_verdadeiro': nome_verdadeiro,
                'nome_reconhecido': nome_reconhecido if nome_reconhecido else 'DESCONHECIDO',
                'distancia': distancia,
                'ambiguo': ambiguo,
                'correto': (nome_reconhecido == nome_verdadeiro),
                'id_rosto': rosto_teste['id_rosto']
            }
            resultados_detalhados.append(resultado)
            
            status = "✓" if resultado['correto'] else "✗"
            if ambiguo:
                status = "⚠"
            print(f"  {status} Rosto {idx_teste + 1}/{len(rostos_pessoa)}: "
                  f"Reconhecido como '{nome_reconhecido}' (dist: {distancia:.3f})")
    
    print(f"\n{'─' * 80}")
    print("Validação concluída!")
    print(f"{'─' * 80}\n")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'distancias': distancias,
        'resultados_detalhados': resultados_detalhados,
        'rostos_por_pessoa': rostos_por_pessoa_filtrado,
        'total_pessoas': total_pessoas,
        'total_rostos': total_rostos
    }


def gerar_relatorio_distancias(dados):
    """Gera relatório detalhado sobre distâncias e reconhecimento."""
    print("\n" + "=" * 80)
    print("RELATÓRIO DE DISTÂNCIAS E RECONHECIMENTO")
    print("=" * 80)
    
    resultados = dados['resultados_detalhados']
    distancias = dados['distancias']
    
    # Estatísticas gerais
    total_testes = len(resultados)
    reconhecidos = sum(1 for r in resultados if r['nome_reconhecido'] != 'DESCONHECIDO')
    corretos = sum(1 for r in resultados if r['correto'])
    incorretos = reconhecidos - corretos
    nao_reconhecidos = total_testes - reconhecidos
    ambiguos = sum(1 for r in resultados if r['ambiguo'])
    
    print(f"\n Estatísticas Gerais:")
    print(f"  • Total de testes realizados: {total_testes}")
    print(f"  • Reconhecidos automaticamente: {reconhecidos} ({reconhecidos/total_testes*100:.1f}%)")
    print(f"    ├─ Corretos: {corretos} ({corretos/total_testes*100:.1f}%)")
    print(f"    └─ Incorretos: {incorretos} ({incorretos/total_testes*100:.1f}%)")
    print(f"  • Não reconhecidos: {nao_reconhecidos} ({nao_reconhecidos/total_testes*100:.1f}%)")
    print(f"  • Rejeitados por ambiguidade: {ambiguos} ({ambiguos/total_testes*100:.1f}%)")
    
    if reconhecidos > 0:
        taxa_acerto_reconhecidos = corretos / reconhecidos * 100
        print(f"\n Taxa de acerto dos reconhecidos: {taxa_acerto_reconhecidos:.1f}%")
    
    # Estatísticas de distância
    distancias_validas = [d for d in distancias if d is not None]
    dist_corretos = [r['distancia'] for r in resultados if r['correto'] and r['distancia'] is not None]
    dist_incorretos = [r['distancia'] for r in resultados if not r['correto'] and r['nome_reconhecido'] != 'DESCONHECIDO' and r['distancia'] is not None]
    dist_nao_reconhecidos = [r['distancia'] for r in resultados if r['nome_reconhecido'] == 'DESCONHECIDO' and r['distancia'] is not None]
    
    print(f"\n Estatísticas de Distância (Euclidiana):")
    print(f"  • Distância média geral: {np.mean(distancias_validas):.3f} ± {np.std(distancias_validas):.3f}")
    print(f"  • Distância mínima: {np.min(distancias_validas):.3f}")
    print(f"  • Distância máxima: {np.max(distancias_validas):.3f}")
    print(f"  • Mediana: {np.median(distancias_validas):.3f}")
    
    if dist_corretos:
        print(f"\n  ✓ Reconhecimentos CORRETOS:")
        print(f"    • Média: {np.mean(dist_corretos):.3f} ± {np.std(dist_corretos):.3f}")
        print(f"    • Min: {np.min(dist_corretos):.3f} | Max: {np.max(dist_corretos):.3f}")
    
    if dist_incorretos:
        print(f"\n  ✗ Reconhecimentos INCORRETOS:")
        print(f"    • Média: {np.mean(dist_incorretos):.3f} ± {np.std(dist_incorretos):.3f}")
        print(f"    • Min: {np.min(dist_incorretos):.3f} | Max: {np.max(dist_incorretos):.3f}")
    
    if dist_nao_reconhecidos:
        print(f"\n  ○ NÃO reconhecidos:")
        print(f"    • Média: {np.mean(dist_nao_reconhecidos):.3f} ± {np.std(dist_nao_reconhecidos):.3f}")
        print(f"    • Min: {np.min(dist_nao_reconhecidos):.3f} | Max: {np.max(dist_nao_reconhecidos):.3f}")
    
    print(f"\n  Parâmetros do Modelo:")
    print(f"  • Tolerância de distância: {EMBEDDING_TOLERANCIA}")
    print(f"  • Margem de ambiguidade: {EMBEDDING_MARGEM_AMBIGUIDADE}")
    
    # Casos problemáticos
    if incorretos > 0:
        print(f"\n Casos de Reconhecimento INCORRETO:")
        for r in resultados:
            if not r['correto'] and r['nome_reconhecido'] != 'DESCONHECIDO':
                print(f"  • {r['id_rosto']}: '{r['nome_verdadeiro']}' → "
                      f"reconhecido como '{r['nome_reconhecido']}' (dist: {r['distancia']:.3f})")
    
    print("=" * 80)


def plotar_graficos(dados):
    """Gera todos os gráficos de análise."""
    y_true = dados['y_true']
    y_pred = dados['y_pred']
    distancias = dados['distancias']
    resultados = dados['resultados_detalhados']
    
    # Obter lista de classes únicas
    classes_unicas = sorted(list(set(y_true)))
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=classes_unicas + ['DESCONHECIDO'])
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Matriz de Confusão
    ax1 = fig.add_subplot(gs[0, 0])
    labels_cm = classes_unicas + ['DESCONHECIDO']
    
    # Limitar labels se houver muitas classes
    if len(labels_cm) > 20:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Contagem'})
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_xlabel('Previsto (labels ocultos - muitas classes)')
        ax1.set_ylabel('Verdadeiro (labels ocultos - muitas classes)')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                    xticklabels=labels_cm, yticklabels=labels_cm,
                    cbar_kws={'label': 'Contagem'})
        ax1.set_xlabel('Previsto')
        ax1.set_ylabel('Verdadeiro')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
    ax1.set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
    
    # 2. Distribuição de Classes (Verdadeiro)
    ax3 = fig.add_subplot(gs[0, 1])
    counter_true = Counter(y_true)
    nomes_classes = list(counter_true.keys())
    contagens = list(counter_true.values())
    
    if len(nomes_classes) <= 15:
        ax3.barh(nomes_classes, contagens, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Quantidade de Testes', fontsize=10)
        ax3.set_title('Distribuição de Classes\n(Rótulos Verdadeiros)', fontsize=11, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        plt.setp(ax3.get_yticklabels(), fontsize=8)
    else:
        # Muitas classes: mostrar histogram
        ax3.hist(contagens, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Quantidade de Testes por Classe', fontsize=10)
        ax3.set_ylabel('Frequência', fontsize=10)
        ax3.set_title(f'Distribuição de Classes\n({len(nomes_classes)} classes)', fontsize=11, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # 3. Distribuição de Distâncias
    ax5 = fig.add_subplot(gs[1, 0])
    
    dist_corretos = [r['distancia'] for r in resultados if r['correto'] and r['distancia'] is not None]
    dist_incorretos = [r['distancia'] for r in resultados 
                       if not r['correto'] and r['nome_reconhecido'] != 'DESCONHECIDO' and r['distancia'] is not None]
    dist_nao_reconhecidos = [r['distancia'] for r in resultados 
                             if r['nome_reconhecido'] == 'DESCONHECIDO' and r['distancia'] is not None]
    
    bins = np.linspace(0, 1.0, 30)
    
    if dist_corretos:
        ax5.hist(dist_corretos, bins=bins, alpha=0.6, label='Corretos', color='green', edgecolor='black')
    if dist_incorretos:
        ax5.hist(dist_incorretos, bins=bins, alpha=0.6, label='Incorretos', color='red', edgecolor='black')
    if dist_nao_reconhecidos:
        ax5.hist(dist_nao_reconhecidos, bins=bins, alpha=0.6, label='Não Reconhecidos', 
                color='gray', edgecolor='black')
    
    ax5.axvline(EMBEDDING_TOLERANCIA, color='blue', linestyle='--', linewidth=2, 
               label=f'Tolerância ({EMBEDDING_TOLERANCIA})')
    ax5.set_xlabel('Distância Euclidiana', fontsize=10)
    ax5.set_ylabel('Frequência', fontsize=10)
    ax5.set_title('Distribuição de Distâncias', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(axis='y', alpha=0.3)
    
    # 4. Análise de Reconhecimento
    ax6 = fig.add_subplot(gs[1, 1])
    
    total_testes = len(resultados)
    reconhecidos = sum(1 for r in resultados if r['nome_reconhecido'] != 'DESCONHECIDO')
    corretos = sum(1 for r in resultados if r['correto'])
    incorretos = reconhecidos - corretos
    nao_reconhecidos = total_testes - reconhecidos
    
    categorias = ['Reconhecidos\nCorretos', 'Reconhecidos\nIncorretos', 'Não\nReconhecidos']
    valores_cat = [corretos, incorretos, nao_reconhecidos]
    cores_cat = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    wedges, texts, autotexts = ax6.pie(valores_cat, labels=categorias, autopct='%1.1f%%',
                                        colors=cores_cat, startangle=90, 
                                        textprops={'fontsize': 9, 'weight': 'bold'})
    
    # Adicionar contagens absolutas
    for i, (autotext, valor) in enumerate(zip(autotexts, valores_cat)):
        autotext.set_text(f'{valor}\n({valor/total_testes*100:.1f}%)')
        autotext.set_fontsize(8)
    
    ax6.set_title('Análise de Reconhecimento', fontsize=11, fontweight='bold')
    
    # Título geral
    fig.suptitle('Análise Técnica Completa do Modelo de Reconhecimento Facial', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adicionar informações no rodapé
    info_text = (f"Total de Pessoas: {dados['total_pessoas']} | "
                f"Total de Testes: {total_testes} | "
                f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=9, style='italic')
    
    plt.show()


def main():
    """Função principal."""
    print("\n" + "=" * 80)
    print("INICIANDO ANÁLISE TÉCNICA DO MODELO")
    print("=" * 80 + "\n")
    
    # Executar validação cruzada
    dados = executar_validacao_cruzada()
    
    if dados is None:
        print("\n❌ Análise não pôde ser realizada. Verifique os dados no banco.")
        return
    
    # Gerar relatório de distâncias
    gerar_relatorio_distancias(dados)
    
    # Plotar gráficos
    print("\n📊 Gerando gráficos... (feche a janela para continuar)")
    plotar_graficos(dados)
    
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
