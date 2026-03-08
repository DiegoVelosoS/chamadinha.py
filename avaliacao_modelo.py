import sqlite3
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DB_NAME = 'rostos.db'

# Exemplo de função para avaliar um modelo de classificação de rostos
## y_true: rótulos verdadeiros, y_pred: rótulos previstos, y_score: probabilidades (para ROC AUC)
def avaliar_modelo(y_true, y_pred, y_score=None, class_names=None):
    # Calcula métricas
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = None
    if y_score is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except Exception as e:
            roc_auc = f"Não pôde ser calculado: {e}"

    # Exibe métricas
    print("Matriz de Confusão:")
    print(cm)
    print(f"Accuracy: {acc:.2f}")
    print(f"Precisão: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1-Score: {f1:.2f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.2f}' }")

    # Dashboard visual
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Matriz de confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None)
    axes[0,0].set_title('Matriz de Confusão')
    axes[0,0].set_xlabel('Previsto')
    axes[0,0].set_ylabel('Verdadeiro')

    # Métricas
    metricas = ['Accuracy', 'Precisão', 'Recall', 'F1-Score']
    valores = [acc, prec, rec, f1]
    axes[0,1].bar(metricas, valores, color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'])
    axes[0,1].set_ylim(0, 1)
    axes[0,1].set_title('Métricas de Desempenho')

    # ROC AUC (se disponível)
    if y_score is not None and not isinstance(roc_auc, str):
        try:
            from sklearn.metrics import roc_curve, auc
            n_classes = y_score.shape[1]
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(np.array(y_true) == i, y_score[:, i])
                axes[1,0].plot(fpr, tpr, label=f'Classe {class_names[i] if class_names else i} (AUC = {auc(fpr, tpr):.2f})')
            axes[1,0].plot([0, 1], [0, 1], 'k--')
            axes[1,0].set_title('Curvas ROC por Classe')
            axes[1,0].set_xlabel('FPR')
            axes[1,0].set_ylabel('TPR')
            axes[1,0].legend()
        except Exception as e:
            axes[1,0].text(0.5, 0.5, f'ROC não pôde ser exibido: {e}', ha='center')
    else:
        axes[1,0].axis('off')

    # Distribuição dos rótulos
    df = pd.DataFrame({'Verdadeiro': y_true, 'Previsto': y_pred})
    sns.countplot(x='Verdadeiro', data=df, ax=axes[1,1], palette='Blues')
    axes[1,1].set_title('Distribuição dos Rótulos Verdadeiros')

    plt.tight_layout()
    plt.show()

# Exemplo de uso: carregar y_true, y_pred, y_score de arquivos pickle ou gerar exemplos fictícios
def exemplo_avaliacao():
    # Exemplo fictício
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 0, 2, 2, 1]
    y_score = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.2, 0.7],
        [0.1, 0.2, 0.7],
        [0.2, 0.7, 0.1]
    ])
    class_names = ['Aluno A', 'Aluno B', 'Aluno C']
    avaliar_modelo(y_true, y_pred, y_score, class_names)

if __name__ == "__main__":
    exemplo_avaliacao()
