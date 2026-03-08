from gerar_tabela_rostos import gerar_tabela_rostos


if __name__ == '__main__':
    tabela = gerar_tabela_rostos()
    if tabela.empty:
        print('Nenhum rosto cadastrado na tabela.')
    else:
        print(tabela.to_string(index=False))
