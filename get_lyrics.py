import json
from dotenv import load_dotenv
import os
import base64
from requests import post,get


def get_letra_vagalume(artist, song, api_key):
    """
    Busca a letra de uma música no Vagalume.

    Args:
        artist: Nome do artista.
        song: Nome da música.
        api_key: Chave de API do Vagalume.

    Returns:
        A letra da música, caso encontrada. Caso contrário, retorna None.
    """

    url = f"https://api.vagalume.com.br/search.php?art={artist}&mus={song}&apikey={api_key}"
    response = get(url)

    if response.status_code == 200:
        data = response.json()
        if data.get('mus'):
            return data['mus'][0]['text']
        else:
            return "Letra não encontrada."
    else:
        return "Erro ao buscar a letra."

# Exemplo de uso
#artist = "U2"
#song = "One"
api_key = "YOUR_API_KEY"  # Substitua pela sua chave API

#letra = get_letra_vagalume(artist, song, api_key)
#print(letra)





# def print_letras(arquivo, database_file, api_key):
#     """
#     Imprime as letras das músicas em um novo arquivo.

#     Args:
#         arquivo: Nome do arquivo com as informações das músicas.
#         database_file: Nome do arquivo de saída.
#         api_key: Chave de API do Vagalume.
#     """

#     with open(database_file, "w", encoding="utf-8") as database:
#         database.write("Track_name;Artist_Name;genra;lyrics\n")

#         with open(arquivo, "r", encoding="utf-8") as file:
#             for line in file:
#                 # Ignora a primeira linha (cabeçalho)
#                 if line.strip().startswith("Track_name"):
#                     continue

#                 # Separa os campos da linha
#                 track_name, artist_name, genre = line.strip().split(';')

#                 # Busca a letra no Vagalume
#                 letra = get_letra_vagalume(artist_name, track_name, api_key)

#                 # Imprime a linha com a letra
#                 database.write(f"{track_name};{artist_name};{genre};{letra}\n")

# ... (função get_letra_vagalume)



def print_letras(arquivo, database_file, api_key):
    """
    Imprime as letras das músicas em um novo arquivo, com a letra em uma única célula.

    Args:
        arquivo: Nome do arquivo com as informações das músicas.
        database_file: Nome do arquivo de saída.
        api_key: Chave de API do Vagalume.
    """

    with open(database_file, "w", encoding="utf-8") as database:
        database.write("Track_name;Artist_Name;genra;lyrics\n")

        with open(arquivo, "r", encoding="utf-8") as file:
            for line in file:
                # Ignora a primeira linha (cabeçalho)
                if line.strip().startswith("Track_name"):
                    continue

                # Separa os campos da linha
                track_name, artist_name, genre = line.strip().split(';')

                # Busca a letra no Vagalume
                letra = get_letra_vagalume(artist_name, track_name, api_key)

                # Escapa aspas duplas dentro da letra
                letra_escaped = letra.replace('"', '""')

                # Envolve a letra entre aspas duplas
                letra_formatada = f'"{letra_escaped}"'

                # Imprime a linha com a letra formatada
                database.write(f"{track_name};{artist_name};{genre};{letra_formatada}\n")


# Exemplo de uso
arquivo = "teste.txt"
database_file = "database.txt"
api_key = "YOUR_API_KEY"
print_letras(arquivo, database_file, api_key)