import json
from dotenv import load_dotenv
import os
import base64
from requests import post,get
from lyricsgenius import Genius
import re

def remover_caracteres_nao_suportados(song):
   song = re.sub(r'\u200b', '',song)
   song = re.sub(r'\u0435', '',song)
   return song
def remover_marcadores(musica):
  # Expressão regular para encontrar os marcadores
  padrao = r"\[Verse \d+\]|\[Chorus\]"

  # Remove todos os marcadores encontrados
  musica_limpa = re.sub(padrao, "", musica)

  return musica_limpa
def remover_primeira_linha(texto):
    linhas = texto.splitlines()
    return '\n'.join(linhas[1:])
def string_treatment(song):
    
    song=remover_primeira_linha(song)
    song=remover_marcadores(song)
    song=remover_caracteres_nao_suportados(song)
    song = str.join(" ", song.splitlines())
    return song
   


#letra = get_letra_vagalume(artist, song, api_key)
#print(letra)

load_dotenv()
client_id=os.getenv("CLIENT_ID")
client_secret=os.getenv("CLIENT_SECRET")
token=os.getenv("G_CLIENT_ACCESS_TOKEN")
genius = Genius(token)



def print_letras(arquivo, database_file):
    

    with open(database_file, "a") as database:
        #database.write("Track_name;Artist_Name;genra;lyrics\n")

        with open(arquivo, "r") as file:
            for line in file:
                # Ignora a primeira linha (cabeçalho)
                if line.strip().startswith("Track_name"):
                    continue
                
                # Separa os campos da linha
                track_name, artist_name, genre = line.strip().split(';')

                # Busca a letra no Vagalume
                #letra = get_letra_vagalume(artist_name, track_name, api_key)
                
                try:
                    song=genius.search_song(track_name, artist_name)
                    # Escapa aspas duplas dentro da letra
                    #letra_escaped = letra.replace('"', '""')
                    if(song!=None):
                        song_lyrics= string_treatment(song.lyrics)
                except:
                    print(f"Nao consegui adicionar {track_name}")
                    song_lyrics=""
                finally:

                    # Envolve a letra entre aspas duplas
                    #letra_formatada = f'"{letra_escaped}"'

                    # Imprime a linha com a letra formatada
                    try:
                        database.write(f"{track_name};{artist_name};{genre};{song_lyrics};\n")
                    except:
                        print("Erro ao imprimir")
                    


# Exemplo de uso
arquivo = "tracks.txt"
database_file = "database.txt"

def main():
    
    print_letras(arquivo, database_file)

if __name__ == '__main__':
    main()
