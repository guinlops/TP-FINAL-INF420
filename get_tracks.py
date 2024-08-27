import json
from dotenv import load_dotenv
import os
import base64
from requests import post,get

load_dotenv()
client_id=os.getenv("CLIENT_ID")
client_secret=os.getenv("CLIENT_SECRET")


def get_token():
    auth_string=client_id+":"+ client_secret
    auth_bytes=auth_string.encode("utf-8")
    auth_base64=str(base64.b64encode(auth_bytes),"utf-8")
    url="https://accounts.spotify.com/api/token"
    headers={
        "Authorization": "Basic "+ auth_base64,
        "Content-Type":"application/x-www-form-urlencoded"
    }
    data={"grant_type":"client_credentials"}
    result=post(url,headers=headers,data=data)
    json_result=json.loads(result.content)
    token=json_result["access_token"]
    return token
token=get_token()

def get_auth_header(token):
    return{"Authorization":"Bearer "+ token}

def search_for_artist(token, artist_name):
    url="https://api.spotify.com/v1/search"
    headers=get_auth_header(token)
    query=f"?q={artist_name}&type=artist&limit=1"

    query_url=url+query
    result=get(query_url,headers=headers)
    json_result=json.loads(result.content)["artists"]["items"]
    #print(json_result)
    if(len(json_result)==0):
        print("No artist with this name exists...")
        return None
    
    return json_result[0]

def get_songs_by_artist(token,artist_id):
    url=f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers=get_auth_header(token)
    result=get(url,headers=headers)
    json_result=json.loads(result.content)["tracks"]
    return json_result


def get_playlist_tracks(token, playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = get_auth_header(token)

    result = get(url, headers=headers)
    json_result = json.loads(result.content) 

    tracks = json_result["items"]
    return tracks


def get_playlist_obj(token, playlist_name):
  """
  Busca por uma playlist pública no Spotify.

  Args:
    token: Token de acesso da sua aplicação Spotify.
    query: Termo de busca para a playlist.

  Returns:
    Um dicionário com informações sobre a playlist encontrada, ou None se não encontrar.
  """

  url = "https://api.spotify.com/v1/search"
  headers = get_auth_header(token)
  params = {"q": playlist_name, "type": "playlist"}

  response = get(url, headers=headers, params=params)
  response.raise_for_status()

  json_response = response.json()
  playlists = json_response.get('playlists', {}).get('items', [])

  if playlists:
    return playlists[0]  # Retorna a primeira playlist encontrada
  else:
    return None



def print_tracks(tracks,playlist_genra):
    
    try:
        with open("tracks.txt", "a") as arquivo:
            #print("Track_name;Artist_Name;genra",file=arquivo)
            for track in tracks:
                song_name = track["track"]["name"]
                artist_name = track["track"]["artists"][0]["name"]
                print(f"{song_name};{artist_name};{playlist_genra}", file=arquivo)
    except FileNotFoundError:
        print("Erro: O arquivo não foi encontrado.")
    except PermissionError:
        print("Erro: Permissão negada para acessar o arquivo.")
    except Exception as e:
        print(f"Erro inesperado ao escrever no arquivo: {str(e)}")

def get_playlist(playlist_name,playlist_genra):
     playlist_obj=get_playlist_obj(token,playlist_name)
     playlist_id=playlist_obj['id']
     tracks=get_playlist_tracks(token,playlist_id)
     print_tracks(tracks,playlist_genra)


def main():
    playlists = [
        "This is Metallica",
        "This is Led Zeppelin",
        "This is Pink Floyd",
        "This is Jimi Hendrix",
        "This is Van Halen",
        "This is Queen",
        "This is Eagles",
        "This is U2",
        "This is The Rolling Stones",
        "This is Pearl Jam",
        "This is Aerosmith",
        "This is Red Hot Chili Peppers",
        "This is Dire Straits",
        "This is Nirvana",
        "This is The Beatles",
        "This is Alice In Chains",
        "This is Audioslave",
        "This is Black Sabbath",
        "This is Iron Maiden",
        "This is Soundgarden"
    ]
    
    for playlist in playlists:
        get_playlist(playlist, "rock")  # Passando o gênero como "rock" para todas as playlists
if __name__ == '__main__':
    main()
