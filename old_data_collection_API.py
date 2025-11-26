#fileName : data_collection.py
#Description : is a script for collecting data about songs from the spotify API 
#Author : Paarth Sharma 
#Data created : 23rd november 2025

#imports and initialization 
import time
import spotipy as spo
from spotipy.oauth2 import SpotifyClientCredentials 
import polars as pl 
import os
import csv

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# Add this check
if not CLIENT_ID or not CLIENT_SECRET:
    print("ERROR: CLIENT_ID or CLIENT_SECRET not found in environment variables!")
    print(f"CLIENT_ID exists: {CLIENT_ID is not None}")
    print(f"CLIENT_SECRET exists: {CLIENT_SECRET is not None}")
    exit(1)
else:
    print("Spotify credentials loaded successfully")

#authorise the spotify api 
sp = spo.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
)

#declare artists
artists =[]

# read names into a content file 
with open("./artists/artists.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 0:
            # Clean the artist name - remove everything in parentheses
            artist_name = row[0].strip()
            # Remove anything in parentheses
            artist_name = artist_name.split('(')[0].strip()
            artists.append(artist_name)

#helper functions

#safe API calls 
def safe_API_calls(func, *args, **kwargs) :
    retry_delay = 2
    max_delay = 30

    #run the program forever so it does not crash on a error message
    while True:

        #try the API call if it doesn't work catch the exception
        try : 
            result = func(*args, **kwargs)
            print("  ✓ API call successful")
            return result
        except Exception as e:
            #convert error message to a string
            error_str = str(e)

            #rate limited : 429 response 
            if "429" in error_str or "rate" in error_str.lower() :
                print(f"rate limit hit sleeping for {retry_delay} seconds ...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)
            
            #network error 
            elif "connection" in error_str.lower() or "timeout" in error_str.lower() :
                print(f"network error occured retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay*2, max_delay)
            
            #other error
            else :
                print(error_str)
                return None

#get artist ID 
def get_artist_id(name):
    
    #search for the artist
    result = safe_API_calls(
        sp.search,
        q=f"artist:{name}",
        type = "artist", 
        limit = 1
    )

    #check if the artist exists
    if not result :
        return None
    
    #make a default object 
    items = result["artists"]["items"]

    if len(items) == 0 :
        return None
    
     #make sure we have the artists key
    if "artists" not in result:
        print(f"  No 'artists' key in result for: {name}")
        return None
    
    items = result["artists"]["items"]

    if len(items) == 0:
        print(f"  No items found for: {name}")
        return None
    
    artist_id = items[0]["id"]
    print(f"  Found artist ID: {artist_id} for {name}")
    
    return items[0]["id"]

#get tracks for an artist 
def get_tracks_for_artists(artist_id) :
    
    print(f"  Artist ID: {artist_id}")

    #search tracks for the artist 
    result = safe_API_calls(
        sp.artist_top_tracks,
        artist_id,
        country='US'
    )

    if not result :
        return None

    items = result.get("tracks", [])

    track_list = []

    for item in items : 
        track_list.append( {
            "track_id" : item["id"],
            "track_name" : item["name"],
            "artist_name" : item["artists"][0]["name"],
            "popularity" : item["popularity"],
            "duration_sec" : item["duration_ms"]/1000  
        })

    #return tracks in the form of list 
    return track_list


#add audio features 
def get_audio_features(tracks) :

    final_track_with_features = []

    #get track ids for batching tracks to make the API calls more efficient
    
    track_ids = [item["track_id"] for item in tracks]

    #loop over the track list 
    for i in range(0, len(track_ids), 100) : 

        #get the batch ids for one batch
        batch_ids = track_ids[i : i+100]

        #make a batch of tracks 
        batch_tracks = tracks[i : i+100]

        #call the API to get the features 
        features = safe_API_calls(
            sp.audio_features,
            batch_ids
        )

        if not features : 
            continue 
        
        # create a copy of the batch of features 
        batch_features = features

        #call the API to get the features 
        try:
            features = sp.audio_features(batch_ids)  # Try without safe_API_calls first
            
            if not features:
                print(f"  Warning: No features returned for batch {i}")
                continue
            
            #loop through the tracks and add the features in the tracks 
            for index in range(len(batch_ids)):
                track = batch_tracks[index].copy()  # Make a copy to avoid modifying original
                feature = features[index]

                if feature is not None: 
                    #merge the tracks and the features 
                    for key, value in feature.items():
                        track[key] = value
                    
                    #save the final track to the list 
                    final_track_with_features.append(track)
            
        except Exception as e:
            print(f"  Error getting audio features for batch {i}: {str(e)}")
            # Continue with tracks without features
            for track in batch_tracks:
                final_track_with_features.append(track)
        
        #add a sleep timer to not hit the API limit 
        time.sleep(0.1)

    #return list 
    return final_track_with_features


#main loop 

#loop through the artists and find the 
for artist in artists : 
    
    print(f"collecting songs for :{artist}")

    #get artist ID for the artist 
    artist_id = get_artist_id(artist)

    #show a message is artist is not found 
    if artist_id is None :
        print(f"skipping {artist} because no artist ID found ")
        continue
    
    #get tracks for the artist
    tracks = get_tracks_for_artists(artist_id)

    #if nothing is returned show a message
    if tracks is None : 
        print(f"no tracks found for {artist}")
        continue
    
    #get audio features for the tracks 
    tracks = get_audio_features(tracks)

    #fill in the artist tracks in the big list of songs 
    all_tracks.extend(tracks)

#main loop 

all_tracks = []
successful_artists = 0
failed_artists = []  # Changed to list to track names

print(f"\n{'='*60}")
print(f"Starting data collection for {len(artists)} artists")
print(f"{'='*60}\n")

#loop through the artists and find the 
for idx, artist in enumerate(artists, 1): 
    
    print(f"\n[{idx}/{len(artists)}] Collecting songs for: {artist}")
    print("-" * 50)

    #get artist ID for the artist 
    artist_id = get_artist_id(artist)

    #show a message if artist is not found 
    if artist_id is None:
        print(f"✗ Skipping {artist} - no artist ID found")
        failed_artists.append(artist)  # Track the name
        continue
    
    #get tracks for the artist
    tracks = get_tracks_for_artists(artist_id)

    #if nothing is returned show a message
    if tracks is None: 
        print(f"✗ No tracks found for {artist}")
        failed_artists.append(artist)  # Track the name
        continue
    
    #get audio features for the tracks 
    tracks = get_audio_features(tracks)

    #fill in the artist tracks in the big list of songs 
    all_tracks.extend(tracks)
    successful_artists += 1
    print(f" Successfully collected {len(tracks)} tracks for {artist}")

print(f"\n{'='*60}")
print(f"Data collection complete!")
print(f"Successful: {successful_artists} artists")
print(f"Failed: {len(failed_artists)} artists")
if failed_artists:
    print(f"\nFailed artists:")
    for artist in failed_artists:
        print(f"  - {artist}")
print(f"\nTotal tracks collected: {len(all_tracks)}")
print(f"{'='*60}\n")

print("DONE")
