from datetime import datetime
from typing import Callable, Union, cast
import json
import logging
import os
from posixpath import basename
import essentia
from essentia import Pool
from essentia.standard import MonoLoader, TensorflowPredictMAEST, TensorflowPredict
import numpy as np
from dataclasses import asdict, dataclass
from xml.dom.minidom import parse
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

@dataclass
class RekordboxTrack:
    id: str
    title: str
    artist: str
    location: str
    album: str | None = None
    genre: str | None = None
    subgenre: str | None = None
    


def parse_rekordbox_location(location: str) -> str:
    location = location.replace('file://localhost', '')
    location = unquote(location)
    return location


def parse_rekordbox_xml(file_path: str) -> list[RekordboxTrack]:
    dom = parse(file_path)
    tracks = dom.getElementsByTagName('TRACK')
    rb_tracks = []

    for track in tracks:
        id = track.getAttribute('TrackID')
        title = track.getAttribute('Name')
        artist = track.getAttribute('Artist')
        location = track.getAttribute('Location')
        genre = track.getAttribute('Genre')
        if not location:
            continue
        location = parse_rekordbox_location(location)
        genre = genre if genre else None
        rb_tracks.append(RekordboxTrack(id=id, title=title, artist=artist, genre=genre, location=location))

    return rb_tracks


PatchableAttr = Union[str, Callable[[RekordboxTrack], str | None]]

def patch_rekordbox_xml(file_path: str, rb_tracks: list[RekordboxTrack], patchable_attrs: dict[str, PatchableAttr]):
    dom = parse(file_path)
    tracks = dom.getElementsByTagName('TRACK')

    # Make timestamped backup
    base, ext = os.path.splitext(file_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = f"{base}_{timestamp}{ext}"
    with open(backup_path, 'w', encoding='utf-8') as f:
        dom.writexml(f, encoding='utf-8')

    for track in rb_tracks:
        logging.info(f"Patching {track.title}")
        # find track node with given id
        try:
            node = next(t for t in tracks if t.getAttribute("Name") == track.title) 
        except StopIteration:
            logging.error(f"Node not found for {track.title}")
            continue

        for xml_attr, track_attr in patchable_attrs.items():
            logging.info(f"Processing attribute {track_attr}")
            if node.hasAttribute(xml_attr) and node.getAttribute(xml_attr) != "":
                logging.info("attribute {attr} is already set, skipping")
                continue

            track_dict = asdict(track)
            logging.info(f"Track data: {track_dict}")
            # if track_attr not in track_dict or not track_dict[track_attr]:
            #     logging.error(f"Track does not have attribute {track_attr}, skipping")
            #     continue

            if callable(track_attr):
                value = track_attr(track)
            else:
                key = cast(str, track_attr)
                value = track_dict[key]
            if not value:
                logging.error(f"Value for attribute {track_attr} is empty, skipping")
                continue

            logging.info(f"Setting {xml_attr} to {value}")
            node.setAttribute(xml_attr, value)

    # Save the modified XML
    with open(file_path, 'w', encoding='utf-8') as f:
        dom.writexml(f, encoding='utf-8')

# Suppress Essentia warnings about networks
essentia.log.warningActive = False
essentia.log.infoActive = True

# Load classes once
with open("genre_discogs519.json", "r", encoding="utf-8") as f:
    classes = json.load(f)["classes"]


# Persistent models for main thread / process
embedding_model = TensorflowPredictMAEST(
    graphFilename="discogs-maest-30s-pw-519l-2.pb",
    output="PartitionedCall/Identity_12",
)
model = TensorflowPredict(
    graphFilename="genre_discogs519-discogs-maest-30s-pw-519l-1.pb",
    inputs=["embeddings"],
    outputs=["PartitionedCall/Identity_1"],
)


def load_audio(path: str):
    """Load audio from file."""
    logging.info(f"Loading {basename(path)}")
    return MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()


def classify_batch(audio_list: MonoLoader):
    results = []
    for audio in audio_list:
        embeddings = embedding_model(audio)
        pool = Pool()
        pool.set("embeddings", embeddings)
        predictions = model(pool)["PartitionedCall/Identity_1"]
        preds = predictions[0].squeeze()
        top_idx = np.argmax(preds)
        top_class = classes[top_idx]
        results.append(tuple(top_class.split("---")))
    return results


def process_tracks(rb_tracks: list[RekordboxTrack], batch_size: int = 8):
    """Process all tracks in batches with parallel audio loading."""
    filename = "classified_tracks.jsonl"
    # Clear output file
    with open(filename, "w") as f:
        pass

    total_batches = (len(rb_tracks) + batch_size - 1) // batch_size

    for i in range(0, len(rb_tracks), batch_size):
        batch_tracks = rb_tracks[i:i + batch_size]
        batch_number = i // batch_size + 1

        # Parallel audio loading
        logging.info(f"Loading batch {batch_number} of {total_batches}")
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            audios = list(executor.map(lambda t: load_audio(t.location), batch_tracks))

        # Compute predictions in batch
        logging.info(f"Classifying batch {batch_number} of {total_batches}")
        predictions = classify_batch(audios)

        # Update tracks and write results
        with open(filename, "a", encoding="utf-8") as f:
            for track, (genre, subgenre) in zip(batch_tracks, predictions):
                track.genre = genre
                track.subgenre = subgenre
                obj = asdict(track)
                f.write(json.dumps(obj) + "\n")
                logging.info(f"Classified {track.title} as {genre} / {subgenre}")


if __name__ == "__main__":
    rb_tracks = parse_rekordbox_xml("rekordbox.xml")
    print(f"Parsed {len(rb_tracks)} tracks from Rekordbox XML")
    
    rb_tracks = [t for t in rb_tracks if not t.genre]
    print(f"{len(rb_tracks)} tracks to classify (without genre)")

    process_tracks(rb_tracks, batch_size=4)

    patch_rekordbox_xml("rekordbox.xml", rb_tracks, patchable_attrs={
        "Genre": lambda t: f"{t.genre} / {t.subgenre}" if t.genre and t.subgenre else t.genre,
    })

    # with open("classified_tracks.jsonl", "r", encoding="utf-8") as f:
    #     classified_tracks = [RekordboxTrack(**json.loads(line)) for line in f]
    #     logging.info(f"Loaded {len(classified_tracks)} classified tracks from JSONL")
    #
    # patch_rekordbox_xml("rekordbox.xml", classified_tracks, patchable_attrs={
    #     "Genre": lambda t: f"{t.genre} / {t.subgenre}" if t.genre and t.subgenre else t.genre,
    # })


