import json
import logging
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
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

@dataclass
class RekordboxTrack:
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
        title = track.getAttribute('Name')
        artist = track.getAttribute('Artist')
        location = track.getAttribute('Location')
        genre = track.getAttribute('Genre')
        if not location:
            continue
        location = parse_rekordbox_location(location)
        genre = genre if genre else None
        rb_tracks.append(RekordboxTrack(location=location, title=title, artist=artist, genre=genre))

    return rb_tracks


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
    process_tracks(rb_tracks, batch_size=2)


