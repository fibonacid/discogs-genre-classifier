import json
from posixpath import basename
from essentia import Pool
from essentia.standard import MonoLoader, TensorflowPredictMAEST, TensorflowPredict
import numpy as np
import glob


def classify_music(path: str):
    audio = MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()

    embedding_model = TensorflowPredictMAEST(
        graphFilename="discogs-maest-30s-pw-519l-2.pb",
        output="PartitionedCall/Identity_12",
    )
    embeddings = embedding_model(audio)

    pool = Pool()
    pool.set("embeddings", embeddings)

    model = TensorflowPredict(
        graphFilename="genre_discogs519-discogs-maest-30s-pw-519l-1.pb",
        inputs=["embeddings"],
        outputs=["PartitionedCall/Identity_1"],
    )
    predictions = model(pool)["PartitionedCall/Identity_1"]

    # Load the JSON file
    with open("genre_discogs519.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract the classes list
    classes = data["classes"]

    print(f"Loaded {len(classes)} classes")
    print(classes[:5])  # print first 5
    # squeeze to remove singleton dimensions
    preds = predictions[0].squeeze()  # shape -> (519,)

    top_idx = np.argmax(preds)
    top_class = classes[top_idx]
    genre, subgenre = top_class.split("---")

    print(f"Predicted genre: {genre}")
    print(f"Predicted subgenre: {subgenre}")

    return (genre, subgenre)


mp3_files = glob.glob("/Users/lorenzo/Music/**/*.mp3", recursive=True)
print(mp3_files)

for file in mp3_files:
    print(f"Classifing {basename(file)}")
    classify_music(file)
