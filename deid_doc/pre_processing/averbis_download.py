import json
import os
from datetime import datetime
from pathlib import Path

import requests

# python -m deid_doc.pre_processing.download

if __name__ == "__main__":
    datasets_folder = Path.cwd().parent / "datasets" / "melanom-export"
    new_folder = datasets_folder / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    # Set the API token in your env variables
    session.headers.update({"api-token": f"{os.environ['AVERBIS_TOKEN']}"})
    for b in range(1, 7):
        result = session.get(
            "https://ahd.diz.uk-essen.de/health-discovery/rest/v1/textanalysis/"
            f"projects/annotation-lab/documentSources/verlaufsdoku-batch{b}/processes/"
            f"deid-verlaufsdoku-batch{b}/export"
        )
        result.raise_for_status()
        batch_result = result.json()
        with (new_folder / f"batch_{b}.json").open("w") as fp:
            json.dump(batch_result, fp)
