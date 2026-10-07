"""Download the solvent-specific 3D conformers that CycPeptMPDB provides for each peptide.

For every peptide ID, CycPeptMPDB hosts the lowest-energy conformer in three environments: vacuum,
chloroform (a membrane-like, low-dielectric solvent) and water. They are saved as

    data/sdf_vacuum/id_0001_vacuum.mol
    data/sdf_chloroform/id_0001_chloroform.mol
    data/sdf_water/id_0001_water.mol

and converted into E3FP fingerprints in notebooks/01_classical_ml.ipynb. Existing files are skipped,
so an interrupted download resumes where it stopped.

Usage:
    python scripts/download_cycpeptmpdb_3d.py [--first-id 1] [--last-id 9000]
"""
import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_URL = "http://cycpeptmpdb.com/static//peptides"
SLEEP_TIME = 1.0  # seconds between peptides, to keep the load on the server low

# (URL template, output folder, file-name suffix) for each environment
SOURCES = [
    (BASE_URL + "/mol/CycPeptMPDB_ID_{pid}.mol", "sdf_vacuum", "vacuum"),
    (BASE_URL + "/mol_CHCl3/CycPeptMPDB_ID_{pid}_CHCl3.mol", "sdf_chloroform", "chloroform"),
    (BASE_URL + "/mol_H2O/CycPeptMPDB_ID_{pid}_H2O.mol", "sdf_water", "water"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-id", type=int, default=1, help="first peptide ID to download (default: 1)")
    parser.add_argument("--last-id", type=int, default=9000,
                        help="last peptide ID to download (default: 9000; CycPeptMPDB has 8,466 peptides)")
    args = parser.parse_args()

    for _, folder, _ in SOURCES:
        (DATA_DIR / folder).mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    # Identify as a browser; some servers block the default python-requests User-Agent
    session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})

    counts = {name: 0 for _, _, name in SOURCES}
    for pid in tqdm(range(args.first_id, args.last_id + 1)):
        for url_template, folder, name in SOURCES:
            path = DATA_DIR / folder / f"id_{pid:04d}_{name}.mol"
            if path.exists():
                continue
            try:
                response = session.get(url_template.format(pid=pid), timeout=30)
            except requests.RequestException as exc:
                tqdm.write(f"  ✗ ERROR   {name.upper():10} id_{pid:04d}: {exc}")
                continue
            if response.status_code == 200:
                path.write_bytes(response.content)
                counts[name] += 1
                tqdm.write(f"  ✓ SAVED   {name.upper():10} id_{pid:04d} ({len(response.content)} bytes)")
            else:
                tqdm.write(f"  ✗ SKIPPED {name.upper():10} id_{pid:04d} (HTTP {response.status_code})")
        time.sleep(SLEEP_TIME)

    print("\nDownload summary")
    for name, n_files in counts.items():
        print(f"  {name.capitalize():10}: {n_files} files")
    print(f"Saved under {DATA_DIR}")


if __name__ == "__main__":
    main()
