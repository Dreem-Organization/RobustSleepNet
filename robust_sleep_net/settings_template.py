import os

VERSION = "1_03"
# Suggested directory where H5 and memmaps will be stored
BASE_DIRECTORY = "/data/"
BASE_DIRECTORY_H5 = BASE_DIRECTORY + "h5/"
if not os.path.exists("/memmap/"):
    BASE_DIRECTORY_MEMMAP = BASE_DIRECTORY + "memmap/"
else:
    BASE_DIRECTORY_MEMMAP = "/memmap/"

EXPERIMENTS_DIRECTORY = BASE_DIRECTORY + "experiments/"
DATASET_DIRECTORY = BASE_DIRECTORY + "datasets/"

DODH_BASE_DIRECTORY = DATASET_DIRECTORY + "dodh/"
DODH_SETTINGS = {
    "records": DODH_BASE_DIRECTORY + "h5/",
    "h5_directory": BASE_DIRECTORY_H5 + "dodh/",
    "memmap_directory": BASE_DIRECTORY_MEMMAP + "dodh/",
}

DODO_BASE_DIRECTORY = DATASET_DIRECTORY + "dodo/"
DODO_SETTINGS = {
    "records": DODO_BASE_DIRECTORY + "h5/",
    "h5_directory": BASE_DIRECTORY_H5 + "dodo/",
    "memmap_directory": BASE_DIRECTORY_MEMMAP + "dodo/",
}

SLEEP_EDF_BASE_DIRECTORY = DATASET_DIRECTORY + "sleep_edf/"
SLEEP_EDF_SETTINGS = {
    "base_directory": SLEEP_EDF_BASE_DIRECTORY,
    "edf_directory": [
        f"{SLEEP_EDF_BASE_DIRECTORY}/sleep-edf-database-expanded-1.0.0/sleep-cassette/",
        f"{SLEEP_EDF_BASE_DIRECTORY}/sleep-edf-database-expanded-1.0.0/sleep-telemetry",
    ],
    "h5_directory": BASE_DIRECTORY_H5 + "sleep_edf/",
    "memmap_directory": BASE_DIRECTORY_MEMMAP + "sleep_edf/",
}
SLEEP_EDF_IN_BED_SETTINGS = {
    "base_directory": SLEEP_EDF_BASE_DIRECTORY,
    "edf_directory": [
        f"{SLEEP_EDF_BASE_DIRECTORY}/sleep-edf-database-expanded-1.0.0/sleep-cassette/",
        f"{SLEEP_EDF_BASE_DIRECTORY}/sleep-edf-database-expanded-1.0.0/sleep-telemetry",
    ],
    "h5_directory": BASE_DIRECTORY_H5 + "sleep_edf_in_bed/",
    "memmap_directory": BASE_DIRECTORY_MEMMAP + "sleep_edf_in_bed/",
}

CAP_BASE_DIRECTORY = DATASET_DIRECTORY + "cap/"
CAP_SETTINGS = {
    "base_directory": CAP_BASE_DIRECTORY,
    "h5_directory": BASE_DIRECTORY_H5 + "cap/",
    "memmap_directory": BASE_DIRECTORY_MEMMAP + "cap/",
}

MASS_BASE_DIRECTORY = DATASET_DIRECTORY + "mass/"
MASS_SETTINGS = {
    "records_directory": MASS_BASE_DIRECTORY + "records/",  # exemple de nom de fichier
    "annotations_directory": MASS_BASE_DIRECTORY + "annotations/",  # exemple de nom de fichier
    "h5_directory": BASE_DIRECTORY_H5 + "mass/",
    "memmap_directory": BASE_DIRECTORY_MEMMAP + "mass/",
}

MESA_BASE_DIRECTORY = DATASET_DIRECTORY + "mesa/"

MESA_SETTINGS = {
    "records_directory": MESA_BASE_DIRECTORY + "records/",
    "annotations_directory": MESA_BASE_DIRECTORY + "annotations/",
    "index_directory": MESA_BASE_DIRECTORY + "index.json",
    "h5_directory": BASE_DIRECTORY_H5 + "mesa/",
    "memmap_directory": BASE_DIRECTORY_MEMMAP + "mesa/",
}

SHHS_BASE_DIRECTORY = DATASET_DIRECTORY + "shhs/"
SHHS_SETTINGS = {
    "records_directory": SHHS_BASE_DIRECTORY + "records/",
    "annotations_directory": SHHS_BASE_DIRECTORY + "annotations/",
    "h5_directory": BASE_DIRECTORY_H5 + "shhs/",
    "memmap_directory": BASE_DIRECTORY_MEMMAP + "shhs/",
}

MROS_BASE_DIRECTORY = DATASET_DIRECTORY + "mros/"
MROS_SETTINGS = {
    "records_directory": MROS_BASE_DIRECTORY + "records/",
    "annotations_directory": MROS_BASE_DIRECTORY + "annotations/",
    "h5_directory": BASE_DIRECTORY_H5 + "mros/",
    "memmap_directory": BASE_DIRECTORY_MEMMAP + "mros/",
}

if not os.path.isdir(BASE_DIRECTORY):
    os.mkdir(BASE_DIRECTORY)

if not os.path.isdir(BASE_DIRECTORY_H5):
    os.mkdir(BASE_DIRECTORY_H5)

if not os.path.isdir(BASE_DIRECTORY_MEMMAP):
    os.mkdir(BASE_DIRECTORY_MEMMAP)
