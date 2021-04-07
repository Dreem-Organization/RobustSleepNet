import os
import h5py
import json
from robust_sleep_net.preprocessings.h5_to_memmap import h5_to_memmaps
from robust_sleep_net.datasets.dataset import DreemDataset
from robust_sleep_net.models.modulo_net import ModuloNet
import tempfile
import shutil


def generate_memmap_description(path_to_h5):
    with h5py.File(path_to_h5, 'r') as h5:
        description = json.loads(h5.attrs['description'])
        signals_for_memmap = []
        signals_name_for_memmap = []
        for signal in description:
            signals_name_for_memmap.append(signal['name'])
            signals_for_memmap.append(signal['path'])
        memmaps_description = {
            "signals": [
                {
                    "name": "eeg",
                    "signals": signals_for_memmap,
                    "signals_name": signals_name_for_memmap,
                    "processings": [
                        {
                            "type": "filter_bandpass",
                            "args": {}
                        },
                        {
                            "type": "poly_resample",
                            "args": {
                                "target_frequency": 60
                            }
                        },
                        {
                            "type": "normalize_signal_IQR",
                            "args": {
                                "clip_IQR": 20
                            }
                        },
                        {
                            "type": "padding",
                            "args": {
                                "padding_duration": 900,
                                "value": 0
                            }
                        }
                    ]
                }
            ],
            "features": []
        }
        return memmaps_description


def inference_on_h5(path_to_h5, path_to_model):
    memmap_description = generate_memmap_description(path_to_h5)
    memmap_directory = tempfile.mkdtemp()
    memmap_folder, groups_description, features_description = h5_to_memmaps(
        records=[path_to_h5],
        memmap_description=memmap_description,
        memmap_directory=memmap_directory,
        num_workers=1,
        error_tolerant=False,
    )

    padding = [groups_description[x]['padding'] for x in groups_description]
    if len(padding) == 0:
        padding = [features_description[x]['padding'] for x in features_description]
    try:
        padding = padding[0]
        padding = padding // 30
    except IndexError:
        raise IndexError('Could not find padding values')
    records = [f"{memmap_folder}/{x}/" for x in os.listdir(memmap_folder) if '.json' not in x]
    inference_dataset = DreemDataset(groups_description,features_description={},
                                     temporal_context=21, records=records)
    net = ModuloNet.load(path_to_model)
    net.eval()
    hypnodensity = net.predict_on_dataset(inference_dataset,return_prob=True,mode="arithmetic")[inference_dataset.records[0]]
    hypnogram = hypnodensity.argmax(1)
    shutil.rmtree(memmap_directory)
    if padding > 0:
        return hypnogram[padding:-padding],hypnodensity[padding:-padding]
    else:
        return hypnogram, hypnodensity


