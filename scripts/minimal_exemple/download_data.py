import boto3
from botocore import UNSIGNED
from botocore.client import Config
import tqdm
import os
from scripts.datasets_to_h5.cap.to_h5_mne import format_cap_to_h5
from scripts.datasets_to_h5.sleep_edf.to_h5 import format_sleep_edf_to_h5


def download_dodo(settings, force=False):
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_objects = client.list_objects(Bucket='dreem-dod-o')["Contents"]
    print("\n Downloading H5 files and annotations from S3 for DOD-O")
    if not os.path.isdir(settings['h5_directory']):
        os.makedirs(settings['h5_directory'])
    for bucket_object in tqdm.tqdm(bucket_objects):
        filename = bucket_object["Key"]
        if force or not os.path.exists(settings['h5_directory'] + "/{}".format(filename)):
            client.download_file(
                Bucket="dreem-dod-o",
                Key=filename,
                Filename=settings['h5_directory'] + "/{}".format(filename)
            )
    return settings['h5_directory']


def download_dodh(settings, force=False):
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_objects = client.list_objects(Bucket='dreem-dod-h')["Contents"]
    print("\n Downloading H5 files and annotations from S3 for DOD-H")
    if not os.path.isdir(settings['h5_directory']):
        os.makedirs(settings['h5_directory'])
    for bucket_object in tqdm.tqdm(bucket_objects):
        filename = bucket_object["Key"]
        if force or not os.path.exists(settings['h5_directory'] + "/{}".format(filename)):
            client.download_file(
                Bucket="dreem-dod-h",
                Key=filename,
                Filename=settings['h5_directory'] + "/{}".format(filename)
            )
    return settings['h5_directory']


def download_cap(settings, parallel=False):
    from subprocess import call
    out_directory = settings['base_directory'].replace("physionet.org/files/capslpdb/1.0.0/", "")
    call(["wget", "-r", "-N", "-c", "-np", "https://physionet.org/files/capslpdb/1.0.0/", "-P", out_directory])
    settings['base_directory'] = os.path.join(out_directory, 'physionet.org/files/capslpdb/1.0.0/')
    return format_cap_to_h5(settings, parallel=parallel)


def download_sleep_edf(settings, parallel=False):
    from subprocess import call
    out_directory = settings['base_directory'].replace("physionet.org/files/sleep-edfx/1.0.0/", "")
    call(["wget", "-r", "-N", "-c", "-np", "https://physionet.org/files/sleep-edfx/1.0.0/", "-P", out_directory])
    settings['base_directory'] = os.path.join(out_directory, 'physionet.org/files/sleep-edfx/1.0.0/')
    settings['edf_directory'] = [
        os.path.join(out_directory, 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'),
        os.path.join(out_directory, 'physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/'),
    ]
    return format_sleep_edf_to_h5(settings, parallel=parallel)
