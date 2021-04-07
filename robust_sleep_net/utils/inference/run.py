from robust_sleep_net.utils.inference.edf_to_h5 import edf_to_h5
from robust_sleep_net.utils.inference.inference import inference_on_h5
import tempfile
import shutil
import os
import numpy as np
import datetime as dt
import pytz
import click
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except:
    dir_path = "robust_sleep_net/utils/inference"

def hypnogram_to_txt(hypnogram, start_time, filename,electrodes):
    converter = {
        -1: "MT",
        0: "WAKE",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
    }

    with open(filename, "w") as f:
        f.write("Dreem Hypnogram Export\n")
        f.write(
            "Recording Date: {}\n".format(start_time.strftime("%D"))
        )
        f.write("\n")
        f.write("Events Included: {}\n".format([x for x in converter.values()]))
        f.write("Channels Included: {}\n".format(electrodes))
        f.write(
            "Start Time: {}\n".format(start_time.strftime("%D - %T"))
        )
        f.write("\n")
        f.write("Time [hh:mm:ss]\tSleep Stage\tDuration[s]\n")
        for i, stage in enumerate(hypnogram):
            f.write(
                "{}\t{}\t30\n".format(
                    (start_time + dt.timedelta(seconds=i * 30)).strftime("%T"),
                    converter[stage]
                )
            )

    return filename


def score_edf_record(path_to_edf, electrodes, outfolder=None, timezone='Europe/London', lights_on=None,
                     lights_off=None, start_minute=False, start_30s=False,consensus_of_models = False):
    if lights_off is not None:
        assert start_minute is False and start_30s is False, 'When lights on are provided, start_minute and start_30s must be false'

    assert start_minute is False or start_30s is False, "start_minute and start_30s can't both be True"
    if lights_on is not None:
        assert isinstance(lights_on, int), 'lights_on must be a timestamp'
    if lights_off is not None:
        assert isinstance(lights_off, int), 'lights_off must be a timestamp'

    if outfolder is None:
        outfolder = os.path.join('/', *path_to_edf.split('/')[:-1])

    temp_dir = tempfile.mkdtemp()
    path_to_record, start_time, stop_time, electrodes = edf_to_h5(path_to_edf,
                                                      h5_filename=f"{temp_dir}/record.h5",
                                                      electrodes=electrodes, force=True, lights_on=lights_on,
                                                      lights_off=lights_off, start_minute=start_minute,
                                                      start_30s=start_30s)

    start_time = pytz.timezone('UTC').localize(
        dt.datetime.utcfromtimestamp(start_time)).astimezone(pytz.timezone(timezone))

    hypnograms, hypnodensities = [], []

    if consensus_of_models:
        models_to_use = os.listdir(os.path.join(dir_path,"pretrained_models/"))
    else:
        models_to_use = os.listdir(os.path.join(dir_path, "pretrained_models/"))[:1]

    for model in models_to_use:
        try:
            hypnogram, hypnodensity = inference_on_h5(path_to_record,
                                                      os.path.join(dir_path, f"pretrained_models/{model}/best_model.gz"))
            hypnograms.append(hypnogram)
            hypnodensities.append(hypnodensity)
        except:
            pass

    hypnodensities_unreduced = np.array(hypnodensities)
    hypnodensities = hypnodensities_unreduced.mean(0)
    hypnograms = hypnodensities.argmax(1)
    shutil.rmtree(temp_dir)
    hypnogram_to_txt(hypnograms, start_time, f"{outfolder}/hypnogram.txt", electrodes)


@click.command()
@click.argument("path_to_edf")
@click.option("--electrodes",default = None, help='List of electrodes to use to perform the sleep staging.')
@click.option("--outfolder", help='Folder where the hypnogram will be save')
@click.option("--timezone", default='Europe/London', help='Timezone of the EDF file')
@click.option("--lights_on", default=None, help='Light on time (timestamp format)')
@click.option("--lights_off", default=None, help='Light off time (timestamp format)')
@click.option("--start_minute", default=False, help='Should the staging start on round minutes ?')
@click.option("--start_30s", default=False, help='Should the staging start on round 30 seconds ?')
@click.option("--consensus_of_models", default=False, help='Use the consensus from several model or a single model')
def main(path_to_edf, electrodes, outfolder, timezone, lights_on,
         lights_off, start_minute, start_30s,consensus_of_models):
    score_edf_record(path_to_edf, electrodes, outfolder, timezone, lights_on,
                     lights_off, start_minute, start_30s,consensus_of_models)


if __name__ == '__main__':
    record = '/media/antoine/2Tb-ext/data/datasets/shhs/records/shhs2-200077.edf'
    derivations = ['EEG', 'EEG(sec)', 'EOG(L)', 'EOG(R)']
    score_edf_record(record, derivations)
