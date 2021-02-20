import utils

from pathlib import Path
import csv


logger = utils.get_logger('DEBUG', __name__)


def main():
    # path to data folder
    data_path = Path('.').resolve().parent / 'data'
    logger.debug("data_path = '{}'".format(data_path))
    files = [
        {"read": 'wd_korean_16.f12', "write": 'wd_korean_16_indexed.f12'},
        {"read": 'wd_korean_64.f12', "write": 'wd_korean_64_indexed.f12'}
    ]
    # iterate through files
    for file in files:
        samples = []
        printed = False
        input_path = data_path / file['read']
        logger.info("Reading '{}'".format(input_path))
        # read samples
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line:
                    items = line.split()
                    # store a list as a string
                    sample = (idx, items[0], ' '.join(items[1:]))
                    samples.append(sample)
                    if not printed:
                        logger.info("Sample = '{}'".format(sample))
                        printed = True
        # write samples with row indexes
        output_path = data_path / file['write']
        logger.info("Writing a file to '{}'".format(output_path))
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            for sample in samples:
                writer.writerow(sample)


if __name__ == "__main__":
    main()
