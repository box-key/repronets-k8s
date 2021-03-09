import csv


def read_examples(data_path):
    """Read a wd_*.f12 file in NET-COLING2018 dataset and returns as a list."""
    src_examples, tgt_examples = [], []
    with open(data_path, 'r', encoding='utf-8') as lines:
        for line in lines:
            if line:
                items = line.split()
                src_ex = list(items[0].strip())
                tgt_ex = items[1:]
                src_examples.append(src_ex)
                tgt_examples.append(tgt_ex)
    assert len(src_examples) == len(tgt_examples)
    return src_examples, tgt_examples


def write_examples(examples, data_path):
    """Write a sample file in the opennmt format."""
    with open(data_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(' '.join(example) + '\n')


def read_test(data_path):
    """Read a wd_*.f1 file in NET-COLING2018 dataset and returns as a list."""
    test_examples = []
    with open(data_path, 'r', encoding='utf-8') as lines:
        for line in lines:
            if line:
                test_ex = list(line.strip())
                test_examples.append(test_ex)
    return test_examples


def make_truth(src_test_path, output_path):
    """Create a *.words file from an opennmt source file for scoring."""
    # read test file
    with open(src_test_path, 'r', encoding='utf-8') as lines:
        samples = [''.join(line).replace(' ', '') for line in lines if line]
    # write samples
    with open(output_path, 'w', encoding='utf-8') as f:
        for x in samples:
            f.write(x)


def make_response(src_truth_path, prediction_path, output_path, beam_size=3):
    """Create a *.decoded file from opennmt translation file (specified by
    prediction_path) and *.words file (specified by src_truth_path).
    """
    with open(src_truth_path, 'r', encoding='utf-8') as lines:
        samples = [line.replace('\n', '') for line in lines if line]
    with open(prediction_path, 'r', encoding='utf-8') as lines:
        predictions = [line.replace('\n', '') for line in lines if line]
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(len(predictions)):
            sample_idx = i // beam_size
            line = '{}\t{}\n'.format(samples[sample_idx], predictions[i])
            f.write(line)


def merge_samples(original_sample_path, fixed_sample_path, output_path=None):
    """Merge corrected transliterations into the original dataset.
    Samples are matched based on indices.

    Parameters
    ----------
    * original_sample_path : str
        A tsv file which follows the following format:
        <index>\\t<source>\\t<target>\\t<freq>
    * fixed_sample_path : str
        A tsv file which follows the following format:
        <index>\\t<source>\\t<target>
    * output_path : str (optional)
        If specified, output merged samples in the following format:
        <source>\\t<target>\\t<freq>

    Output
    ------
    correct_samples : dict of dict
        A dictionary where the key is index and the value is a dictionary:
        {'source': str, 'target': str, 'freq': int}
    """
    # get original samples (must be indexed)
    original_samples = {}
    with open(original_sample_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line:
                items = line.split('\t')
                idx = int(items[0])
                original_samples[idx] = {
                    'source': items[1],
                    'target': items[2].replace('\n', ''),
                    'freq': int(items[3])
                }
    # get manually fixed samples
    fixed_samples = {}
    with open(fixed_sample_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line:
                items = line.split('\t')
                idx = int(items[0])
                fixed_samples[idx] = {
                    'source': items[1],
                    'target': items[2].replace('\n', '')
                }
    # merge files
    # warning: shallow copy!
    correct_samples = original_samples
    for i, correct_sample in fixed_samples.items():
        correct_samples[i]['source'] = correct_sample['source']
        correct_samples[i]['target'] = correct_sample['target']
    # output merged samples as a tsv file if output_path is specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            writer = csv.writer(f, delimiter='\t')
            for _, sample in correct_samples.items():
                writer.writerow([sample['source'], sample['target'], sample['freq']])
    return correct_samples


def remove_bad_samples(original_sample_path, bad_sample_path, output_path=None):
    """Remove samples listed in a file indicated by `bad_sample_path`.

    Parameters
    ----------
    * original_sample_path : str
        A tsv file which follows the following format:
        <index>\\t<source>\\t<target>\\t<freq>
    * fixed_sample_path : str
        A tsv file which follows the following format:
        <index>\\t<source>\\t<target>
    * output_path : str (optional)
        If specified, output merged samples in the following format:
        <source>\\t<target>\\t<freq>

    Output
    ------
    correct_samples : dict of dict
        A dictionary where the key is index and the value consists of {'source': str, 'target': str}

    """
    # get manually fixed samples
    bad_sample_indices = []
    with open(bad_sample_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line:
                items = line.split('\t')
                idx = int(items[0])
                bad_sample_indices.append(idx)
    bad_sample_index_set = set(bad_sample_indices)
    # get original samples (must be indexed)
    good_samples = {}
    with open(original_sample_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line:
                items = line.split('\t')
                idx = int(items[0])
                if idx not in bad_sample_index_set:
                    good_samples[idx] = {
                        'source': items[1],
                        'target': items[2].replace('\n', ''),
                        'freq': int(items[3])
                    }
    # output merged samples as a tsv file if output_path is specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            writer = csv.writer(f, delimiter='\t')
            for _, sample in good_samples.items():
                writer.writerow([sample['source'], sample['target'], sample['freq']])
    return good_samples
