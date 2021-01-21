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
