import numpy


def split_tvt_set(data, ratio, seed=-1):
    tot = sum(ratio)
    train_size, validation_size, test_size = (len(
        data)*ratio[0]) // tot, (len(data)*ratio[1]) // tot, (len(data)*ratio[2]) // tot
    if seed == -1:
        seed = numpy.random.random_integers(0, 1000)
    numpy.random.seed(seed)
    indices = numpy.random.permutation(len(data))
    return data.iloc[indices[0:train_size]], data.iloc[indices[train_size:train_size + validation_size]], data.iloc[
        indices[test_size:]]


def split_from_labels(data, labels, axis=1):
    return data.drop(labels, axis), data[labels].copy()
