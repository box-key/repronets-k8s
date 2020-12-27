def print_shape(**kwargs):
    for name, tensor in kwargs.items():
        print("tensor = '{}' has shape = {}".format(name, tensor.shape))
