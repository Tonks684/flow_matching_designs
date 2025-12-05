import torch

from flow_matching_designs.sampling.sampler import MNISTSampler, Sampleable


def test_mnist_sampler_is_sampleable_subclass():
    assert issubclass(MNISTSampler, Sampleable)


def test_mnist_sampler_sample_shapes_and_labels():
    sampler = MNISTSampler(data_root="./data")
    sampler = sampler.to("cpu")  # ensure dummy buffer is on CPU

    num_samples = 4
    x, y = sampler.sample(num_samples)

    assert x.shape[0] == num_samples
    assert y.shape == (num_samples,)

    # MNIST is grayscale 32x32 in this project
    assert x.ndim == 4
    assert x.shape[1] == 1
    assert x.shape[2] == 32
    assert x.shape[3] == 32

    # labels in range [0, num_classes-1]
    assert y.min().item() >= 0
    assert y.max().item() < sampler.num_classes

    # samples live on same device as sampler.dummy
    assert x.device == sampler.dummy.device
    assert y.device == sampler.dummy.device
