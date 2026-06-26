from dataclasses import dataclass, field


@dataclass
class Config:
    # Data
    dataset: str = "cone"  # "gaussians" | "cylinder" | "cone" | "mnist" | "fashion_mnist"
    activation: str = "relu"  # "relu" | "tanh" | "sigmoid" | "gelu" | "leaky_relu" | "elu"
    d: int = 100                # input dimension (overridden to 784 for mnist)
    K: int = 2                  # number of classes (overridden by mnist_classes for mnist)
    n_per_class: int = 500
    val_frac: float = 0.2       # fraction of data held out for validation
    noise_std: float = 0.1
    data_seed: int = 42    # controls dataset generation and DataLoader shuffle order
    model_seed: int = 0    # controls model weight initialisation

    # Gaussians-specific
    cov_rank: int = 3     # rank of low-rank covariance component

    # Cylinder-specific
    # K equal sectors of the circle define the K classes;
    # noise_std controls radial noise around the unit circle

    # MNIST-specific
    mnist_classes: list = field(default_factory=lambda: list(range(10)))
    data_dir: str = "data"      # where torchvision downloads MNIST

    # Model
    H: int = 512          # width of each hidden layer
    depth: int = 1        # number of hidden layers

    # Training
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    optimizer: str = "adam"   # "adam" | "sgd"
    momentum: float = 0.9     # only used when optimizer="sgd"

    # Snapshots
    snapshot_every: int = 1

    # Output
    output_dir: str = "output"
