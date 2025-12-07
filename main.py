from typing import Final, Annotated, Literal
import os
from datetime import datetime
from pathlib import Path
import typer
from pytorch_lightning import Trainer
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
import lightning as L
from bayesian_optimizer import optimize
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from torch.utils.tensorboard import SummaryWriter
from trainer import LitResNet, split_dataset, StopOnLowValLoss
from model import ResNet

app = typer.Typer()


NORMALIZE_MAP: Final[dict[str, transforms.Normalize]] = {
    "fashion-mnist": transforms.Normalize(mean=(0.286,), std=(0.353,)),
}


BATCH_SIZE = 128 if torch.cuda.is_available() else 64

try:
    DEFAULT_CPU_COUNT = int(os.cpu_count() / 2)  # type: ignore[operator]
except TypeError:
    DEFAULT_CPU_COUNT = 4
writer = SummaryWriter()


@app.command()
def train(
    n_init_samples: Annotated[
        int, typer.Option(help="number of initial samples to generate for the GP")
    ] = 3,
    n_bayesian_optimizer_runs: Annotated[
        int, typer.Option(help="How many iterations to run the optimizer")
    ] = 10,
    dataset: Annotated[
        Literal["fashion-mnist"], typer.Option(help="Dataset to use")
    ] = "fashion-mnist",
    dataset_save_path: Annotated[
        str, typer.Option(help="path to save dataset")
    ] = "./data",
    number_of_workers: Annotated[
        int, typer.Option(help="number of CPU workers to load dataset")
    ] = DEFAULT_CPU_COUNT,
    device_number: Annotated[
        int | None,
        typer.Option(help="device to use for training, useful for multi-gpu machines"),
    ] = None,
    max_epochs: Annotated[
        int, typer.Option(help="number of epochs to run each experiment")
    ] = 10,
    seed: Annotated[int, typer.Option(help="seed value when running experiments")] = 8,
    plot_graph_location: Annotated[
        str,
        typer.Option(help="filename of the bayesian optimization plots"),
    ] = f"bayesian-opt-{datetime.now().strftime('%Y%m%d%H%M')}.pdf",
    precision: Annotated[
        str, typer.Option(..., formats=["16-true", "32-true"])
    ] = "32-true",
    plot_graph: Annotated[
        bool,
        typer.Option(
            help="To save bayesian optimize plots. If yes, plots will be saved"
        ),
    ] = True,
):
    """runs the whole bayesian optimization process;
    trains the model, predicts the learning rate
    that results in a minimum validation loss
    """
    root_path = Path(".")
    model_save_path = Path(root_path / "models/")  # multiclass classification
    Path(model_save_path).mkdir(parents=False, exist_ok=True)
    L.seed_everything(seed, workers=True)

    normalize = NORMALIZE_MAP[dataset]

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    dataset_train = datasets.FashionMNIST(
        dataset_save_path, train=True, download=True, transform=train_transforms
    )
    dataset_val = datasets.FashionMNIST(
        dataset_save_path, train=True, download=True, transform=test_transforms
    )
    dataset_train = split_dataset(dataset_train)
    dataset_val = split_dataset(dataset_val, train=False)
    # dataset_test = datasets.FashionMNIST(
    #     dataset_save_path, train=False, download=True, transform=test_transforms
    # )

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=number_of_workers,
    )
    val_dataloader = DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=number_of_workers
    )
    # test_dataloader = DataLoader(
    #     dataset_test,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=number_of_workers,
    # )

    if device_number is None:
        device_number = 0
    resnet_model = ResNet(num_blocks=[3, 3])

    logger = TensorBoardLogger(save_dir=model_save_path, name="tensorboad_logs")

    def objective_func(lr: float, epochs: int = max_epochs):
        trainer = Trainer(
            accelerator="gpu",
            devices=[device_number],
            enable_checkpointing=True,
            precision=precision,
            enable_model_summary=True,
            enable_progress_bar=True,
            max_epochs=epochs,
            gradient_clip_val=1.0,  # clip gradients to prevent exploding gradients
            callbacks=[
                EarlyStopping(monitor="val_step_loss", patience=10, mode="min"),
                StopOnLowValLoss(target_loss=0.1),
                ModelSummary(max_depth=1),
            ],
            logger=logger,
        )
        model = LitResNet(model=resnet_model, lr=lr)
        # summary_str = summarize(model, max_depth=-1)  # -1 = full depth
        # print(summary_str)
        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )
        # output = trainer.test(model, dataloaders=test_dataloader)
        out = trainer.callback_metrics["val_loss_min"]
        return out.item()

    optimize(
        objective_func=objective_func,
        n_init_samples=n_init_samples,
        iterations=n_bayesian_optimizer_runs,
        plot_graph_location=plot_graph_location if plot_graph else None,
    )


@app.command(name="infer")
def sample_inference():
    print("Hello from bayesian-optimization!")
    raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
    app()
