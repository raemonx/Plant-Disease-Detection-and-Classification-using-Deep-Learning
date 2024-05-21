import json
import os
import warnings

import fire
import ray
import torch
from ray.tune.schedulers import ASHAScheduler

from inference import predict
from preprocessing import preprocess
from training import train
from utils import save_model, setup_output_dir

warnings.filterwarnings("ignore")

# Ray's old verbosity setting
os.environ['RAY_AIR_NEW_OUTPUT'] = "1"

# Directory paths for the root and models directory
ROOT = os.getcwd()

def run(dataset: str, model: str = None, pretrained: bool = False,
        optimizer: str = "SGD", loss_func = "CrossEntropyLoss",
        max_epochs=2, batch_size: int = 32, lr: float = 0.001,
        tune_hyperparams: bool = False, train_mode=True):
    setup_output_dir()

    # Configure the model filename based on whether it is pretrained or not
    model_filename = f"{model}_{dataset}_{max_epochs}"
    if pretrained:
        model_filename = f"{model}_{dataset}_ft"
    if tune_hyperparams:
        model_filename = f"{model}_{dataset}_ht"

    # Call the preprocess function, preprocess the data and load the train, validation,test datasets and classes
    train_dataset, val_dataset, test_dataset, num_classes = preprocess(ROOT, dataset)

    # Parameters for the training process
    params = {
        "train_ds": train_dataset,
        "valid_ds": val_dataset,
        "test_ds": test_dataset,
        "num_classes": num_classes,
        "model": model,
        "pretrained": pretrained,
        "criterion": loss_func,
        "optimizer": optimizer,
        "lr_scheduler": "ReduceLROnPlateau",
        "num_epochs": max_epochs,
        "tune_hyperparams": tune_hyperparams,
        "model_filename": model_filename
    }

    if train_mode:
        # Check if hyperparameter tuning is requested
        if tune_hyperparams:
            # Set up the hyperparameter tuner with ASHAScheduler
            scheduler = ASHAScheduler(
                max_t=max_epochs,
                grace_period=1,
                reduction_factor=2,
            )
            # Configure the tuner
            tuner = ray.tune.Tuner(
                ray.tune.with_resources(
                    ray.tune.with_parameters(train, params=params),
                    resources={"gpu": 1}
                ),
                tune_config=ray.tune.TuneConfig(
                    metric="loss",
                    mode="min",
                    scheduler=scheduler,
                    num_samples=10,
                ),
                run_config=ray.train.RunConfig(
                    name=f"{model_filename}",
                    storage_path=f"{ROOT}/ray_results",
                    verbose=1
                ),
                param_space=dict(
                    batch_size=ray.tune.choice([8, 16, 32, 64]),
                    lr=ray.tune.uniform(lower=1e-4, upper=1e-2)
                )
            )
            # Run the tuning process
            results = tuner.fit()
            best_result = results.get_best_result("loss", "min")

            # Print the best trial results
            print("Best trial config: {}".format(best_result.config))
            print("Best trial final validation loss: {}".format(
                best_result.metrics["loss"]))
            print("Best trial final validation accuracy: {}".format(
                best_result.metrics["accuracy"]))

            with best_result.checkpoint.as_directory() as checkpoint_dir:
                model = torch.load(os.path.join(checkpoint_dir, f"models/{model_filename}.pth"))
                save_model(model, model_filename)

            with open(os.path.join(ROOT, "models/best_config.json"), "w") as f:
                json.dump(best_result.config, f, indent=4)

        else:
            # Configure hyperparameter tuning from command line input
            config = dict(
                batch_size=batch_size,
                lr=lr
            )

            # Start Training
            train(config, params)

    if os.path.isfile(os.path.join(ROOT, "models/best_config.json")):
        print("Loading optimal values of batch_size for inferencing...")
        with open(os.path.join(ROOT, "models/best_config.json"), "r") as f:
            data = json.load(f)
            batch_size = data['batch_size']

    predict(test_dataset,
            model_filename=model_filename,
            criteria=loss_func,
            batch_size=batch_size)


if __name__ == "__main__":
    fire.Fire(run)
