import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision import models

from prepare_data import upload_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
NUM_CLASSES = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 7
LOG_EVERY_N_BATCHES = 50

DATA_DIR = Path("data/processed")
CLASS_MAPPING_PATH = DATA_DIR / "class_mapping.json"
EXPERIMENT_NAME = "concrete-crack-classification"
REGISTERED_MODEL_NAME = "concrete-crack-detector"

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


def return_data_loaders():
    loaders, class_mapping = upload_data()
    train_loader, val_loader, test_loader = loaders
    return train_loader, val_loader, test_loader


def prepare_model():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

    return model


def evaluate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            preds_all.extend(preds)
            targets_all.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(targets_all, preds_all),
        "f1_score": f1_score(targets_all, preds_all, pos_label=0),
        "recall": recall_score(targets_all, preds_all, pos_label=0),
        "precision": precision_score(targets_all, preds_all, pos_label=0),
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Used device: %s", device)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    loss_fn = torch.nn.CrossEntropyLoss()
    model = prepare_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader, test_loader = return_data_loaders()

    assert (
        CLASS_MAPPING_PATH.exists()
    ), f"{CLASS_MAPPING_PATH} doesn't exist - run prepare_data.py. "

    best_val_f1 = -1.0
    best_state_dict = None
    epochs_without_improvement = 0
    global_step = 0

    with mlflow.start_run(run_name="mobilenet_v3_small") as run:
        mlflow.log_param("model_type", "mobilenet_v3_small")
        mlflow.log_param("epochs", NUM_EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("early_stopping_patience", EARLY_STOPPING_PATIENCE)

        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                global_step += 1

                if batch_idx % LOG_EVERY_N_BATCHES == 0:
                    mlflow.log_metric("batch_loss", loss.item(), step=global_step)

            epoch_loss = running_loss / len(train_loader.dataset)
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

            val_metrics = evaluate(model, val_loader, device)
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_f1_score", val_metrics["f1_score"], step=epoch)
            mlflow.log_metric("val_recall", val_metrics["recall"], step=epoch)
            mlflow.log_metric("val_precision", val_metrics["precision"], step=epoch)

            logger.info(
                "Epoch [%d/%d] train_loss=%.4f val_f1=%.4f val_recall=%.4f",
                epoch + 1,
                NUM_EPOCHS,
                epoch_loss,
                val_metrics["f1_score"],
                val_metrics["recall"],
            )

            if val_metrics["f1_score"] > best_val_f1:
                best_val_f1 = val_metrics["f1_score"]
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state_dict, "best_model_checkpoint.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                logger.info(
                    "No improvements in val_f1 since %d epochs - stopping training on epoch %d",
                    EARLY_STOPPING_PATIENCE,
                    epoch + 1,
                )
                break
        mlflow.log_metric("epochs_trained", epoch + 1)

        model.load_state_dict(best_state_dict)
        mlflow.log_metric("best_val_f1", best_val_f1)

        test_metrics = evaluate(model, test_loader, device)
        mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
        mlflow.log_metric("test_f1_score", test_metrics["f1_score"])
        mlflow.log_metric("test_recall", test_metrics["recall"])
        mlflow.log_metric("test_precision", test_metrics["precision"])

        logger.info(
            "Results on test dataset - accuracy=%.4f f1=%.4f recall=%.4f precision=%.4f",
            test_metrics["accuracy"],
            test_metrics["f1_score"],
            test_metrics["recall"],
            test_metrics["precision"],
        )

        model_cpu = model.to("cpu")
        model_cpu.eval()

        sample_images, _ = next(iter(train_loader))
        input_example = sample_images[:1].numpy()

        with torch.no_grad():
            output_example = model_cpu(torch.from_numpy(input_example)).numpy()

        signature = infer_signature(model_input=input_example, model_output=output_example)

        mlflow.pytorch.log_model(
            pytorch_model=model_cpu,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="pickle",
        )

        mlflow.log_artifact(str(CLASS_MAPPING_PATH), artifact_path="metadata")

        run_id = run.info.run_id

    logger.info("Registering model in MLflow, run_id=%s", run_id)
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_details.version,
    )
    logger.info(
        "Registered as %s version %s with alias 'champion'",
        REGISTERED_MODEL_NAME,
        model_details.version,
    )


if __name__ == "__main__":
    train()
