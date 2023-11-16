from simpson_classifier_lightning.model import MyCnn
from simpson_classifier_lightning.datasets import get_datasets
import argparse
import torch
import uuid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model in Simpson's dataset")

    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--optimizer', choices=['AdamW', 'SGD', 'RMSprop', 'Adagrad'], default='AdamW')
    parser.add_argument('--criterion', choices=['CrossEntropy', 'NLLLoss', 'KLDivLoss'], default='CrossEntropy')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--balanced', type=bool, default=False)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device=='gpu':
        raise SystemError('Dont see CUDA')

    name_run = str(uuid.uuid4())

    if not os.path.isdir(SOURCE_DIR + "weights"):
        os.mkdir(SOURCE_DIR + "weights")

    if not os.path.isdir(SOURCE_DIR + "weights/" + name_run):
        os.mkdir(SOURCE_DIR + "weights/" + name_run)

    dm = SimpsonDataModule(args.batch, args.balanced)
    model = MyModel(42, args.optimizer, args.criterion)

    trainer = L.Trainer(
        max_epochs=args.epoch,
        accelerator=arg.device,
        devices=1,
        weights_save_path=SOURCE_DIR + "weights/" + name_run
    )
    trainer.fit(model, dm)
