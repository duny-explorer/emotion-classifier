from simpson_classifier_lightning.model import MyModel
from simpson_classifier_lightning.dataset import SimpsonDataModule
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import uuid
import wandb
import os


class LogPredictionSamplesCallback(Callback):
    def __init__(self, val_samples, logger, n=10):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.n = n
        self.wandb_logger=logger
        
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)

        images = [img for img in val_imgs[:self.n]]
        captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' 
                    for y_i, y_pred in zip(val_labels[:self.n], preds[:self.n])]
            
        self.wandb_logger.log_image(
                key='simpson_val', 
                images=images, 
                caption=captions)

        columns = ['image', 'ground truth', 'prediction']
        data = [[wandb.Image(x_i), y_i, y_pred]
                for x_i, y_i, y_pred in list(zip(val_imgs[:self.n],
                                                val_labels[:self.n], preds[:self.n]))]
        self.wandb_logger.log_table(
                key='simpson_val',
                columns=columns,
                data=data)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model in Simpson's dataset")

    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--optimizer', choices=['AdamW', 'SGD', 'RMSprop', 'Adagrad'], default='AdamW')
    parser.add_argument('--criterion', choices=['CrossEntropy', 'NLLLoss', 'KLDivLoss'], default='CrossEntropy')
    parser.add_argument('--scheduler', choices=['LambdaLR', 'MultiplicativeLR', 'StepLR',
                                                'MultiStepLR', 'ExponentialLR',
                                                 None], default=None)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--balanced', type=bool, default=False)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device=='gpu':
        raise SystemError('Dont see CUDA')

    name_run = str(uuid.uuid4()) + '_lighthing'
    SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

    if not os.path.isdir(SOURCE_DIR + "weights"):
        os.mkdir(SOURCE_DIR + "weights")

    if not os.path.isdir(SOURCE_DIR + "weights/" + name_run):
        os.mkdir(SOURCE_DIR + "weights/" + name_run)

    dm = SimpsonDataModule(args.batch, args.balanced)
    dm.setup()
    model = MyModel(42, args.optimizer, args.criterion, args.scheduler)
    wandb_logger = WandbLogger(project="simpsons-classifier", name=name_run, log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max",
                                          dirpath=SOURCE_DIR + "weights/" + name_run,
                                          filename='{epoch}-max-{val_acc:.4f}', save_weights_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')

    val_samples = next(iter(dm.val_dataloader()))
    
    trainer = Trainer(
        max_epochs=args.epoch,
        accelerator=args.device,
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LogPredictionSamplesCallback(val_samples, wandb_logger), early_stopping_callback]
    )
    trainer.fit(model, dm)
    wandb.finish()
