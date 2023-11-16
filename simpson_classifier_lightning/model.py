import torch
import torch.nn as nn
import lightning as L


class MyModel(L.LightningModule):
    def __init__(self, num_classes=42, optimizer_name='AdamW', criterion_name='CrossEntropy'):
        super().__init__()

        self.num_classes = num_classes
        self.optimizer_name = optimizer_name
        self.loss = None

        if criterion_name == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss()
        elif criterion_name == 'NLLLoss':
            self.loss = nn.NLLLoss()
        elif criterion_name == 'KLDivLoss':
            self.loss = KLDivLoss(reduction='none', log_target=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(2048, n_classes),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = nn.functional.interpolate(x, size=(4, 4), align_corners=False, mode='bilinear')
        x = x.view(x.size(0), 4 * 4 * 256)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        logs={"train_loss": loss}
        output={
              "loss": loss,
              "log": logs
          }
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer_name == 'AdamW':
            return torch.optim.AdamW(self.parameters(), amsgrad=True)
        elif self.optimizer_name == 'SGD':
            return torch.optim.SGD(self.parameters())
        elif self.optimizer_name == 'RMSprop':
            return torch.optim.RMSprop(self.parameters())
        elif self.optimizer_name == 'Adagrad':
            return torch.optim.Adagrad(self.parameters())
