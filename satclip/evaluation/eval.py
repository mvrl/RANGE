import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
import torchmetrics

from sklearn.preprocessing import MinMaxScaler
import argparse

#local import 
from ..utils.load_model import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='code for evaluating the embeddings')
    parser.add_argument('--ckpt_path', type=str, help='Path to the pretrained model',
    default='/projects/bdec/adhakal2/hyper_satclip/logs/SAPCLIP/0tflzztx/checkpoints/epoch=162-acc_eco=0.000.ckpt')
    parser.add_argument('--task_name', type=str, help='Name of the task', default='biome')
    parser.add_argument('--log_dir', type=str, help='Path to the log directory', default='./')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=6)
    args = parser.parse_args()

    return args
    
# NN for linear probe of embeddings for classification tasks
class ClassificationNet(L.LightningModule):
    def __init__(self,
     model,
     input_dims: int=256,
     output_dims: int=10,
     **kwargs):
        super().__init__()
        self.location_encoder = model.eval()
        #freeze the model
        for param in self.location_encoder.parameters():
            param.requires_grad=False
        
        self.linear = torch.nn.Linear(input_dims, output_dims)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_dims)
        #save all values for acc calculation
        self.true_labels = []
        self.predicted_labels = []
        
    def forward(self, x):
        location_embeddings = self.location_encoder(x)
        out = self.linear(location_embeddings)
        return out

    def shared_step(self, batch):
        x,y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def val_loss(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        #save all values for acc calculation
        self.true_labels.append(y)
        y_hat = torch.argmax(y_hat, dim=1)
        self.predicted_labels.append(y_hat)
        return loss
    
    def on_validation_epoch_end(self):
        predicted_labels = torch.cat(self.predicted_labels)
        true_labels = torch.cat(self.true_labels)
        acc = self.acc(predicted_labels, true_labels)
        self.log('val_acc', acc, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)



if __name__ == '__main__':
    args = get_args()
    sapclip_model = load_checkpoint(args.ckpt_path)
    model = ClassificationNet(sapclip_model, 256, 10)
    import code; code.interact(local=dict(globals(), **locals()))
    x = torch.rand(64, 256)
    y = torch.randint(0, 10, (64,))
    out = model((x,y))
    
   