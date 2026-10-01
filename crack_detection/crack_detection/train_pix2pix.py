import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import cv2
import numpy as np
import glob
from tqdm import tqdm
import json
import logging
from collections import defaultdict
import matplotlib.pyplot as plt


class ResidualDataset(Dataset):
    """Dataset for Pix2Pix residual learning."""
    
    def __init__(self, data_dir, split='train'):
        self.unet_pred_dir = os.path.join(data_dir, 'unet_predictions')
        self.residual_dir = os.path.join(data_dir, 'residuals')
        
        # Get all prediction files
        self.unet_pred_files = sorted(glob.glob(os.path.join(self.unet_pred_dir, '*.png')))
        self.residual_files = sorted(glob.glob(os.path.join(self.residual_dir, '*.png')))
        
        assert len(self.unet_pred_files) == len(self.residual_files), \
            f"Mismatch: {len(self.unet_pred_files)} predictions vs {len(self.residual_files)} residuals"
        
        print(f"Loaded {len(self.unet_pred_files)} training pairs")
    
    def __len__(self):
        return len(self.unet_pred_files)
    
    def __getitem__(self, idx):
        # Load UNet prediction (input to Pix2Pix)
        unet_pred = cv2.imread(self.unet_pred_files[idx], cv2.IMREAD_GRAYSCALE)
        unet_pred = unet_pred.astype(np.float32) / 255.0  # [0, 1]
        
        # Load residual (target for Pix2Pix)
        residual = cv2.imread(self.residual_files[idx], cv2.IMREAD_GRAYSCALE)
        residual = (residual.astype(np.float32) / 127.5) - 1.0  # [0, 255] -> [-1, 1]
        
        # Convert to tensors and add channel dimension
        unet_pred = torch.from_numpy(unet_pred).unsqueeze(0)  # [1, H, W]
        residual = torch.from_numpy(residual).unsqueeze(0)    # [1, H, W]
        
        return unet_pred, residual


class Generator(nn.Module):
    """U-Net style generator for residual learning."""
    
    def __init__(self, input_channels=1, output_channels=1, ngf=64):
        super().__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip connections
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),  # 16 = 8 + 8 (skip)
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1] for residuals
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.encoder1(x)      # ngf x 192 x 192
        e2 = self.encoder2(e1)     # 2*ngf x 96 x 96
        e3 = self.encoder3(e2)     # 4*ngf x 48 x 48
        e4 = self.encoder4(e3)     # 8*ngf x 24 x 24
        e5 = self.encoder5(e4)     # 8*ngf x 12 x 12
        
        # Bottleneck
        b = self.bottleneck(e5)    # 8*ngf x 6 x 6
        
        # Decoder with skip connections
        d6 = self.decoder6(b)
        d6 = torch.cat([d6, e5], 1)
        
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], 1)
        
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], 1)
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], 1)
        
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], 1)
        
        d1 = self.decoder1(d2)
        
        return d1


class Discriminator(nn.Module):
    """PatchGAN discriminator."""
    
    def __init__(self, input_channels=2, ndf=64):  # 2 = unet_pred + residual
        super().__init__()
        
        # Input: concatenated [unet_prediction, residual]
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False)
            # Removed Sigmoid - using BCEWithLogitsLoss now
        )
    
    def forward(self, unet_pred, residual):
        x = torch.cat([unet_pred, residual], 1)
        return self.model(x)


class Pix2PixTrainer:
    """Complete Pix2Pix training pipeline for residual learning."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Initialize models
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=config['lr'], 
            betas=(0.5, 0.999)
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=config['lr'], 
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
        self.l1_loss = nn.L1Loss()
        
        # Mixed precision
        self.scaler_gen = GradScaler()
        self.scaler_disc = GradScaler()
        
        # Training history
        self.train_history = defaultdict(list)
        
        # Print model info
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Generator parameters: {gen_params:,}")
        print(f"Discriminator parameters: {disc_params:,}")
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_l1_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (unet_pred, real_residual) in enumerate(pbar):
            unet_pred = unet_pred.to(self.device)
            real_residual = real_residual.to(self.device)
            
            batch_size = unet_pred.size(0)
            
            # Real and fake labels
            real_label = torch.ones(batch_size, 1, 46, 46).to(self.device)  # PatchGAN output size
            fake_label = torch.zeros(batch_size, 1, 46, 46).to(self.device)
            
            # Train Discriminator
            self.disc_optimizer.zero_grad()
            
            with autocast():
                # Real pairs
                real_validity = self.discriminator(unet_pred, real_residual)
                d_real_loss = self.adversarial_loss(real_validity, real_label)
                
                # Fake pairs
                fake_residual = self.generator(unet_pred)
                fake_validity = self.discriminator(unet_pred, fake_residual.detach())
                d_fake_loss = self.adversarial_loss(fake_validity, fake_label)
                
                d_loss = (d_real_loss + d_fake_loss) * 0.5
            
            self.scaler_disc.scale(d_loss).backward()
            self.scaler_disc.step(self.disc_optimizer)
            self.scaler_disc.update()
            
            # Train Generator
            self.gen_optimizer.zero_grad()
            
            with autocast():
                fake_residual = self.generator(unet_pred)
                fake_validity = self.discriminator(unet_pred, fake_residual)
                
                # Combined loss
                g_adv_loss = self.adversarial_loss(fake_validity, real_label)
                g_l1_loss = self.l1_loss(fake_residual, real_residual)
                
                g_loss = g_adv_loss + self.config['lambda_l1'] * g_l1_loss
            
            self.scaler_gen.scale(g_loss).backward()
            self.scaler_gen.step(self.gen_optimizer)
            self.scaler_gen.update()
            
            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            running_l1_loss += g_l1_loss.item()
            
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}',
                'L1': f'{g_l1_loss.item():.4f}'
            })
        
        return {
            'g_loss': running_g_loss / len(dataloader),
            'd_loss': running_d_loss / len(dataloader),
            'l1_loss': running_l1_loss / len(dataloader)
        }
    
    def save_checkpoint(self, epoch, save_path, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'train_history': dict(self.train_history),
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
    
    def save_sample_results(self, dataloader, epoch, save_dir):
        """Save sample results during training."""
        self.generator.eval()
        
        with torch.no_grad():
            # Get first batch
            unet_pred, real_residual = next(iter(dataloader))
            unet_pred = unet_pred[:4].to(self.device)  # First 4 samples
            real_residual = real_residual[:4].to(self.device)
            
            # Generate fake residuals
            fake_residual = self.generator(unet_pred)
            
            # Create visualization
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            
            for i in range(4):
                # UNet prediction
                axes[i, 0].imshow(unet_pred[i, 0].cpu().numpy(), cmap='gray')
                axes[i, 0].set_title(f'UNet Pred {i+1}' if i == 0 else '')
                axes[i, 0].axis('off')
                
                # Real residual
                axes[i, 1].imshow(real_residual[i, 0].cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1)
                axes[i, 1].set_title(f'Real Residual {i+1}' if i == 0 else '')
                axes[i, 1].axis('off')
                
                # Fake residual
                axes[i, 2].imshow(fake_residual[i, 0].cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1)
                axes[i, 2].set_title(f'Generated Residual {i+1}' if i == 0 else '')
                axes[i, 2].axis('off')
                
                # Corrected result
                corrected = torch.clamp(unet_pred[i, 0] + fake_residual[i, 0], 0, 1)
                axes[i, 3].imshow(corrected.cpu().numpy(), cmap='gray')
                axes[i, 3].set_title(f'Corrected Result {i+1}' if i == 0 else '')
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'), dpi=150)
            plt.close()
        
        self.generator.train()


def main():
    """Main training function."""
    
    config = {
        'data_dir': 'pix2pix_data',
        'output_dir': 'pix2pix_results',
        'model_dir': 'pix2pix_models',
        'batch_size': 8,  # Increased from 4 to 8
        'num_epochs': 100,
        'lr': 2e-4,
        'lambda_l1': 100.0,  # Standard Pix2Pix L1 weight
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_every': 10,
        'sample_every': 5,
        'num_workers': 2,
    }
    
    # Create directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['output_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Training configuration: {json.dumps(config, indent=2)}")
    
    # Create dataset and dataloader
    dataset = ResidualDataset(config['data_dir'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    logging.info(f"Dataset size: {len(dataset)}")
    
    # Initialize trainer
    trainer = Pix2PixTrainer(config)
    
    # Training loop
    best_l1_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        logging.info(f"Epoch {epoch}/{config['num_epochs']}")
        
        # Train
        metrics = trainer.train_epoch(dataloader, epoch)
        
        # Log metrics
        logging.info(f"G_loss: {metrics['g_loss']:.4f}, "
                    f"D_loss: {metrics['d_loss']:.4f}, "
                    f"L1_loss: {metrics['l1_loss']:.4f}")
        
        # Save history
        trainer.train_history['g_loss'].append(metrics['g_loss'])
        trainer.train_history['d_loss'].append(metrics['d_loss'])
        trainer.train_history['l1_loss'].append(metrics['l1_loss'])
        
        # Save best model based on L1 loss
        is_best = metrics['l1_loss'] < best_l1_loss
        if is_best:
            best_l1_loss = metrics['l1_loss']
            logging.info(f"New best L1 loss: {best_l1_loss:.4f}")
        
        # Save checkpoint
        if epoch % config['save_every'] == 0 or is_best:
            save_path = os.path.join(config['model_dir'], f'pix2pix_epoch_{epoch}.pth')
            trainer.save_checkpoint(epoch, save_path, is_best=is_best)
        
        # Save sample results
        if epoch % config['sample_every'] == 0:
            trainer.save_sample_results(dataloader, epoch, config['output_dir'])
    
    logging.info("Pix2Pix training completed!")


if __name__ == "__main__":
    main()
