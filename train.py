import torch
from dataset import *
import sys
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import *
from generator import *
from torch.utils.tensorboard import SummaryWriter
import random
import os
from torchvision.utils import make_grid


class BubbleModel(object):
    def __init__(self,args):

        # make necessary directory
        os.makedirs('loss/'+args.task, exist_ok=True)
        os.makedirs('saved_images/'+args.task, exist_ok=True)
        os.makedirs('saved_models/'+args.task, exist_ok=True)

        if args.gen_mask_type == 'unet':
            self.image2mask = UnetGenerator(
                img_channels=args.image_channel,
                out_channels=args.mask_channel,
            ).to(args.device)
        elif args.gen_mask_type == 'resnet':
            self.image2mask = ResNet(
                img_channels=args.image_channel,
                out_channels=args.mask_channel,
            ).to(args.device)

        if args.gen_image_type == 'unet':
            self.mask2image = UnetGenerator(
                img_channels=args.mask_channel,
                out_channels=args.image_channel,
            ).to(args.device)
        elif args.gen_image_type == 'resnet':
            self.mask2image = ResNet(
                img_channels=args.mask_channel,
                out_channels=args.image_channel,
            ).to(args.device)
        self.dis_image = Discriminator(in_channels=args.image_channel).to(args.device)
        self.dis_mask = Discriminator(in_channels=args.mask_channel).to(args.device)

        # freeze pretrained model weights

        if args.load_pretrain:
            freeze_layers(self.image2mask.initial)
            freeze_layers(self.image2mask.down_blocks)
            freeze_layers(self.image2mask.res_blocks)
            freeze_layers(self.image2mask.up_blocks)
            freeze_layers(self.mask2image.initial)
            freeze_layers(self.mask2image.down_blocks)
            freeze_layers(self.mask2image.res_blocks)
            freeze_layers(self.mask2image.up_blocks)

        elif args.load_pretrain == 'False':
            self.generator_opt = optim.Adam(
                list(self.image2mask.parameters()) + list(self.mask2image.parameters()),
                lr=args.lr,
                betas=(0.5, 0.999),
            )

        self.generator_opt = optim.Adam(
            list(
                filter(lambda p: p.requires_grad, self.image2mask.parameters())
            ) +
            list(
                filter(lambda p: p.requires_grad, self.mask2image.parameters())
            ),
            lr=args.lr,
            betas=(0.5, 0.999),
        )

        self.discriminator_opt = optim.Adam(
            list(self.dis_mask.parameters()) + list(self.dis_image.parameters()),
            lr=args.lr,
            betas=(0.5, 0.999),
        )
        
        self.train_dataset = BubbleDataset(
            args=args,
            mask_root=args.mask_dir,
            image_root=args.image_dir,
            func='train',
            transform=True,
            cropping=args.cropping,
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.val_dataset = BubbleDataset(
            args=args,
            mask_root=args.test_mask_dir,
            image_root=args.test_image_dir,
            # func='validation',
            func='testing',
            transform=True,
            cropping=False,
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.test_dataset = BubbleDataset(
            args=args,
            mask_root=args.test_mask_dir,
            image_root=args.test_image_dir,
            func='testing',
            transform=True,
            cropping=False
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        self.generator_loss = 0
        self.discriminator_loss = 0
        self.val_gloss = 0
        self.val_dloss = 0
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.epoch = 0
        self.best_generator_loss = 1
        self.best_discriminator_loss = 0

        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

        self.writer_disloss = []
        self.writer_genloss = []
        self.writer_cycleloss = []
        self.writer_shapeloss = []

        self.checkpoint_gmask = 'saved_models/' + args.task + '/genmask.pth.tar'
        self.checkpoint_gimage = 'saved_models/' + args.task + '/genimage.pth.tar'
        self.checkpoint_dmask = 'saved_models/' + args.task + '/dismask.pth.tar'
        self.checkpoint_dimage = 'saved_models/' + args.task + '/disimage.pth.tar'
        self.pretrain_gmask = f'pretrain/{args.pretrain_dir}/genmask.pth.tar'
        self.pretrain_gimage = f'pretrain/{args.pretrain_dir}/genimage.pth.tar'




    def load_model(self,args):
        if args.load_model:
            print('=> loading checkpoints')
            self.epoch = load_checkpoint(
                self.checkpoint_gmask, self.image2mask, self.generator_opt, args.lr,
            )
            self.epoch = load_checkpoint(
                self.checkpoint_gimage, self.mask2image, self.generator_opt, args.lr,
            )
            self.epoch = load_checkpoint(
                self.checkpoint_dmask, self.dis_mask, self.discriminator_opt, args.lr,
            )
            self.epoch = load_checkpoint(
                self.checkpoint_dimage, self.dis_image, self.discriminator_opt, args.lr,
            )

    def save_model(self,args):
        if args.save_model:
            print('=> saving checkpoints')
            save_checkpoint(model=self.image2mask, optimizer=self.generator_opt, epoch=self.epoch,
                            filename=self.checkpoint_gmask)
            save_checkpoint(model=self.mask2image, optimizer=self.generator_opt, epoch=self.epoch,
                            filename=self.checkpoint_gimage)
            save_checkpoint(model=self.dis_image, optimizer=self.discriminator_opt, epoch=self.epoch,
                            filename=self.checkpoint_dimage)
            save_checkpoint(model=self.dis_mask, optimizer=self.discriminator_opt, epoch=self.epoch,
                            filename=self.checkpoint_dmask)

    def load_pretrain(self,args):
        if args.load_pretrain:
            print('=> loading pretrain models')
            self.epoch = load_checkpoint(
                self.pretrain_gmask, self.image2mask, self.generator_opt, args.lr,
            )
            self.epoch = load_checkpoint(
                self.pretrain_gimage, self.mask2image, self.generator_opt, args.lr,
            )

    # pretrain does not follow the cyclegan structure, instead it uses a simple L1 loss with two encoder-decoder 
    # to make the comparison
    def pretrain(self,args):
        loop = tqdm(self.train_loader, leave=True)
        real_idx = 0
        for idx, (image, mask) in enumerate(loop):
            image = image.to(device=args.device, dtype=torch.float)
            mask = mask.to(device=args.device, dtype=torch.float)

            with torch.cuda.amp.autocast():
                fake_mask = self.image2mask(image)
                fake_image = self.mask2image(mask)
                image_loss = self.mse(image, fake_image)
                mask_loss = self.mse(mask, fake_mask)

                loss = (image_loss + mask_loss)
                self.generator_loss += loss

            self.generator_opt.zero_grad()
            self.g_scaler.scale(loss).backward()
            self.g_scaler.step(self.generator_opt)
            self.g_scaler.update()

            loop.set_postfix(Loss=self.generator_loss / (idx + 1))

        self.generator_loss = 0

        # validation stage

        for idx, (image,mask) in enumerate(self.val_loader):
            image = image.to(device=args.device, dtype=torch.float)
            mask = mask.to(device=args.device, dtype=torch.float)

            # Train Discriminators H and Z
            with torch.no_grad():
                fake_image = self.mask2image(mask)
                fake_mask = self.image2mask(image)

            saved_image_dir = 'saved_images/' + args.task + '/'

            if idx == 0:
                save_image(fake_mask*0.5+0.5, f"{saved_image_dir}fmask_{self.epoch}.png")
                save_image(fake_image * 0.5 + 0.5, f"{saved_image_dir}freal_{self.epoch}.png")
                save_image(mask*0.5+0.5, f"{saved_image_dir}mask_{self.epoch}.png")
                save_image(image*0.5 + 0.5, f"{saved_image_dir}real_{self.epoch}.png")


    def train(self,args):
        loop = tqdm(self.train_loader,leave=True)
        real_idx = 0
        for idx, (image,mask) in enumerate(loop):

            skip = False
            # enabling skipping to increase training speed
            if args.fast_training:
                skip = random.choice([True, True, True, False])
            if skip:
                continue

            real_idx += 1
            image = image.to(device=args.device, dtype=torch.float)
            mask = mask.to(device=args.device, dtype=torch.float)

            # Train Discriminators H and Z
            with torch.cuda.amp.autocast():
                fake_mask = self.image2mask(image)
                D_M_real = self.dis_mask(mask)
                D_M_fake = self.dis_mask(fake_mask.detach())
                self.generator_loss += D_M_real.mean().item()
                self.discriminator_loss += D_M_fake.mean().item()
                D_M_real_loss = self.mse(D_M_real, torch.ones_like(D_M_real))
                D_M_fake_loss = self.mse(D_M_fake, torch.zeros_like(D_M_fake))
                D_M_loss = D_M_real_loss + D_M_fake_loss

                fake_image = self.mask2image(mask)
                D_I_real = self.dis_image(image)
                D_I_fake = self.dis_image(fake_image.detach())
                D_I_real_loss = self.mse(D_I_real, torch.ones_like(D_I_real))
                D_I_fake_loss = self.mse(D_I_fake, torch.zeros_like(D_I_fake))
                D_I_loss = D_I_real_loss + D_I_fake_loss

                # put it togethor
                D_loss = (D_M_loss + D_I_loss) / 2

            self.discriminator_opt.zero_grad()
            self.d_scaler.scale(D_loss).backward()
            self.d_scaler.step(self.discriminator_opt)
            self.d_scaler.update()

            self.writer_disloss.append(D_loss.detach().to('cpu').numpy())


            # Train Generators H and Z
            with torch.cuda.amp.autocast():

                # adversarial loss for both generators
                D_M_fake = self.dis_mask(fake_mask)
                D_I_fake = self.dis_image(fake_image)
                loss_G_H = self.mse(D_M_fake, torch.ones_like(D_M_fake))
                loss_G_Z = self.mse(D_I_fake, torch.ones_like(D_I_fake))

                # cycle loss
                cycle_image = self.mask2image(fake_mask)
                cycle_mask = self.image2mask(fake_image)

                cycle_image_loss = self.L1(image, cycle_image)
                #cycle_mask_loss = nn.BCEWithLogitsLoss()
                cycle_mask_loss = self.L1(mask, cycle_mask)

                # geometric loss


                rot_image = torch.rot90(image,-1,[2,3])
                rot_mask = torch.rot90(mask,-1,[2,3])
                rot_fake_mask = self.image2mask(rot_image)
                rot_fake_image = self.mask2image(rot_mask)
                back_fake_image = torch.rot90(rot_fake_image,-3,[2,3])
                back_fake_mask = torch.rot90(rot_fake_mask,-3,[2,3])

                geo_image_loss = self.L1(fake_image,back_fake_image)
                geo_mask_loss = self.L1(fake_mask,back_fake_mask)
                geo_loss = geo_image_loss + geo_mask_loss



                # add all togethor
                G_loss = (
                        loss_G_Z * args.adversarial_weight
                        + loss_G_H * args.adversarial_weight
                        + cycle_image_loss * args.cycle_consistency_weight
                        + cycle_mask_loss * args.cycle_consistency_weight
                        + geo_loss * args.geometry_weight
                )

            self.generator_opt.zero_grad()
            self.g_scaler.scale(G_loss).backward()
            self.g_scaler.step(self.generator_opt)
            self.g_scaler.update()

            writer_cycle_loss = cycle_image_loss + cycle_mask_loss
            writer_gen_loss = loss_G_H + loss_G_H

            self.writer_genloss.append(writer_gen_loss.detach().to('cpu').numpy())
            self.writer_cycleloss.append(writer_cycle_loss.detach().to('cpu').numpy())
            self.writer_shapeloss.append(geo_loss.detach().to('cpu').numpy())

            loop.set_postfix(I_real=self.generator_loss / (real_idx), I_fake=self.discriminator_loss/ (real_idx))

        self.generator_loss = 0
        self.discriminator_loss = 0
        loss_dir = 'loss/' + args.task + '/'
        np.save(loss_dir + 'dis_loss.npy',self.writer_disloss)
        np.save(loss_dir + 'gen_loss.npy', self.writer_genloss)
        np.save(loss_dir + 'cycle_loss.npy', self.writer_cycleloss)
        np.save(loss_dir + 'geo_loss.npy', self.writer_shapeloss)

    def validate(self,args):
        for idx, (image,mask) in enumerate(self.val_loader):
            image = image.to(device=args.device, dtype=torch.float)
            mask = mask.to(device=args.device, dtype=torch.float)

            # Train Discriminators H and Z
            with torch.no_grad():
                fake_image = self.mask2image(mask)
                fake_mask = self.image2mask(image)

            saved_image_dir = 'saved_images/' + args.task + '/'
            # if idx == 0:
            save_image(fake_mask[0, :, :, :]*0.5+0.5, f"{saved_image_dir}fmask_{idx}_{self.epoch}.png")
            save_image(fake_image[0, :, :, :] * 0.5 + 0.5, f"{saved_image_dir}freal_{idx}_{self.epoch}.png")
            save_image(mask[0, :, :, :]*0.5+0.5, f"{saved_image_dir}mask_{idx}_{self.epoch}.png")
            save_image(image[0, :, :, :]*0.5 + 0.5, f"{saved_image_dir}real_{idx}_{self.epoch}.png")
            # break

    def test(self,args):
        for idx, (image,mask) in enumerate(self.test_loader):
            mask = mask.to(device=args.device, dtype=torch.float)
            image = image.to(device=args.device, dtype=torch.float)
            # Train Discriminators H and Z
            with torch.no_grad():
                fake_image = self.mask2image(mask)
                fake_mask = self.image2mask(image)

            image_dir = args.test_save_dir
            os.makedirs(image_dir, exist_ok=True)

            save_image(fake_mask * 0.5 + 0.5, image_dir + f'/fmask{idx}.png')
            save_image(fake_image * 0.5 + 0.5, image_dir + f'/freal{idx}.png')
            save_image(mask * 0.5 + 0.5, image_dir + f'/mask{idx}.png')
            save_image(image * 0.5 + 0.5, image_dir + f'/real{idx}.png')

            print(f'finish generating {idx} images')

    def set_learning_rate(self,lr):

        for g in self.generator_opt.param_groups:
            g['lr'] = lr
        for d in self.generator_opt.param_groups:
            d['lr'] = lr