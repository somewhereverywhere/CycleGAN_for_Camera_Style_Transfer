def train_cyclegan(data_root='processed', num_epochs=200, save_path='checkpoints', resume_from='checkpoints/cyclegan_checkpoint_epoch_250.pth'):
    import os, gc, torch, random, numpy as np
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from torch.cuda.amp import autocast, GradScaler
    from generator import Generator
    from discriminator import Discriminator
    from dataset import ImageDataset

    # Set up reproducibility and device
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_path, exist_ok=True)

    # Models
    G_AB = Generator(n_residual_blocks=6).to(device)
    G_BA = Generator(n_residual_blocks=6).to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    lr = 0.0002
    optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # Schedulers
    lambda_rule = lambda epoch: 1.0 - max(0, epoch - 100) / float(100 + 1)
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
    scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)

    # Keep your mixed precision scaler
    scaler = GradScaler()

    # Load checkpoint
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        G_AB.load_state_dict(checkpoint['G_AB'])
        G_BA.load_state_dict(checkpoint['G_BA'])
        D_A.load_state_dict(checkpoint['D_A'])
        D_B.load_state_dict(checkpoint['D_B'])
        try:
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
            optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
        except ValueError:
            print("Skipping optimizer load due to parameter mismatch.")
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Data loader (512x512)
    dataset = ImageDataset(data_root)  # update your ImageDataset to support this
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(start_epoch, num_epochs):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        epoch_loss_G, epoch_loss_D = 0.0, 0.0

        for i, batch in enumerate(dataloader):
            try:
                real_A = batch['A'].to(device)
                real_B = batch['B'].to(device)
                optimizer_G.zero_grad(set_to_none=True)

                with autocast():
                    # Identity
                    loss_identity = criterion_identity(G_AB(real_B), real_B) * 5.0
                    loss_identity += criterion_identity(G_BA(real_A), real_A) * 5.0

                    # GAN
                    fake_B = G_AB(real_A)
                    pred_fake_B = D_B(fake_B)
                    valid = torch.ones_like(pred_fake_B)
                    loss_GAN_AB = criterion_GAN(pred_fake_B, valid.to(device))

                    fake_A = G_BA(real_B)
                    pred_fake_A = D_A(fake_A)
                    loss_GAN_BA = criterion_GAN(pred_fake_A, valid.to(device))

                    # Cycle
                    loss_cycle = criterion_cycle(G_BA(fake_B), real_A) * 10.0
                    loss_cycle += criterion_cycle(G_AB(fake_A), real_B) * 10.0

                    # Total generator loss
                    loss_G = loss_identity + loss_GAN_AB + loss_GAN_BA + loss_cycle

                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)

                # Discriminator A
                optimizer_D_A.zero_grad(set_to_none=True)
                with autocast():
                    pred_real = D_A(real_A)
                    loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))
                    pred_fake = D_A(fake_A.detach())
                    loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))
                    loss_D_A = (loss_real + loss_fake) * 0.5
                scaler.scale(loss_D_A).backward()
                scaler.step(optimizer_D_A)

                # Discriminator B
                optimizer_D_B.zero_grad(set_to_none=True)
                with autocast():
                    pred_real = D_B(real_B)
                    loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))
                    pred_fake = D_B(fake_B.detach())
                    loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))
                    loss_D_B = (loss_real + loss_fake) * 0.5
                scaler.scale(loss_D_B).backward()
                scaler.step(optimizer_D_B)

                scaler.update()

                epoch_loss_G += loss_G.item()
                epoch_loss_D += (loss_D_A.item() + loss_D_B.item())

                if i % 25 == 0:
                    print(f"[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] Loss_G: {loss_G.item():.4f}, Loss_D: {(loss_D_A.item() + loss_D_B.item()):.4f}")

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"🧠 Out of memory at batch {i}, clearing cache...")
                    torch.cuda.empty_cache(); gc.collect()
                else:
                    raise e

        # LR scheduler step
        scheduler_G.step(); scheduler_D_A.step(); scheduler_D_B.step()

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            torch.save({
                'G_AB': G_AB.state_dict(),
                'G_BA': G_BA.state_dict(),
                'D_A': D_A.state_dict(),
                'D_B': D_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict(),
                'epoch': epoch
            }, f"{save_path}/cyclegan_checkpoint_epoch_{epoch+1}.pth")
            print(f"💾 Saved checkpoint at epoch {epoch+1}")

        # Save sample outputs every 10 epochs
        if (epoch + 1) % 10 == 0:
            sample_dir = os.path.join(save_path, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            with torch.no_grad():
                G_AB.eval()
                sample_fake_B = G_AB(real_A)
                save_image(torch.cat([real_A, sample_fake_B], 0), f"{sample_dir}/sample_epoch{epoch+1}.png", normalize=True)
                G_AB.train()

        torch.cuda.empty_cache(); gc.collect()
        print(f"✅ Epoch {epoch+1} completed. Avg Loss G: {epoch_loss_G/len(dataloader):.4f}, Avg Loss D: {epoch_loss_D/len(dataloader):.4f}")
train_cyclegan(
    data_root='processed',
    num_epochs=300,
    save_path='checkpoints',
    resume_from='checkpoints/cyclegan_checkpoint_epoch_250.pth'
)