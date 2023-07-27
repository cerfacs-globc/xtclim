from tqdm import tqdm
import torch

def final_loss(bce_loss, mu, logvar, beta=0.1):
    """
    Adds up reconstruction loss (BCELoss) and Kullback-Leibler divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    :param beta: weight over the KL-Divergence
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta*KLD

def train(model, dataloader, dataset, device, optimizer, criterion, beta):
    # trains the model over shuffled data set
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        # total loss = reconstruction loss + KL divergence
        loss = final_loss(bce_loss, mu, logvar, beta)
        loss.backward() # backpropagate loss to learn from the mistakes
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter # average loss over the batches
    return train_loss

def validate(model, dataloader, dataset, device, criterion, beta):
    # evaluate test data: no backpropagation
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar, beta)
            running_loss += loss.item()
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images

def evaluate(model, dataloader, dataset, device, criterion, beta):
    model.eval()
    running_loss = 0.0
    losses = []
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            # evaluate anomalies with reconstruction error only
            loss = criterion(reconstruction, data)
            running_loss += loss.item()
            losses.append(loss.item()) # keep track of all losses
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images, losses