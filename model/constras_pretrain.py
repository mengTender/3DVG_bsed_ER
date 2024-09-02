import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import random
from models.vit import load_vit_base
import os


class FaceExpressionViT(nn.Module):
    def __init__(self, num_classes):
        super(FaceExpressionViT, self).__init__()
        self.vit = load_vit_base(7)
        self.vit.heads = nn.Linear(self.vit.heads.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


class CosineContrastiveLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(CosineContrastiveLoss, self).__init__()
        self.beta = beta

    def forward(self, output1, output2, label):
        cosine_similarity = F.cosine_similarity(output1, output2)
        loss_cosine = torch.mean(
            -label * self.beta * cosine_similarity +
            (1 - label) * self.beta * torch.clamp(cosine_similarity, min=0)
        )
        return loss_cosine


class EuclideanContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(EuclideanContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            label * euclidean_distance.pow(2) +
            (1 - label) * torch.clamp(self.margin - euclidean_distance, min=0).pow(2)
        )
        return loss_contrastive


def create_pairs(images, labels):
    pairs = []
    num_classes = torch.max(labels).item() + 1

    class_to_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(labels):
        class_to_indices[label.item()].append(images[idx])

    for idx, image in enumerate(images):
        label = labels[idx].item()

        positive_idx = random.choice(class_to_indices[label])
        pairs.append((image, positive_idx, 1))

        negative_label = random.choice([l for l in range(num_classes) if l != label])
        negative_idx = random.choice(class_to_indices[negative_label])
        pairs.append((image, negative_idx, 0))

    return pairs


def evaluate(model, val_loader, cosine_loss_fn, euclidean_loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            pairs = create_pairs(images, labels)
            for img1, img2, lbl in pairs:
                img1, img2, lbl = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device), torch.tensor(lbl).to(
                    device)
                output1, output2 = model(img1), model(img2)

                loss_cosine = cosine_loss_fn(output1, output2, lbl)
                loss_euclidean = euclidean_loss_fn(output1, output2, lbl)
                loss = loss_cosine + loss_euclidean
                val_loss += loss.item()

    return val_loss / len(val_loader)


def train(model, train_loader, val_loader, cosine_loss_fn, euclidean_loss_fn, optimizer, device, num_epochs, save_path, lambda1):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            pairs = create_pairs(images, labels)
            for img1, img2, lbl in pairs:
                img1, img2, lbl = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device), torch.tensor(lbl).to(
                    device)
                output1, output2 = model(img1), model(img2)

                loss_cosine = cosine_loss_fn(output1, output2, lbl)
                loss_euclidean = euclidean_loss_fn(output1, output2, lbl)
                loss = lambda1 * loss_cosine + (1 - lambda1) * loss_euclidean

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        val_loss = evaluate(model, val_loader, cosine_loss_fn, euclidean_loss_fn, device)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with validation loss: {best_val_loss:.4f}")

    print("Training complete.")


def main(args):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = FaceExpressionViT(num_classes=args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    cosine_loss_fn = CosineContrastiveLoss(beta=args.beta)
    euclidean_loss_fn = EuclideanContrastiveLoss(margin=args.margin)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, train_loader, val_loader, cosine_loss_fn, euclidean_loss_fn, optimizer, device, args.num_epochs,
          args.save_path, args.lambda1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a ViT for facial expression recognition using contrastive learning.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to SFEW 2.0')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of expression classes')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta value for cosine contrastive loss')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin value for euclidean contrastive loss')
    parser.add_argument('--save_path', type=str, default='./checkpoint/pretrained_vit.pth', help='Path to save the best model')
    parser.add_argument('--lambda1', type=float, default=0.5)

    args = parser.parse_args()
    main(args)
