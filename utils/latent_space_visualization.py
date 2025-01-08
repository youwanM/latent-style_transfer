import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# Path to the folder containing latent vector files
latent_folder = "../vae-samples/"

# Collect all .pth files in the folder
latent_files = [f for f in os.listdir(latent_folder) if f.startswith("latent-") and f.endswith(".pth")]

# Initialize a list to store all mu vectors and corresponding file IDs
all_mu = []
file_ids = []

# Load each latent vector file and extract mu
for latent_file in latent_files:
    file_path = os.path.join(latent_folder, latent_file)
    latent_data = torch.load(file_path, weights_only=False)
    mu = latent_data["mu"]  # Extract the mu tensor
    all_mu.append(mu)
    file_ids.append(latent_file)  # Save the file ID

# Concatenate all mu tensors into a single tensor
all_mu = torch.cat(all_mu, dim=0)  # Shape: (20, 4, 34, 33, 20)

# Reshape the tensor to 2D for PCA and t-SNE
all_mu_flat = all_mu.view(all_mu.shape[0], -1)  # Flatten to (20, 4*34*33*20)

# Convert to NumPy array for PCA and t-SNE
mu_np = all_mu_flat.cpu().detach().numpy()

# Perform PCA
pca = PCA(n_components=2)
mu_pca = pca.fit_transform(mu_np)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(mu_np) - 1))
mu_tsne = tsne.fit_transform(mu_np)

# Plot the PCA results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(mu_pca[:, 0], mu_pca[:, 1], alpha=0.7, cmap="viridis")
for i, file_id in enumerate(file_ids):
    plt.text(mu_pca[i, 0], mu_pca[i, 1], file_id, fontsize=7, alpha=0.3)
plt.title("PCA of Latent Space (Mu)")
plt.xlabel("PCA Dim 1")
plt.ylabel("PCA Dim 2")

# Plot the t-SNE results
plt.subplot(1, 2, 2)
plt.scatter(mu_tsne[:, 0], mu_tsne[:, 1], alpha=0.7, cmap="viridis")
for i, file_id in enumerate(file_ids):
    plt.text(mu_tsne[i, 0], mu_tsne[i, 1], file_id, fontsize=7, alpha=0.3)
plt.title("t-SNE of Latent Space (Mu)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")

# Show the plots
plt.tight_layout()
plt.show()
