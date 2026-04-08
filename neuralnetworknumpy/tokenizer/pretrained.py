import urllib.request
import os

TOKENIZER_URL = "https://github.com/Sendy45/NeuralNetworkFromScratch/releases/download/v3.0.0/tokenizer.json"

def download_tokenizer(save_path="tokenizer.json"):
    """Download the pretrained BPE tokenizer."""
    if os.path.exists(save_path):
        print(f"Tokenizer already exists at {save_path}, skipping download.")
        return save_path

    print(f"Downloading tokenizer...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent    = min(downloaded / total_size * 100, 100)
        print(f"\r  {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(TOKENIZER_URL, save_path, reporthook=_progress)
    print(f"\nSaved to {save_path}")
    return save_path
