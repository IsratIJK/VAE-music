"""
Global hyperparameters and path configuration for the VAE-music project.

All paths are relative to the project root.
"""

import os
from pathlib import Path

# -- Project root (two levels above this file: config/ -> project root) ---------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -- Data / output directories --------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Sub-directories used at runtime (created automatically by each script)
FMA_DIR = DATA_DIR / "fma" / "fma_metadata"
LMD_DIR = DATA_DIR / "lmd"
GTZAN_CSV = DATA_DIR / "gtzan" / "features_30_sec.csv"
BANGLA_DIR = DATA_DIR / "audio" / "bangla"

# -- VAE / Training hyperparameters ---------------------------------------------
DEVICE_STR = "auto"   # "auto" -> cuda if available, else cpu
MAX_SAMPLES = 10_000  # cap per dataset to limit memory
LATENT_DIM = 32  # dimensionality of VAE latent space
HIDDEN_DIMS = (256, 128, 64)  # MLP hidden layer sizes
CONV_CHANNELS = (32, 64, 128)  # Conv1D encoder channel sizes
EPOCHS = 100    # training epochs
LR = 1e-3   # AdamW learning rate
BATCH_SIZE = 256   # mini-batch size
BETA_DEFAULT = 1.0  # β for standard VAE ELBO
BETA_VAE_B = 4.0  # β for Beta-VAE (encourages disentanglement)
LYRIC_DIM = 32   # TF-IDF -> PCA dimension for lyrics embedding
KMEANS_NINIT = 20  # KMeans n_init (higher = more stable)

# -- Bangla dataset config ------------------------------------------------------
BANGLA_QUERIES = {
    "Rabindra_Sangeet": "rabindra sangeet full song playlist",
    "Baul": "baul song bangla authentic",
    "Folk": "bangla folk song lok geeti traditional",
    "Modern_Pop": "bangla modern song adhunik gaan",
    "Classical": "bangla classical music raga",
}
N_BANGLA_PER_GENRE = 30  # target tracks per genre (~150 total)

# -- Genre keyword vocabulary for TF-IDF lyric embedding -------------------------
GENRE_VOCAB: dict[str, str] = {
    "Hip-Hop": "beats rhythm rap flow street hustle trap bars verse hook",
    "Rock": "guitar riff solo distortion drums loud electric power chord",
    "Folk": "acoustic story ballad rural nature simple honest heartfelt",
    "Folk_English": "acoustic story ballad rural nature simple honest heartfelt",
    "Experimental": "noise abstract texture ambient drone silence space weird",
    "International": "world rhythm dance culture tradition heritage melody exotic",
    "Electronic": "synth bass drop rave pulse frequency modulate loop sample",
    "Pop": "catchy love chorus hook radio melody sweet dance bright",
    "Classical": "orchestra symphony piano forte harmony movement sonata",
    "Classical_Western": "orchestra symphony piano forte harmony movement sonata",
    "jazz": "improvise swing blue note chord modal bebop cool groove",
    "country": "truck road heartbreak southern twang fiddle boots honky",
    "metal": "heavy scream distortion dark blast beat growl power",
    "reggae": "island skank offbeat roots dub conscious unity love peace",
    "blues": "soul pain cry lament twelve bar slide guitar wail moan",
    "disco": "dance floor groove funky night club shiny mirror ball",
    "other": "music note rhythm harmony melody beat sound instrument",
    "Rabindra_Sangeet": "akash batas nodi alo prem swapna jiban surer dhara rabi",
    "Baul": "moner manush deha mon ektar jibon fakir sufi sahaj path",
    "Folk_Lok_Geeti": "mati gram nadi borshar palligeeti lila khela bhawaiya",
    "Modern_Pop": "shundor prem birah mon hriday asha smriti chaowa pawa",
    "Film_Bangla": "cinema nayak nayika gan premer katha hero heroin naach",
}

# -- Visualization color / marker settings -----------------------------------
MODEL_COLORS: dict[str, str] = {
    "MLP-VAE": "#1565C0",
    "Beta-VAE": "#6A1B9A",
    "CVAE": "#00838F",
    "Conv-VAE": "#2E7D32",
    "Autoencoder": "#E65100",
    "Multimodal": "#AD1457",
    "PCA": "#B71C1C",
    "Spectral": "#546E7A",
}
LANG_COLORS: dict[str, str] = {"English": "#1565C0", "Bangla": "#FF5722"}
LANG_MARKERS: dict[str, str] = {"English": "o", "Bangla": "^"}

# -- Advanced extension hyperparameters ----------------------------------------
# Extension 2: β-sweep values for sensitivity analysis
BETA_VALUES: list[float] = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
# Number of epochs for β-sweep (reduced for speed)
SWEEP_EPOCHS: int = 60

# Extension 4: SLERP interpolation steps
N_INTERP: int = 12

# Extension A: Contrastive VAE
LAMBDA_VALUES: list[float] = [0.1, 0.5, 1.0]   # InfoNCE weight sweep
CONTRASTIVE_TEMPERATURE: float = 0.07   # InfoNCE temperature τ

# Extension B: DANN-VAE
DANN_COMMON_DIM: int = 32  # PCA-aligned feature dimension across domains
DANN_DOMAIN_WEIGHT: float = 0.5  # λ_d weight on domain adversarial loss

# -- Reproducibility seeds ------------------------------------------------------
NUMPY_SEED = 42
TORCH_SEED = 42

# -- Dataset download URLs -------------------------------------------------------
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
LMD_URL = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"
GTZAN_URLS = [
    "https://raw.githubusercontent.com/Manishankar9977/Music-genre-classification/main/features_30_sec.csv",
    "https://raw.githubusercontent.com/Coder-Vishali/Music_Genre_Classification/main/features_30_sec.csv",
    "https://raw.githubusercontent.com/nikitaa30/Music-Genre-Classification/master/features_30_sec.csv",
]
