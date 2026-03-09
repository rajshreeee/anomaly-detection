import os
from pathlib import Path

# --- DIRECTORY LOGIC ---
BASE_DIR = Path(__file__).parent.parent.absolute()

# --- COMMON SETTINGS ---
CATEGORY = 'carpet'
DATASET_PATH = os.path.join(BASE_DIR, "data/", "carpet")
RESULTS_PATH = os.path.join(BASE_DIR, "results/")
TEST_DATASET_PATH = os.path.join(DATASET_PATH, "test")
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, "train")
TRAIN_CLS_DATASET_PATH = os.path.join(TRAIN_DATASET_PATH, "good")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "ground_truth")
SPLIT_PATH   = f'{RESULTS_PATH}/data/split_indices.json'
MODEL_PATH = os.path.join(BASE_DIR, "models")
SEED = 42

# --- PATCHCORE (PC_) ---
PC_IMAGE_SIZE = 224
PC_BATCH_SIZE = 8
PC_LAYERS = ['layer2', 'layer3']
PC_PRETRAIN_DIM = 1024
PC_TARGET_DIM = 1024
PC_PATCH_SIZE = 3
PC_CORESET_RATIO = 0.10
PC_CORESET_PROJ_DIM = 128
PC_N_NEAREST = 1
PC_OUTPUT_DIR = os.path.join(RESULTS_PATH, "patchcore")
PC_MEMORY_BANK_FAISS = f'{PC_OUTPUT_DIR}/memory_bank.faiss'
PC_MEMORY_BANK_NPY = f'{PC_OUTPUT_DIR}/memory_bank.npy'

# --- EFFICIENT_AD (EA_) ---
EAD_APPROACH = "efficient_ad"
EAD_IMAGE_SIZE   = 256
EAD_OUT_CHANNELS = 384
EAD_TRAIN_STEPS  = 10000
EAD_BATCH_SIZE   = 1
EAD_OUTPUT_DIR = os.path.join(RESULTS_PATH, EAD_APPROACH)
EAD_MODEL_PATH = os.path.join(MODEL_PATH, EAD_APPROACH)

EAD_TEACHER_PATH = f'{EAD_MODEL_PATH}/teacher_small.pth'
EAD_STUDENT_PATH = f'{EAD_MODEL_PATH}/student_final.pth'
EAD_AUTOENCODER_PATH = f'{EAD_MODEL_PATH}/autoencoder_final.pth'
EAD_STATS_PATH = f'{EAD_MODEL_PATH}/stats.pth'

EAD_LR           = 1e-4
EAD_WEIGHT_DECAY = 1e-5
EAD_VAL_RATIO    = 0.1
EAD_SEED         = 42

os.makedirs(PC_OUTPUT_DIR, exist_ok=True)
os.makedirs(EAD_OUTPUT_DIR, exist_ok=True)