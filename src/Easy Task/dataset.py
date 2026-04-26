"""
dataset.py
----------
YouTube downloading, audio feature extraction, lyrics pipeline,
combined dataset building, scaling/encoding, and dataset visualisation.

"""

import os
import re
import glob
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings('ignore')


# Constants

# Actual dim breakdown: 40+13+12+8+6+9+3+2+9 = 102  (was silently trimmed to 100)
AUDIO_DIM = 102   # FIXED: matches actual concatenated dims exactly
LYRICS_DIM = 0
TOTAL_DIM = AUDIO_DIM + LYRICS_DIM  # 102 (audio-only mode)

CLIP_DURATION = 30   # seconds
N_PER_GENRE_EN = 20
N_PER_GENRE_BN = 20

# Genre definitions

ENGLISH_GENRES = {
    'Rock': 'ytsearch20:classic rock songs hits 30 seconds',
    'Pop': 'ytsearch20:english pop music hits songs',
    'Jazz': 'ytsearch20:jazz music instrumental classic',
    'Classical': 'ytsearch20:classical music orchestral symphony',
    'HipHop': 'ytsearch20:hip hop rap music hits songs',
    'Blues':'ytsearch20:blues guitar music classic songs',
    'Country':'ytsearch20:country music songs classic hits',
}

BANGLA_GENRES = {
    'Baul':'ytsearch20:bangla baul song folk mystic',
    'Folk':'ytsearch20:bangla folk gaan traditional',
    'Rabindra': 'ytsearch20:rabindra sangeet tagore bengali song',
    'ModernPop': 'ytsearch20:modern bangla pop song hits',
    'Classical': 'ytsearch20:bangla classical music ustad',
}

# Lyrics seed vocabulary (used only when LYRICS_DIM > 0)

GENRE_LYRICS_SEEDS = {
    'pop':'love heart dance floor night feel emotions chorus hook beat drop',
    'rock':'rebel fight loud guitar riff freedom raw power scream noise',
    'jazz':'blues swing improvise soul night bar trumpet scat cool groove',
    'classical':'symphony movement allegro sonata orchestral harmony counterpoint theme',
    'hiphop':'flow bars rhyme verse hustle street rap beat producer sample',
    'electronic':'synth drop bass filter wave pulse rave circuit machine loop',
    'folk':'river mountain story old ballad countryside wooden fiddle simple',
    'rnb':'smooth groove rhythm soulful melody sensual harmony velvet night',
    'baul':'mon akash nodi bhalobasha fakir dotar song soul river mystic',
    'rabindrasangeet': 'prem akash alo ananda brishti shanto hriday kotha gan',
    'nazrul':'bidrohi fire jago shokal shondhya desher gaan patriot rhythm',
    'adhunik':'bhalobasha jibon smriti kotha mon hashi kanna shukh dukho',
    'folk_bangla':'nodir kule gram gaon mati shonar bangla desh maa bhalobasha',
    'film':'nacho gao khushi ananda hero heroine scene song soundtrack',
}


# Download English Songs via yt-dlp

def download_genre_yt(genre_name, search_query, out_dir, n=20):
    """Download up to n WAV clips for a single English genre."""
    genre_dir = f'{out_dir}/{genre_name}'
    os.makedirs(genre_dir, exist_ok=True)

    existing = len([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
    if existing >= n:
        print(f' {genre_name}: {existing} tracks already downloaded, skipping.')
        return

    print(f' Downloading {genre_name} (target {n})...')

    cmd = [
        'yt-dlp',
        '--quiet',
        '--no-warnings',
        '--cookies-from-browser', 'chrome',
        '--extract-audio',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--postprocessor-args', f'-t {CLIP_DURATION}',
        '--output', f'{genre_dir}/%(title)s.%(ext)s',
        '--playlist-end', str(n),
        '--no-playlist',
        search_query
    ]

    try:
        subprocess.run(cmd, timeout=300)
    except subprocess.TimeoutExpired:
        print(f' {genre_name}: Timeout reached, using partial downloads')

    # ALWAYS count what we got (even after timeout)
    downloaded = len([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
    if downloaded == 0:
        print(f' {genre_name}: No tracks downloaded')
    else:
        print(f' {genre_name}: {downloaded} tracks available (used for training)')


def download_english_songs(english_dir='/content/english',
                           n_per_genre=N_PER_GENRE_EN):
    """Download all English genre songs and print a summary."""
    os.makedirs(english_dir, exist_ok=True)
    print('Downloading English songs by genre from YouTube...')
    for genre, query in ENGLISH_GENRES.items():
        try:
            download_genre_yt(genre, query, english_dir, n=n_per_genre)
        except Exception as e:
            print(f'Skipping {genre} due to error: {e}')
            continue

    print('\nEnglish dataset summary:')
    total_english = 0
    for genre in ENGLISH_GENRES:
        gdir = f'{english_dir}/{genre}'
        n = len([f for f in os.listdir(gdir) if f.endswith('.wav')]) \
            if os.path.exists(gdir) else 0
        total_english += n
        print(f'   {genre:15s}: {n} tracks')
    print(f'   {"TOTAL":15s}: {total_english} tracks')


# Download Bangla Songs via yt-dlp

def download_bangla_genre(genre_name, search_query, out_dir, n=20):
    """Download up to n WAV clips for a single Bangla genre."""
    genre_dir = f'{out_dir}/{genre_name}'
    os.makedirs(genre_dir, exist_ok=True)

    existing = len([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
    if existing >= n:
        print(f'   {genre_name}: {existing} tracks already downloaded, skipping.')
        return

    print(f' Downloading {genre_name} ({n} tracks)...')
    cmd = [
        'yt-dlp',
        '--quiet',
        '--no-warnings',
        '--cookies-from-browser', 'chrome',
        '--extract-audio',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--postprocessor-args', f'-t {CLIP_DURATION}',  # clip to 30s
        '--output', f'{genre_dir}/%(title)s.%(ext)s',
        '--playlist-end', str(n),
        '--no-playlist',
        search_query
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    downloaded = len([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
    print(f' {genre_name}: {downloaded} tracks downloaded.')


def download_bangla_songs(bangla_dir='/content/bangla',
                          n_per_genre=N_PER_GENRE_BN):
    """Download all Bangla genre songs and print a summary."""
    os.makedirs(bangla_dir, exist_ok=True)
    print('Downloading Bangla songs by genre...')
    for genre, query in BANGLA_GENRES.items():
        download_bangla_genre(genre, query, bangla_dir, n=n_per_genre)

    print('\nBangla dataset summary:')
    total_bangla = 0
    for genre in BANGLA_GENRES:
        gdir = f'{bangla_dir}/{genre}'
        n = len([f for f in os.listdir(gdir) if f.endswith('.wav')]) \
            if os.path.exists(gdir) else 0
        total_bangla += n
        print(f'   {genre:15s}: {n} tracks')
    print(f'   {"TOTAL":15s}: {total_bangla} tracks')


# Feature Extraction with Librosa

def extract_audio_features(file_path, sr=22050, duration=30):
    """
    Extract a fixed 102-dim audio feature vector from a WAV/MP3 file.

    Breakdown: MFCCs×40 | MFCC-Δ×13 | Chroma×12 | Mel-stats×8 |
               Tonnetz×6 | Spectral×9 | Rhythm×3 | ZCR+RMS×2 | ChromaCQT×9
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration, mono=True)
        if len(y) < sr * 2:
            return None

        # 1. MFCCs — 40 dims
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1) # (40,)

        # 2. MFCC Deltas — 13 dims
        mfcc_delta = librosa.feature.delta(mfcc[:13])
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1) # (13,)

        # 3. Chroma STFT — 12 dims
        chroma = librosa.feature.chroma_stft(y=y, sr=sr) 
        chroma_mean = np.mean(chroma, axis=1) # (12,)

        # 4. Mel-spectrogram stats — 8 dims (mean+std × 4 bands)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        bands = np.array_split(mel_db, 4, axis=0)
        mel_stats = np.array([
            stat for band in bands
            for stat in (np.mean(band), np.std(band))
        ]) # (8,)

        # 5. Tonnetz — 6 dims
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1) # (6,)

        # 6. Spectral features — 9 dims (FIXED: pinned output shape)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))   # 1
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))  # 1
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)) # 1

        # FIXED: take mean over BOTH axes → single scalar per contrast band
        # avoids variable output shape across different audio files/sr combos
        spec_contrast_raw = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4)
        spec_contrast = np.mean(spec_contrast_raw, axis=1)[:4]   # force exactly 4 values

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        spec_flux = np.mean(onset_env) # 1
        spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y)) # 1

        spectral_feats = np.array([
            spec_centroid, spec_bandwidth, spec_rolloff,
            *spec_contrast, # exactly 4
            spec_flux, spec_flatness # 2
        ]) # total = 9, always  # (9,)

        # 7. Rhythm — 3 dims
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_strength = np.mean(onset_env[beats]) if len(beats) > 0 else 0.0
        onset_rate = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr)
        rhythm_feats = np.array([float(tempo), beat_strength, onset_rate])  # (3,)

        # 8. ZCR + RMS — 2 dims
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        rms = np.mean(librosa.feature.rms(y=y))
        energy_feats = np.array([zcr, rms]) # (2,)

        # 9. Chroma CQT — 9 dims
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cqt_m = np.mean(chroma_cqt, axis=1)[:9] # (9,)

        feature_vec = np.concatenate([
            mfcc_mean, # 40
            mfcc_delta_mean, # 13
            chroma_mean, # 12
            mel_stats, #  8
            tonnetz_mean, #  6
            spectral_feats, #  9  
            rhythm_feats, #  3
            energy_feats, #  2
            chroma_cqt_m, #  9
        ]) # total = 102 — matches AUDIO_DIM exactly

        # Safety pad/trim (should never trigger now, but kept as guard)
        if len(feature_vec) != AUDIO_DIM:
            print(f'Unexpected dim {len(feature_vec)} for {file_path}, adjusting.')
            if len(feature_vec) > AUDIO_DIM:
                feature_vec = feature_vec[:AUDIO_DIM]
            else:
                feature_vec = np.pad(feature_vec, (0, AUDIO_DIM - len(feature_vec)))

        return feature_vec

    except Exception as e:
        return None


# Lyrics Pipeline (with circular-leakage guard)

def load_lyrics_for_track(audio_path):
    """
    Try to load a .txt lyrics file co-located with the audio file.
    Returns None if not found.
    """
    base = os.path.splitext(audio_path)[0]
    genre_dir = os.path.dirname(audio_path)
    if os.path.exists(base + '.txt'):
        with open(base + '.txt', 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    fname_txt = os.path.splitext(os.path.basename(audio_path))[0] + '.txt'
    lyrics_path = os.path.join(genre_dir, 'lyrics', fname_txt)
    if os.path.exists(lyrics_path):
        with open(lyrics_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    return None


def build_lyrics_corpus(records):
    """
    Build a text corpus aligned with records list.
    Uses a genre-neutral fallback to avoid circular metric inflation.
    """
    corpus, has_real_lyrics = [], []
    for rec in records:
        lyrics = load_lyrics_for_track(rec['file'])
        if lyrics and len(lyrics.strip()) > 10:
            corpus.append(lyrics)
            has_real_lyrics.append(True)
        else:
            # FIXED: use a genre-neutral fallback so lyric embeddings don't
            # encode genre identity directly (avoids circular metric inflation)
            corpus.append('music song melody rhythm audio sound')
            has_real_lyrics.append(False)

    n_real = sum(has_real_lyrics)
    print(f' Lyrics coverage: {n_real}/{len(records)} tracks have real lyrics '
          f'({100*n_real/max(len(records),1):.1f}%)')
    if n_real == 0:
        print(' No real lyrics found — lyric dims will carry no signal.')
        print(' Consider audio-only mode or providing .txt lyric files.')
    return corpus, has_real_lyrics


def fit_lyrics_embedder(corpus, n_components=LYRICS_DIM, max_features=2000):
    """
    Fit TF-IDF + TruncatedSVD (LSA) on the corpus.
    Returns (vectorizer, svd, lyric_embeddings).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2),
        sublinear_tf=True, min_df=1,
        strip_accents='unicode', analyzer='word',
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    n_comp = min(n_components, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    lyric_embeddings = svd.fit_transform(tfidf_matrix)
    if lyric_embeddings.shape[1] < n_components:
        pad = np.zeros((lyric_embeddings.shape[0], n_components - lyric_embeddings.shape[1]))
        lyric_embeddings = np.hstack([lyric_embeddings, pad])
    print(f'   TF-IDF vocab: {len(vectorizer.vocabulary_)} | '
          f'LSA explained var: {svd.explained_variance_ratio_.sum():.3f}')
    return vectorizer, svd, lyric_embeddings


# Combined Extraction Loop + Scaling

def build_dataset(english_dir='/content/english',
                  bangla_dir='/content/bangla',
                  lyrics_dim=LYRICS_DIM):
    """
    Scan downloaded audio, extract features, optionally build lyric embeddings,
    scale and label-encode.

    Returns
    -------
    df : pd.DataFrame  — metadata + cluster columns later
    X_scaled : np.ndarray — StandardScaler-normalised feature matrix
    y_genre : np.ndarray — integer genre labels
    y_lang : np.ndarray — integer language labels
    le_genre : LabelEncoder
    le_lang : LabelEncoder
    scaler : StandardScaler (fitted)
    """
    records = []

    print('\nExtracting audio features — English...')
    for genre in tqdm(list(ENGLISH_GENRES.keys()), desc='English genres'):
        genre_path = f'{english_dir}/{genre}'
        if not os.path.exists(genre_path):
            continue
        files = [f for f in os.listdir(genre_path)
                 if f.endswith('.wav') or f.endswith('.mp3')][:20]
        for fname in files:
            fpath = os.path.join(genre_path, fname)
            feat  = extract_audio_features(fpath)
            if feat is not None:
                records.append({'file': fpath, 'genre': genre,
                                'language': 'English', 'audio_features': feat})

    n_english = len(records)
    print(f'English tracks: {n_english}')

    print('\nExtracting audio features — Bangla...')
    for genre in tqdm(list(BANGLA_GENRES.keys()), desc='Bangla genres'):
        genre_dir_path = f'{bangla_dir}/{genre}'
        if not os.path.exists(genre_dir_path):
            continue
        files = [f for f in os.listdir(genre_dir_path)
                 if f.endswith('.wav') or f.endswith('.mp3')]
        for fname in files:
            fpath = os.path.join(genre_dir_path, fname)
            feat  = extract_audio_features(fpath)
            if feat is not None:
                records.append({'file': fpath, 'genre': genre,
                                'language': 'Bangla', 'audio_features': feat})

    n_bangla = len(records) - n_english
    print(f'Bangla tracks: {n_bangla}  |  Total: {len(records)}')

    df = pd.DataFrame(records)

    # Audio-only mode (LYRICS_DIM = 0)
    X_raw = np.vstack(df['audio_features'].values) # (N, 102)

    if lyrics_dim > 0:
        print('\nBuilding lyrics embeddings...')
        corpus, has_real = build_lyrics_corpus(records)
        vectorizer, svd_model, lyric_embs = fit_lyrics_embedder(
            corpus, n_components=lyrics_dim)
        df['has_real_lyrics'] = has_real
        X_raw = np.hstack([X_raw, lyric_embs])
    else:
        print('\nAudio-only mode: lyrics pipeline skipped (LYRICS_DIM=0).')
        df['has_real_lyrics'] = False

    total_dim = AUDIO_DIM + lyrics_dim
    assert X_raw.shape[1] == total_dim, \
        f'Expected {total_dim} dims, got {X_raw.shape[1]}'

    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    le_genre = LabelEncoder()
    y_genre = le_genre.fit_transform(df['genre'])
    le_lang = LabelEncoder()
    y_lang = le_lang.fit_transform(df['language'])

    print(f'\nFeature matrix: {X_scaled.shape}')
    print(f' Audio: {AUDIO_DIM} | Lyrics: {lyrics_dim} | Total: {total_dim}')
    print(f' English: {(df["language"]=="English").sum()} | '
          f'Bangla: {(df["language"]=="Bangla").sum()}')
    print(f' Genres: {list(le_genre.classes_)}')

    return df, X_scaled, y_genre, y_lang, le_genre, le_lang, scaler


# Dataset Visualization

def plot_dataset_distribution(df, out_dir='/content'):
    """Plot and save language + genre distribution bar charts."""
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Language distribution
    lang_counts = df['language'].value_counts()
    colors_lang = ['#4e8ef7', '#f7914e']
    axes[0].bar(lang_counts.index, lang_counts.values,
                color=colors_lang, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Language Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Track Count')
    for i, v in enumerate(lang_counts.values):
        axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

    # Genre distribution
    genre_counts = df.groupby(['genre', 'language']).size().unstack(fill_value=0)
    genre_counts.plot(kind='bar', ax=axes[1],
                      color=['#f7914e', '#4e8ef7'], edgecolor='white')
    axes[1].set_title('Genre Distribution by Language', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Track Count')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=35)
    axes[1].legend(title='Language')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/dataset_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: dataset_distribution.png')
