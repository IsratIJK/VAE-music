"""
dataset.py
----------
Audio feature extraction (65-dim flat + 7680-dim 2D MFCC),
real lyrics pipeline (Genius API + gaanesuno.com),
multi-modal feature builder, dataset downloading (GTZAN/Kaggle/yt-dlp),
and full dataset loading for all three datasets.
"""

import os
import re
import glob
import time
import tarfile
import subprocess
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import lyricsgenius
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from vae import (AUDIO_FEAT_DIM, MFCC_2D_DIM, N_MFCC, N_MFCC_ROWS,
                 TIME_FRAMES, LYRIC_DIM, SEED)

warnings.filterwarnings('ignore')

# Dataset directories
GTZAN_DIR      = '/content/gtzan'
BANGLAGITI_DIR = '/content/banglagiti'
BMGCD_DIR      = '/content/bmgcd'
BANGLA_YT_DIR  = '/content/bangla_audio'
OUTPUT_DIR     = '/content/vae_combined_outputs'
for d in [GTZAN_DIR, BANGLAGITI_DIR, BMGCD_DIR, BANGLA_YT_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Genius / Lyrics config
GENIUS_TOKEN = os.getenv('GENIUS_TOKEN', 'P_idh3O0SAtctPm4mwZZObR0jkagRyOcTYcTAO89DYHfo5auQF6o0AINS63JUx0O')
LYRICS_CACHE = '/content/lyrics_cache'
os.makedirs(LYRICS_CACHE, exist_ok=True)

# Dataset config
N_PER_GENRE             = 20
MIN_SAMPLES_FOR_METRICS = 30
N_BANGLA_PER_GENRE      = 20

BANGLA_QUERIES_YT = {
    'Rabindra_Sangeet': 'rabindra sangeet full song playlist',
    'Baul':             'baul song bangla authentic',
    'Folk':             'bangla folk song lok geeti traditional',
    'Modern_Pop':       'bangla modern song adhunik gaan',
    'Classical':        'bangla classical music raga',
}

BMGCD_QUERIES_YT = {
    'Adhunik':   'bangla adhunik gaan modern',
    'Baul':      'baul gaan bengali mystical folk',
    'Classical': 'bangla shashtriya sangeet raga',
    'Folk':      'bangla palli geeti folk rural',
    'Rabindra':  'rabindra sangeet tagore',
}

# Neutral fallback (no genre vocabulary, no label leakage)
LYRIC_FALLBACK = 'music sound audio rhythm melody beat instrument song'


# Real Lyrics Pipeline (Genius API + gaanesuno.com)

def _sanitize(text):
    return re.sub(r'[\\/*?:"<>|]', '', text).strip()[:80]

def _cache_path(genre, language, track_name):
    folder = os.path.join(LYRICS_CACHE, language, genre)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, _sanitize(track_name) + '.txt')

def _load_cached(genre, language, track_name):
    p = _cache_path(genre, language, track_name)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None

def _save_cached(lyrics, genre, language, track_name):
    with open(_cache_path(genre, language, track_name), 'w', encoding='utf-8') as f:
        f.write(lyrics)

def _is_gtzan_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    return bool(re.match(r'^[a-z_]+\.\d{5}$', base))

def _parse_title_artist(filename):
    base = re.sub(r'_', ' ', os.path.splitext(os.path.basename(filename))[0])
    if ' - ' in base:
        parts = base.split(' - ', 1)
        return parts[1].strip(), parts[0].strip()
    return base.strip(), None

# Genius API (English)
_genius_client = None

def _get_genius():
    global _genius_client
    if _genius_client is None and GENIUS_TOKEN not in ('', 'your_token_here'):
        try:
            _genius_client = lyricsgenius.Genius(GENIUS_TOKEN, timeout=10)
            _genius_client.skip_non_songs  = True
            _genius_client.excluded_terms  = ['(Remix)', '(Live)']
            _genius_client.verbose         = False
        except Exception as e:
            print('Genius init failed:', e)
    return _genius_client

def _clean_genius_lyrics(raw):
    text = re.sub(r'\[.*?\]', '', raw)
    text = re.sub(r'\d+\s*Contributor.*', '', text)
    text = re.sub(r'\d+Embed$', '', text)
    text = re.sub(r'You might also like', '', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def fetch_english_lyrics(filename, genre, retries=2):
    track_name = os.path.splitext(os.path.basename(filename))[0]
    cached = _load_cached(genre, 'English', track_name)
    if cached:
        return cached
    if _is_gtzan_filename(filename):
        return None
    genius = _get_genius()
    if genius is None:
        return None
    title, artist = _parse_title_artist(filename)
    for attempt in range(retries):
        try:
            song = genius.search_song(title, artist or '')
            if song and song.lyrics:
                lyrics = _clean_genius_lyrics(song.lyrics)
                _save_cached(lyrics, genre, 'English', track_name)
                return lyrics
        except Exception:
            time.sleep(2 ** attempt)
    return None

# gaanesuno.com scraper (Bangla)
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def _scrape_gaanesuno(title, artist=None):
    query_str = (title + ' ' + (artist or '')).strip().replace(' ', '+')
    try:
        r    = requests.get(f'https://www.gaanesuno.com/?s={query_str}',
                            headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        result = (soup.select_one('h2.entry-title a') or
                  soup.select_one('.search-result a') or
                  soup.select_one('article a'))
        if not result:
            return None
        page  = requests.get(result['href'], headers=HEADERS, timeout=10)
        psoup = BeautifulSoup(page.text, 'html.parser')
        content = (psoup.select_one('.entry-content') or
                   psoup.select_one('.lyric-content') or
                   psoup.select_one('article .post-content'))
        if not content:
            return None
        for tag in content(['script', 'style', 'ins', 'figure']):
            tag.decompose()
        lyrics = content.get_text(separator='\n').strip()
        return lyrics if len(lyrics) > 50 else None
    except Exception:
        return None

def fetch_bangla_lyrics(filename, genre, retries=2):
    track_name = os.path.splitext(os.path.basename(filename))[0]
    cached = _load_cached(genre, 'Bangla', track_name)
    if cached:
        return cached
    title, artist = _parse_title_artist(filename)
    for attempt in range(retries):
        try:
            lyrics = _scrape_gaanesuno(title, artist)
            if lyrics:
                _save_cached(lyrics, genre, 'Bangla', track_name)
                return lyrics
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return None

def fetch_lyrics(filename, genre, language):
    if language == 'English':
        return fetch_english_lyrics(filename, genre)
    elif language == 'Bangla':
        return fetch_bangla_lyrics(filename, genre)
    return None


# Multi-Modal Feature Builder

def make_multimodal(X_audio_sc, records, lyric_dim=LYRIC_DIM):
    """
    Build multi-modal feature matrices from scaled audio + lyrics.

    Returns
    -------
    X_hybrid   : (N, audio_dim + lyric_dim) - L2(audio) || L2(lyrics)
    has_real   : (N,) bool - True where real lyrics were fetched
    X_lyric_l2 : (N, lyric_dim) - L2-normalized lyrics only
    """
    N = len(records)
    lyric_texts = []
    has_real    = np.zeros(N, dtype=bool)

    for i, rec in enumerate(records):
        fpath    = rec.get('file', '')
        genre    = rec.get('genre', 'other')
        language = rec.get('language', 'English')
        lyric    = fetch_lyrics(fpath, genre, language) if fpath else None
        if lyric:
            lyric_texts.append(lyric)
            has_real[i] = True
        else:
            lyric_texts.append(LYRIC_FALLBACK)

    tfidf   = TfidfVectorizer(max_features=2000, ngram_range=(1, 2),
                              sublinear_tf=True, min_df=1)
    X_tfidf = tfidf.fit_transform(lyric_texts).toarray().astype(np.float32)

    n_comp  = min(lyric_dim, X_tfidf.shape[1], X_tfidf.shape[0] - 1)
    n_comp  = max(n_comp, 2)
    X_lyric = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X_tfidf).astype(np.float32)

    if X_lyric.shape[1] < lyric_dim:
        pad     = np.zeros((N, lyric_dim - X_lyric.shape[1]), dtype=np.float32)
        X_lyric = np.hstack([X_lyric, pad])

    audio_l2   = normalize(X_audio_sc, norm='l2')
    X_lyric_l2 = normalize(X_lyric,    norm='l2')
    X_hybrid   = np.hstack([audio_l2, X_lyric_l2]).astype(np.float32)

    real_pct = has_real.mean() * 100
    print(f'  Lyrics     : {has_real.sum()}/{N} real ({real_pct:.1f}%)'
          f' | {N - has_real.sum()} neutral fallback')
    print(f'  X_hybrid   : {X_hybrid.shape}  (L2 audio || L2 lyrics)')
    return X_hybrid, has_real, X_lyric_l2


def make_genre_onehot(y_labels, le):
    """Convert string labels -> one-hot using a fitted LabelEncoder."""
    idx = le.transform(y_labels)
    return np.eye(len(le.classes_), dtype=np.float32)[idx]


# Audio Feature Extraction + Bangla Dataset Builder

def extract_audio_features(fpath, sr=22050, duration=30, n_mfcc=20):
    """
    65-dim feature vector:
      MFCC mean(20) + std(20) = 40
      Chroma STFT mean        = 12
      Spectral centroid, bandwidth, rolloff, ZCR, RMS = 5
      Tempo = 1 | Spectral contrast mean = 7
    Total = 65
    """
    try:
        y, _ = librosa.load(fpath, sr=sr, duration=duration, mono=True)
        if len(y) < sr * 3:
            return None
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
        sc       = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb       = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sr_feat  = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr      = librosa.feature.zero_crossing_rate(y)
        rms      = librosa.feature.rms(y=y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feat = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            chroma.mean(axis=1),
            [sc.mean(), sb.mean(), sr_feat.mean(), zcr.mean(), rms.mean()],
            [float(tempo)],
            contrast.mean(axis=1),
        ]).astype(np.float32)
        return feat
    except Exception:
        return None


def extract_mfcc_2d(fpath, sr=22050, duration=30,
                    n_mfcc=N_MFCC, time_frames=TIME_FRAMES):
    """
    Extract delta-stacked 2D MFCC spectrogram.
    Output: (3 * n_mfcc * time_frames,) flattened = (7680,)
    """
    try:
        y, _ = librosa.load(fpath, sr=sr, duration=duration, mono=True)
        if len(y) < sr * 3:
            return None
        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc   = np.vstack([mfcc, delta, delta2])   # (3*n_mfcc, T)
        if mfcc.shape[1] >= time_frames:
            mfcc = mfcc[:, :time_frames]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, time_frames - mfcc.shape[1])), mode='constant')
        return mfcc.astype(np.float32).flatten()
    except Exception:
        return None


def download_kaggle_dataset(slug, dest_dir):
    """Download & unzip a Kaggle dataset. Returns dest_dir or None."""
    os.makedirs(dest_dir, exist_ok=True)
    audio_exts = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    existing   = []
    for ext in audio_exts:
        existing += glob.glob(f'{dest_dir}/**/*{ext}', recursive=True)
    if existing:
        print(f'  {slug}: {len(existing)} audio files already present.')
        return dest_dir
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print(f'  WARNING: kaggle.json not found -- cannot download {slug}.')
        return None
    print(f'  Downloading Kaggle: {slug} ...')
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', slug, '-p', dest_dir, '--unzip'],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f'  kaggle CLI error: {result.stderr[:200]}')
            return None
        found = []
        for ext in audio_exts:
            found += glob.glob(f'{dest_dir}/**/*{ext}', recursive=True)
        print(f'  {slug}: {len(found)} audio files extracted.')
        return dest_dir if found else None
    except Exception as e:
        print(f'  Kaggle download failed: {e}')
        return None


def collect_audio_from_dir(root_dir, min_per_genre=5):
    """Walk root_dir. Genre = immediate parent directory of audio file."""
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    records    = []
    root_base  = os.path.basename(os.path.normpath(root_dir))
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in audio_exts:
                fpath = os.path.join(dirpath, fname)
                genre = os.path.basename(dirpath)
                if genre in ('.', '', root_base):
                    genre = 'unknown'
                records.append((fpath, genre))
    counts   = Counter(g for _, g in records)
    filtered = [(f, g) for f, g in records if counts[g] >= min_per_genre]
    dropped  = len(records) - len(filtered)
    if dropped:
        print(f'  (dropped {dropped} samples from genres with <{min_per_genre} tracks)')
    return filtered


def download_bangla_yt(genre, query, n=N_BANGLA_PER_GENRE):
    out_dir  = f'{BANGLA_YT_DIR}/{genre}'
    os.makedirs(out_dir, exist_ok=True)
    existing = glob.glob(f'{out_dir}/*.wav') + glob.glob(f'{out_dir}/*.mp3')
    if len(existing) >= n:
        print(f'  {genre}: {len(existing)} files cached.')
        return existing[:n]
    print(f'  yt-dlp {genre}: fetching up to {n} tracks ...')
    cmd = [
        'yt-dlp', f'ytsearch{n}:{query}',
        '--extract-audio', '--audio-format', 'wav', '--audio-quality', '5',
        '--no-playlist', '--output', f'{out_dir}/%(id)s.%(ext)s',
        '--no-warnings', '--ignore-errors', '--socket-timeout', '30', '--retries', '3',
    ]
    try:
        result = subprocess.run(cmd, timeout=300, check=False,
                                capture_output=True, text=True)
        if result.returncode != 0 and result.stderr:
            print(f'  yt-dlp stderr: {result.stderr[:200]}')
    except subprocess.TimeoutExpired:
        print(f'  yt-dlp timeout for {genre}.')
    files = glob.glob(f'{out_dir}/*.wav') + glob.glob(f'{out_dir}/*.mp3')
    print(f'  {len(files)} files downloaded for {genre}')
    return files


# Download GTZAN Dataset

def _stream_dl(url, dest, desc=''):
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True, unit_divisor=1024, desc=desc, ncols=80
    ) as bar:
        for chunk in r.iter_content(65536):
            f.write(chunk); bar.update(len(chunk))

def _resolve_gtzan_audio(search_root):
    wavs = glob.glob(f'{search_root}/**/*.wav', recursive=True)
    if not wavs:
        return search_root, []
    audio_root = os.path.dirname(wavs[0]).rsplit(os.sep, 1)[0]
    return audio_root, wavs

def download_gtzan():
    """
    Download and extract GTZAN dataset. Returns GTZAN_AUDIO path.
    Falls back to Kaggle if HTTP mirrors fail.
    """
    GTZAN_AUDIO = os.path.join(GTZAN_DIR, 'genres')

    existing_wav = glob.glob(f'{GTZAN_AUDIO}/**/*.wav', recursive=True)
    if not existing_wav:
        existing_wav = glob.glob(f'{GTZAN_DIR}/**/*.wav', recursive=True)

    if len(existing_wav) >= 900:
        print(f'GTZAN cached: {len(existing_wav)} .wav files.')
        GTZAN_AUDIO, _ = _resolve_gtzan_audio(GTZAN_DIR)
        return GTZAN_AUDIO

    GTZAN_TAR = os.path.join(GTZAN_DIR, 'genres.tar.gz')
    GTZAN_MIRRORS = [
        'http://opihi.cs.uvic.ca/sound/genres.tar.gz',
        'https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz?download=true',
        'https://archive.org/download/gtzan_genre/genres.tar.gz',
    ]
    downloaded_via_tar    = False
    downloaded_via_kaggle = False

    for url in GTZAN_MIRRORS:
        try:
            print(f'Trying: {url}')
            _stream_dl(url, GTZAN_TAR, desc='GTZAN')
            if os.path.getsize(GTZAN_TAR) < 10_000_000:
                print('  File too small -- likely error page, skipping.')
                os.remove(GTZAN_TAR); continue
            downloaded_via_tar = True; break
        except Exception as e:
            print(f'  Mirror failed: {e}')

    if not downloaded_via_tar:
        print('  All HTTP mirrors failed -- trying Kaggle ...')
        r2 = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d',
             'andradaolteanu/gtzan-dataset-music-genre-classification',
             '-p', GTZAN_DIR, '--unzip'],
            capture_output=True, text=True, timeout=600,
        )
        _, found = _resolve_gtzan_audio(GTZAN_DIR)
        if found:
            print(f'  GTZAN via Kaggle: {len(found)} files.')
            downloaded_via_kaggle = True
        else:
            raise RuntimeError(
                'All GTZAN sources failed.\n'
                'Option A: Upload genres.tar.gz to /content/gtzan/ manually.\n'
                'Option B: Place kaggle.json at ~/.kaggle/ and re-run.'
            )

    if downloaded_via_tar and os.path.exists(GTZAN_TAR):
        print('Extracting GTZAN ...')
        with tarfile.open(GTZAN_TAR, 'r:gz') as tf:
            for m in tqdm(tf.getmembers(), desc='Extracting', unit='file', ncols=80):
                tf.extract(m, GTZAN_DIR)

    GTZAN_AUDIO, existing_wav = _resolve_gtzan_audio(GTZAN_DIR)
    if len(existing_wav) < 900:
        print(f'  WARNING: only {len(existing_wav)} .wav files found.')
    else:
        print(f'GTZAN ready: {len(existing_wav)} .wav files')
    return GTZAN_AUDIO


# Build All Dataset Arrays

def make_records(paths, y_labels, lang_labels):
    return [{'file': p, 'genre': g, 'language': l}
            for p, g, l in zip(paths, y_labels, lang_labels)]


def build_all_datasets(gtzan_audio_dir=None):
    """
    Extract audio features from GTZAN, BanglaGITI, and BMGCD.
    Falls back to yt-dlp for Bangla datasets if Kaggle download fails.

    Returns
    -------
    X_gtzan, X_gtzan_2d, y_gtzan, lang_gtzan, paths_gtzan, records_gtzan
    X_bg, X_bg_2d, y_bg, lang_bg, paths_bg, records_bg
    X_bm, X_bm_2d, y_bm, lang_bm, paths_bm, records_bm
    scaler_all : StandardScaler fitted on all combined audio features
    """
    FEAT_DIM = AUDIO_FEAT_DIM    # 65

    # GTZAN
    if gtzan_audio_dir is None:
        gtzan_audio_dir = download_gtzan()

    print('Loading GTZAN ...')
    gtzan_wav = sorted(glob.glob(f'{gtzan_audio_dir}/**/*.wav', recursive=True))
    if not gtzan_wav:
        gtzan_wav = sorted(glob.glob(f'{GTZAN_DIR}/**/*.wav', recursive=True))
    print(f'  Found {len(gtzan_wav)} .wav files.')

    gc_g = defaultdict(int)
    X_gtzan, X_gtzan_2d, y_gtzan, lang_gtzan, paths_gtzan = [], [], [], [], []
    for fpath in tqdm(gtzan_wav, desc='  librosa GTZAN', leave=False):
        genre = os.path.basename(os.path.dirname(fpath))
        if gc_g[genre] >= N_PER_GENRE:
            continue
        feat    = extract_audio_features(fpath)
        feat_2d = extract_mfcc_2d(fpath)
        if feat is not None and feat_2d is not None:
            X_gtzan.append(feat); X_gtzan_2d.append(feat_2d)
            y_gtzan.append(genre); lang_gtzan.append('English')
            paths_gtzan.append(fpath); gc_g[genre] += 1

    if len(X_gtzan) == 0:
        raise RuntimeError('GTZAN: no features extracted. Check Step 11 download.')
    X_gtzan    = np.array(X_gtzan,    dtype=np.float32)
    X_gtzan_2d = np.array(X_gtzan_2d, dtype=np.float32)
    y_gtzan    = np.array(y_gtzan)
    lang_gtzan = np.array(lang_gtzan)
    print(f'GTZAN: {X_gtzan.shape} | 2D: {X_gtzan_2d.shape} | Genres: {dict(Counter(y_gtzan))}')

    # BanglaGITI
    print('\nLoading BanglaGITI ...')
    bg_dir = download_kaggle_dataset('priyanjanasarkar/banglagiti', BANGLAGITI_DIR)
    X_bg, X_bg_2d, y_bg, lang_bg, paths_bg = [], [], [], [], []
    if bg_dir:
        recs_bg = collect_audio_from_dir(bg_dir, min_per_genre=0)
        gc_bg   = defaultdict(int)
        for fpath, genre in tqdm(recs_bg, desc='  librosa BanglaGITI', leave=False):
            if gc_bg[genre] >= N_PER_GENRE: continue
            feat    = extract_audio_features(fpath)
            feat_2d = extract_mfcc_2d(fpath)
            if feat is not None and feat_2d is not None:
                X_bg.append(feat); X_bg_2d.append(feat_2d)
                y_bg.append(genre); lang_bg.append('Bangla')
                paths_bg.append(fpath); gc_bg[genre] += 1

    if len(X_bg) < MIN_SAMPLES_FOR_METRICS:
        print('  BanglaGITI unavailable -- yt-dlp fallback ...')
        gc_fb = defaultdict(int)
        for genre, query in BANGLA_QUERIES_YT.items():
            files = download_bangla_yt(genre, query, n=N_PER_GENRE)
            for fpath in tqdm(files, desc=f'  librosa {genre}', leave=False):
                if gc_fb[genre] >= N_PER_GENRE: continue
                feat    = extract_audio_features(fpath)
                feat_2d = extract_mfcc_2d(fpath)
                if feat is not None and feat_2d is not None:
                    X_bg.append(feat); X_bg_2d.append(feat_2d)
                    y_bg.append(genre); lang_bg.append('Bangla')
                    paths_bg.append(fpath); gc_fb[genre] += 1

    if len(X_bg) == 0:
        print('  WARNING: BanglaGITI has 0 usable samples.')
        X_bg    = np.zeros((0, FEAT_DIM),     dtype=np.float32)
        X_bg_2d = np.zeros((0, MFCC_2D_DIM),  dtype=np.float32)
        y_bg    = np.array([]); lang_bg = np.array([])
    else:
        X_bg    = np.array(X_bg,    dtype=np.float32)
        X_bg_2d = np.array(X_bg_2d, dtype=np.float32)
        y_bg    = np.array(y_bg); lang_bg = np.array(lang_bg)
        print(f'BanglaGITI: {X_bg.shape} | Genres: {dict(Counter(y_bg))}')

    # BMGCD
    print('\nLoading BMGCD ...')
    bm_dir = download_kaggle_dataset(
        'mdimranhassan/bangla-music-genre-classification', BMGCD_DIR)
    X_bm, X_bm_2d, y_bm, lang_bm, paths_bm = [], [], [], [], []
    if bm_dir:
        recs_bm = collect_audio_from_dir(bm_dir, min_per_genre=10)
        gc_bm   = defaultdict(int)
        for fpath, genre in tqdm(recs_bm, desc='  librosa BMGCD', leave=False):
            if gc_bm[genre] >= N_PER_GENRE: continue
            feat    = extract_audio_features(fpath)
            feat_2d = extract_mfcc_2d(fpath)
            if feat is not None and feat_2d is not None:
                X_bm.append(feat); X_bm_2d.append(feat_2d)
                y_bm.append(genre); lang_bm.append('Bangla')
                paths_bm.append(fpath); gc_bm[genre] += 1

    if len(X_bm) < MIN_SAMPLES_FOR_METRICS:
        print('  BMGCD unavailable -- yt-dlp fallback ...')
        gc_fb2 = defaultdict(int)
        for genre, query in BMGCD_QUERIES_YT.items():
            files = download_bangla_yt(genre, query, n=N_PER_GENRE)
            for fpath in tqdm(files, desc=f'  librosa {genre}', leave=False):
                if gc_fb2[genre] >= N_PER_GENRE: continue
                feat    = extract_audio_features(fpath)
                feat_2d = extract_mfcc_2d(fpath)
                if feat is not None and feat_2d is not None:
                    X_bm.append(feat); X_bm_2d.append(feat_2d)
                    y_bm.append(genre); lang_bm.append('Bangla')
                    paths_bm.append(fpath); gc_fb2[genre] += 1

    if len(X_bm) == 0:
        print('  WARNING: BMGCD has 0 usable samples.')
        X_bm    = np.zeros((0, FEAT_DIM),     dtype=np.float32)
        X_bm_2d = np.zeros((0, MFCC_2D_DIM),  dtype=np.float32)
        y_bm    = np.array([]); lang_bm = np.array([])
    else:
        X_bm    = np.array(X_bm,    dtype=np.float32)
        X_bm_2d = np.array(X_bm_2d, dtype=np.float32)
        y_bm    = np.array(y_bm); lang_bm = np.array(lang_bm)
        print(f'BMGCD: {X_bm.shape} | Genres: {dict(Counter(y_bm))}')

    # Records for make_multimodal()
    records_gtzan = make_records(paths_gtzan, y_gtzan, lang_gtzan)
    records_bg    = make_records(paths_bg,    y_bg,    lang_bg)
    records_bm    = make_records(paths_bm,    y_bm,    lang_bm)

    # Fit shared scaler on all combined audio features
    _parts = [X for X in [X_gtzan, X_bg, X_bm] if len(X) > 0]
    X_all_for_scaler = np.vstack(_parts).astype(np.float32)
    scaler_all = StandardScaler().fit(X_all_for_scaler)
    print(f'\nscaler_all fitted on {X_all_for_scaler.shape[0]} samples (all datasets combined)')

    # Summary
    print()
    print('=' * 60)
    print('  DATASET SUMMARY')
    print('=' * 60)
    for name, X, y in [('GTZAN', X_gtzan, y_gtzan),
                       ('BanglaGITI', X_bg, y_bg),
                       ('BMGCD', X_bm, y_bm)]:
        if len(X) > 0:
            print(f'  {name:<12}: {X.shape}  Genres: {len(np.unique(y))}  -> {dict(Counter(y))}')
        else:
            print(f'  {name:<12}: EMPTY -- skipped')
    print('=' * 60)

    return (X_gtzan, X_gtzan_2d, y_gtzan, lang_gtzan, paths_gtzan, records_gtzan,
            X_bg,    X_bg_2d,    y_bg,    lang_bg,    paths_bg,    records_bg,
            X_bm,    X_bm_2d,    y_bm,    lang_bm,    paths_bm,    records_bm,
            scaler_all)


print('dataset.py loaded')
print(f'   extract_audio_features: {AUDIO_FEAT_DIM}-dim')
print(f'   extract_mfcc_2d       : {MFCC_2D_DIM}-dim  (MFCC+delta+delta2 x {TIME_FRAMES})')
