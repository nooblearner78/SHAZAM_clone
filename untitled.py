import io
import os
import csv
import pickle
import tempfile
from pathlib import Path
from collections import defaultdict

import librosa
import librosa.display
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# =============================================================================
# CONFIGURATION
# =============================================================================
APP_TITLE = "Audio Fingerprinting Demo"
DEFAULT_SONGS_DIR = "./song data"
DEFAULT_DB_PATH = "fingerprints.pkl"
DEFAULT_QUERY_DURATION = 5
DEFAULT_SAMPLING_RATE = None
FFT_WINDOW_SIZE = 2048
HOP_LENGTH = 512
NEIGHBORHOOD_SIZE = (20, 20)
BACKGROUND_THRESHOLD = -40
FAN_VALUE = 5
VALID_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


# =============================================================================
# CORE FINGERPRINTING FUNCTIONS
# =============================================================================

def extract_constellation_map(audio_path, duration=None):
    """Extract sparse spectral peaks from an audio file."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    y, sr = librosa.load(audio_path, sr=DEFAULT_SAMPLING_RATE, duration=duration)
    if y.size == 0:
        return []

    stft_matrix = librosa.stft(y, n_fft=FFT_WINDOW_SIZE, hop_length=HOP_LENGTH)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)

    local_max_mask = (
        ndimage.maximum_filter(spectrogram_db, size=NEIGHBORHOOD_SIZE) == spectrogram_db
    )
    threshold_mask = spectrogram_db > BACKGROUND_THRESHOLD
    peaks_mask = local_max_mask & threshold_mask

    freq_indices, time_indices = np.where(peaks_mask)
    return list(zip(freq_indices, time_indices))


def generate_hashes(peaks, fan_value=FAN_VALUE):
    """Convert peaks into hash tuples with their anchor time."""
    hashes = []
    peaks = sorted(peaks, key=lambda x: x[1])
    num_peaks = len(peaks)

    for i in range(num_peaks):
        anchor_freq, anchor_time = peaks[i]
        for j in range(1, fan_value + 1):
            if i + j < num_peaks:
                target_freq, target_time = peaks[i + j]
                time_delta = target_time - anchor_time
                hash_tuple = (int(anchor_freq), int(target_freq), int(time_delta))
                hashes.append({
                    "hash": hash_tuple,
                    "absolute_time": int(anchor_time),
                })
    return hashes


def fingerprint_audio_file(audio_path, duration=None):
    """Helper: peaks + hashes for a single audio file."""
    peaks = extract_constellation_map(audio_path, duration=duration)
    hashes = generate_hashes(peaks)
    return peaks, hashes


# =============================================================================
# DATABASE BUILDING / LOADING / SAVING
# =============================================================================

def build_song_database(folder_path, progress_callback=None):
    """Index all audio files in a folder into a fingerprint database."""
    database = defaultdict(list)
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder '{folder_path}' not found. Create it and add reference songs."
        )

    audio_files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VALID_AUDIO_EXTENSIONS]
    if not audio_files:
        raise ValueError(f"No audio files found in '{folder_path}'.")

    total_files = len(audio_files)
    total_hashes = 0

    for idx, file_path in enumerate(audio_files, start=1):
        peaks, song_hashes = fingerprint_audio_file(str(file_path), duration=None)
        for h in song_hashes:
            database[h["hash"]].append((file_path.name, h["absolute_time"]))
        total_hashes += len(song_hashes)

        if progress_callback is not None:
            progress_callback(idx, total_files, file_path.name, len(peaks), len(song_hashes))

    return dict(database), {
        "songs_indexed": total_files,
        "hashes_generated": total_hashes,
    }


def save_database(database, path):
    with open(path, "wb") as f:
        pickle.dump(database, f)


def load_database(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def database_to_bytes(database):
    return pickle.dumps(database)


# =============================================================================
# MATCHING / VISUALIZATION HELPERS
# =============================================================================

def match_hashes(query_hashes, database):
    """Return tallies and best match for a query against the database."""
    offset_tallies = defaultdict(int)
    song_histograms = defaultdict(lambda: defaultdict(int))

    for q_hash in query_hashes:
        hash_val = q_hash["hash"]
        q_time = q_hash["absolute_time"]

        if hash_val in database:
            for song_name, db_time in database[hash_val]:
                time_offset = db_time - q_time
                offset_tallies[(song_name, time_offset)] += 1
                song_histograms[song_name][time_offset] += 1

    if not offset_tallies:
        return None, None, None

    best_match = max(offset_tallies, key=offset_tallies.get)
    best_song, best_offset = best_match
    return best_match, offset_tallies, song_histograms


def build_diagnostic_figure(audio_path, query_duration, database):
    """Create the 3-panel diagnostic figure for a single clip."""
    y, sr = librosa.load(audio_path, sr=None, duration=query_duration)
    stft_matrix = librosa.stft(y, n_fft=FFT_WINDOW_SIZE, hop_length=HOP_LENGTH)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)

    query_peaks, query_hashes = fingerprint_audio_file(audio_path, duration=query_duration)
    best_match, offset_tallies, song_histograms = match_hashes(query_hashes, database)

    if best_match is None:
        return None, None, None, None, None

    best_song, best_offset = best_match
    best_song_data = song_histograms[best_song]
    sorted_items = sorted(best_song_data.items())
    offsets = [k for k, _ in sorted_items]
    counts = [v for _, v in sorted_items]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Spectrogram
    librosa.display.specshow(
        spectrogram_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="linear",
        ax=axes[0],
        cmap="magma",
    )
    axes[0].set_title("1. Spectrogram")
    axes[0].set_ylim(0, 8000)

    # Constellation peaks
    librosa.display.specshow(
        spectrogram_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="linear",
        ax=axes[1],
        cmap="magma",
    )
    if query_peaks:
        freq_bins, time_frames = zip(*query_peaks)
        times = librosa.frames_to_time(time_frames, sr=sr, hop_length=HOP_LENGTH)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_WINDOW_SIZE)[list(freq_bins)]
        axes[1].scatter(times, freqs, edgecolor="cyan", facecolor="none", s=50)
    axes[1].set_title("2. Constellation of Peaks")
    axes[1].set_ylim(0, 8000)

    # Offset histogram
    axes[2].plot(offsets, counts, marker="o", linewidth=1)
    axes[2].axvline(best_offset, linestyle="--")
    axes[2].set_title(f"3. Offset Histogram ({best_song})")
    axes[2].set_xlabel("Time Offset (Frames)")
    axes[2].set_ylabel("Number of Hash Matches")
    if offsets:
        axes[2].set_xlim(best_offset - 200, best_offset + 200)

    plt.tight_layout()
    return fig, best_match, offset_tallies, song_histograms, query_hashes


def fingerprint_uploaded_file(uploaded_file, duration=None):
    """Save an uploaded file to a temp path and fingerprint it."""
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    try:
        peaks, hashes = fingerprint_audio_file(temp_path, duration=duration)
        return temp_path, peaks, hashes
    finally:
        # The temp file is intentionally not deleted immediately because librosa
        # may need to reopen it in downstream plotting paths. Caller can delete.
        pass


def match_and_predict(query_hashes, database):
    best_match, offset_tallies, song_histograms = match_hashes(query_hashes, database)
    if best_match is None:
        return None, None, None, None
    best_song, best_offset = best_match
    score = offset_tallies[best_match]
    return best_song, best_offset, score, song_histograms


# =============================================================================
# STREAMLIT UI
# =============================================================================

def init_state():
    if "database" not in st.session_state:
        st.session_state.database = None
    if "database_stats" not in st.session_state:
        st.session_state.database_stats = None
    if "database_path" not in st.session_state:
        st.session_state.database_path = DEFAULT_DB_PATH


def sidebar_database_controls():
    st.sidebar.header("Database")
    db_path = st.sidebar.text_input("Fingerprint file", value=st.session_state.database_path)
    st.session_state.database_path = db_path

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("Load DB"):
            if os.path.exists(db_path):
                st.session_state.database = load_database(db_path)
                st.session_state.database_stats = {"source": db_path}
                st.sidebar.success("Database loaded.")
            else:
                st.sidebar.error(f"Not found: {db_path}")

    with col_b:
        uploaded_db = st.file_uploader("Upload DB", type=["pkl", "pickle"], label_visibility="collapsed")
        if uploaded_db is not None:
            st.session_state.database = pickle.load(uploaded_db)
            st.session_state.database_stats = {"source": uploaded_db.name}
            st.sidebar.success("Database uploaded.")

    if st.session_state.database is not None:
        st.sidebar.success("Database ready")
        st.sidebar.write(f"Entries: {len(st.session_state.database):,}")
    else:
        st.sidebar.warning("No database loaded")


def home_page():
    st.title(APP_TITLE)
    st.write(
        "This app builds an audio fingerprint database from a folder of reference songs "
        "and matches uploaded clips against that database."
    )
    st.markdown(
        """
        **Workflow**
        1. Build or load the fingerprint database.
        2. Upload a single clip to identify a song and inspect the plots.
        3. Upload multiple clips to evaluate a batch and download the CSV.
        """
    )

    if st.session_state.database is not None:
        st.info("A fingerprint database is currently loaded.")
    else:
        st.warning("Load an existing `fingerprints.pkl` or build a new database from the songs folder.")


def build_database_page():
    st.title("Build Fingerprint Database")
    songs_dir = st.text_input("Songs folder", value=DEFAULT_SONGS_DIR)
    db_out = st.text_input("Output fingerprint file", value=st.session_state.database_path)

    st.caption(
        "This runs the indexing phase once, then saves the result as a pickle file. "
        "Use the saved file later for faster matching."
    )

    if st.button("Build Database"):
        progress_bar = st.progress(0)
        status = st.empty()

        def progress_callback(i, total, file_name, num_peaks, num_hashes):
            pct = int(i / total * 100)
            progress_bar.progress(pct)
            status.write(
                f"Indexing {i}/{total}: **{file_name}** | peaks={num_peaks:,} | hashes={num_hashes:,}"
            )

        try:
            with st.spinner("Building fingerprint database..."):
                database, stats = build_song_database(songs_dir, progress_callback=progress_callback)
                save_database(database, db_out)
                st.session_state.database = database
                st.session_state.database_stats = stats
                st.session_state.database_path = db_out

            progress_bar.progress(100)
            st.success(f"Database built and saved to `{db_out}`")
            st.write(stats)

            db_bytes = database_to_bytes(database)
            st.download_button(
                "Download fingerprints.pkl",
                data=db_bytes,
                file_name=os.path.basename(db_out) or "fingerprints.pkl",
                mime="application/octet-stream",
            )
        except Exception as e:
            st.error(f"Failed to build database: {e}")


def single_clip_page():
    st.title("Single Clip Recognition")

    if st.session_state.database is None:
        st.warning("Load or build the fingerprint database first.")
        return

    uploaded = st.file_uploader("Upload a clip", type=["mp3", "wav", "m4a", "flac", "ogg"])
    query_duration = st.slider("Clip duration (seconds)", 1, 20, DEFAULT_QUERY_DURATION)

    if uploaded is not None:
        if st.button("Identify Clip"):
            temp_path = None
            try:
                with st.spinner("Analyzing clip..."):
                    suffix = Path(uploaded.name).suffix or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getbuffer())
                        temp_path = tmp.name

                    fig, best_match, offset_tallies, song_histograms, query_hashes = build_diagnostic_figure(
                        temp_path,
                        query_duration=query_duration,
                        database=st.session_state.database,
                    )

                if best_match is None:
                    st.error("No matches found.")
                    return

                best_song, best_offset = best_match
                score = offset_tallies[best_match]
                st.success(f"Prediction: {best_song}")
                st.write(f"Best alignment offset: {best_offset} frames")
                st.write(f"Offset spike: {score} aligned hashes")
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Failed to analyze clip: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass


def batch_page():
    st.title("Batch Evaluation")

    if st.session_state.database is None:
        st.warning("Load or build the fingerprint database first.")
        return

    uploads = st.file_uploader(
        "Upload clips",
        type=["mp3", "wav", "m4a", "flac", "ogg"],
        accept_multiple_files=True,
    )

    query_duration = st.slider("Clip duration (seconds)", 1, 20, DEFAULT_QUERY_DURATION, key="batch_duration")

    if uploads and st.button("Run Batch Evaluation"):
        results = []
        progress_bar = st.progress(0)
        status = st.empty()

        for idx, uploaded in enumerate(uploads, start=1):
            temp_path = None
            try:
                suffix = Path(uploaded.name).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    temp_path = tmp.name

                _, hashes = fingerprint_audio_file(temp_path, duration=query_duration)
                best_match, offset_tallies, _ = match_hashes(hashes, st.session_state.database)

                if best_match is None:
                    prediction = "NO_MATCH"
                else:
                    prediction = best_match[0]

                results.append({
                    "filename": uploaded.name,
                    "prediction": os.path.splitext(prediction)[0] if prediction not in {"NO_MATCH", None} else prediction,
                })
                status.write(f"Processed {idx}/{len(uploads)}: {uploaded.name}")
            except Exception as e:
                results.append({"filename": uploaded.name, "prediction": f"ERROR: {e}"})
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

            progress_bar.progress(int(idx / len(uploads) * 100))

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()
    sidebar_database_controls()

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Build Database", "Single Clip", "Batch Evaluation"],
    )

    if page == "Home":
        home_page()
    elif page == "Build Database":
        build_database_page()
    elif page == "Single Clip":
        single_clip_page()
    elif page == "Batch Evaluation":
        batch_page()


if __name__ == "__main__":
    main()
