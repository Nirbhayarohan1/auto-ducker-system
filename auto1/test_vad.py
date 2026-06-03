#!/usr/bin/env python3
"""
Test suite for STFT-based Voice Activity Detection.

Tests VAD on audio clips from a /test_clips folder and reports accuracy.
Supports speech vs. silence/noise classification.

Usage:
    python test_vad.py

Expected structure:
    test_clips/
        speech/
            *.wav, *.mp3, etc.
        silence/
            *.wav, *.mp3, etc.
    OR label files with "speech" or "silence"/"noise" in the filename.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed. Install with: pip install librosa")
    exit(1)

# Import the VAD function from auto3
try:
    from auto3 import compute_vad_score
except ImportError:
    print("ERROR: Could not import compute_vad_score from auto3.py")
    print("Make sure auto3.py is in the same directory.")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Configuration
SAMPLE_RATE = 16000
TEST_CLIPS_DIR = "test_clips"


def get_ground_truth(file_path: str) -> Optional[bool]:
    """
    Extract ground truth label from file path/name.
    
    Returns:
        True if speech, False if silence/noise, None if unknown
    """
    lower_path = str(file_path).lower()
    
    # Check folder name
    if "/speech/" in lower_path or "\\speech\\" in lower_path:
        return True
    if "/silence/" in lower_path or "\\silence\\" in lower_path:
        return False
    if "/noise/" in lower_path or "\\noise\\" in lower_path:
        return False
    
    # Check filename
    filename = os.path.basename(file_path).lower()
    if "speech" in filename or "talk" in filename or "voice" in filename:
        return True
    if "silence" in filename or "noise" in filename or "quiet" in filename:
        return False
    
    return None


def compute_vad_on_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    chunk_size: int = 1024
) -> float:
    """
    Compute average VAD score over an entire audio clip by processing in chunks.
    
    Args:
        audio: Audio waveform (float or int16)
        sr: Sample rate
        chunk_size: Size of chunks for processing
    
    Returns:
        Average VAD score across all chunks
    """
    scores = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        
        # Skip if chunk is too small for meaningful STFT
        if len(chunk) < 512:
            continue
        
        # Convert to int16 if needed
        if audio.dtype != np.int16:
            chunk_int16 = (chunk * 32767).astype(np.int16)
        else:
            chunk_int16 = chunk.astype(np.int16)
        
        score = compute_vad_score(chunk_int16, sr=sr)
        if score >= 0:
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def test_vad_on_clip(clip_path: str) -> Tuple[Optional[bool], bool, float]:
    """
    Test VAD on a single audio clip.
    
    Args:
        clip_path: Path to audio file
    
    Returns:
        Tuple of (ground_truth, prediction, vad_score)
        - ground_truth: True/False/None (from filename/folder)
        - prediction: True if VAD detected speech, False otherwise
        - vad_score: Raw VAD score
    """
    filename = os.path.basename(clip_path)
    ground_truth = get_ground_truth(clip_path)
    
    try:
        # Load audio at target sample rate
        audio, sr = librosa.load(clip_path, sr=SAMPLE_RATE, mono=True)
        
        # Compute average VAD score across the clip
        vad_score = compute_vad_on_audio(audio, sr=SAMPLE_RATE)
        
        # Simple threshold-based prediction
        # Threshold tuned empirically; adjust based on your test results
        threshold = 40.0
        prediction = vad_score > threshold
        
        return ground_truth, prediction, vad_score
    
    except Exception as e:
        log.error(f"Error processing {clip_path}: {e}")
        return None, None, 0.0


def main():
    """Run VAD tests on all audio clips in the test_clips folder."""
    if not os.path.exists(TEST_CLIPS_DIR):
        log.error(f"Test clips directory not found: {TEST_CLIPS_DIR}")
        log.info("Create a 'test_clips' folder with audio files labeled:")
        log.info("  - Folders: test_clips/speech/ and test_clips/silence/")
        log.info("  - OR filenames: *_speech.wav, *_silence.wav, *_noise.wav")
        return
    
    # Find all audio files
    clip_files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
        clip_files.extend(Path(TEST_CLIPS_DIR).glob(f"**/{ext}"))
        # Case-insensitive glob
        clip_files.extend(Path(TEST_CLIPS_DIR).glob(f"**/{ext.upper()}"))
    
    if not clip_files:
        log.info(f"No audio clips found in {TEST_CLIPS_DIR}")
        log.info("Supported formats: .wav, .mp3, .flac, .ogg, .m4a")
        return
    
    log.info(f"Found {len(clip_files)} audio clips. Running VAD tests...\n")
    
    results = []
    for clip_path in sorted(clip_files):
        ground_truth, prediction, vad_score = test_vad_on_clip(str(clip_path))
        
        # Skip clips without ground truth labels
        if ground_truth is None:
            log.warning(f"  Skipped (no label in path): {clip_path.name}")
            continue
        
        if prediction is None:
            log.warning(f"  Failed to process: {clip_path.name}")
            continue
        
        is_correct = ground_truth == prediction
        results.append({
            "file": clip_path.name,
            "path": str(clip_path),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "vad_score": vad_score,
            "correct": is_correct
        })
    
    if not results:
        log.warning("No valid test results (no clips with ground truth labels)")
        log.info("Label your files as: *_speech.*, *_silence.*, or *_noise.*")
        log.info("Or organize them in speech/ and silence/ subdirectories")
        return
    
    # Calculate and report results
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    accuracy = (correct_count / total_count) * 100.0
    
    log.info("=" * 80)
    log.info("VAD TEST RESULTS")
    log.info("=" * 80)
    log.info(f"{'Status':<8} {'Filename':<40} {'Expected':<12} {'Detected':<12} {'Score':<10}")
    log.info("-" * 80)
    
    for r in sorted(results, key=lambda x: (not x["correct"], x["file"])):
        status = "✓ PASS" if r["correct"] else "✗ FAIL"
        expected = "SPEECH" if r["ground_truth"] else "SILENCE"
        detected = "SPEECH" if r["prediction"] else "SILENCE"
        score_str = f"{r['vad_score']:.1f}"
        
        log.info(
            f"{status:<8} {r['file']:<40} {expected:<12} {detected:<12} {score_str:<10}"
        )
    
    log.info("=" * 80)
    log.info(f"Accuracy: {correct_count}/{total_count} = {accuracy:.1f}%")
    log.info("=" * 80)
    
    if accuracy < 80.0:
        log.warning("\nAccuracy below 80%. Consider adjusting:")
        log.warning("  - Threshold in compute_vad_on_audio() function")
        log.warning("  - Checking for mislabeled test clips")
        log.warning("  - Ensuring test audio is clean and representative")
    else:
        log.info("\n✓ VAD performance is good!")


if __name__ == "__main__":
    main()
