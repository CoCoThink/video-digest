"""Bounded 16 kHz PCM chunking with low-energy cut points."""
from __future__ import annotations

from pathlib import Path
import wave


def _write_wav(path: Path, params, audio: bytes, frame_count: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setparams(params)
        writer.setnframes(frame_count)
        writer.writeframes(audio)


def _low_energy_boundary(
    reader: wave.Wave_read,
    start_frame: int,
    target_frame: int,
    search_frames: int,
    window_frames: int,
) -> int:
    """Find a quiet cut before the target, so chunks never exceed their limit."""
    if search_frames <= 0:
        return target_frame

    import numpy as np

    left = max(start_frame + 1, target_frame - search_frames)
    frame_count = target_frame - left
    if frame_count <= window_frames:
        return target_frame

    reader.setpos(left)
    samples = np.frombuffer(reader.readframes(frame_count), dtype="<i2")
    amplitudes = np.abs(samples.astype(np.int32, copy=False))
    window_frames = min(window_frames, amplitudes.size)
    prefix = np.empty(amplitudes.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(amplitudes, dtype=np.int64, out=prefix[1:])
    energies = prefix[window_frames:] - prefix[:-window_frames]

    # On equal-energy silence, prefer the boundary closest to the target.
    minimum = energies.min()
    window_start = int(np.flatnonzero(energies == minimum)[-1])
    local = amplitudes[window_start:window_start + window_frames]
    local_minimum = local.min()
    local_index = int(np.flatnonzero(local == local_minimum)[-1])
    boundary = left + window_start + local_index + 1
    return max(start_frame + 1, min(target_frame, boundary))


def iter_wav_chunks(
    source: Path,
    directory: Path,
    chunk_sec: float,
    *,
    boundary_search_sec: float = 0.0,
    energy_window_ms: float = 100.0,
):
    """Yield contiguous chunks, deleting each materialized WAV after use."""
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")
    if boundary_search_sec < 0 or boundary_search_sec >= chunk_sec:
        raise ValueError("boundary_search_sec must be in [0, chunk_sec)")
    if energy_window_ms <= 0:
        raise ValueError("energy_window_ms must be positive")

    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        if params.nchannels != 1 or params.sampwidth != 2 or params.framerate != 16_000:
            raise ValueError("ASR expects 16 kHz, 16-bit, mono PCM WAV")

        frames_per_chunk = max(1, round(chunk_sec * params.framerate))
        search_frames = round(boundary_search_sec * params.framerate)
        window_frames = max(4, round(energy_window_ms / 1000 * params.framerate))
        start_frame = 0
        index = 0
        while start_frame < params.nframes:
            target_frame = min(start_frame + frames_per_chunk, params.nframes)
            if target_frame < params.nframes:
                end_frame = _low_energy_boundary(
                    reader, start_frame, target_frame, search_frames, window_frames,
                )
            else:
                end_frame = target_frame
            frame_count = end_frame - start_frame
            path = directory / f"chunk_{index:05d}.wav"
            reader.setpos(start_frame)
            _write_wav(path, params, reader.readframes(frame_count), frame_count)
            try:
                yield (
                    path,
                    start_frame / params.framerate,
                    frame_count / params.framerate,
                )
            finally:
                path.unlink(missing_ok=True)
            start_frame = end_frame
            index += 1
