/**
 * Audio feature extraction utilities
 */

import * as tf from '@tensorflow/tfjs-node';
import { WaveFile } from 'wavefile';
import Meyda from 'meyda';
import fs from 'fs';

/**
 * Load a WAV file and convert to mono float32 array at target sample rate
 * @param {string} path - Path to WAV file
 * @param {number} targetSr - Target sample rate (default: 16000)
 * @returns {Float32Array} - Audio samples
 */
export function loadWav(path, targetSr = 16000) {
  const buffer = fs.readFileSync(path);
  const wav = new WaveFile(buffer);
  
  // Convert to mono if stereo
  if (wav.fmt.numChannels > 1) {
    wav.toMono();
  }
  
  // Resample if needed (simple approach - for production use proper resampling)
  const currentSr = wav.fmt.sampleRate;
  const samples = wav.getSamples(false, Float32Array);
  
  if (currentSr !== targetSr) {
    console.warn(`Resampling from ${currentSr} to ${targetSr} Hz (simplified resampling)`);
    // Simple linear interpolation resampling
    const ratio = currentSr / targetSr;
    const newLength = Math.floor(samples.length / ratio);
    const resampled = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const srcIdx = i * ratio;
      const srcIdxFloor = Math.floor(srcIdx);
      const srcIdxCeil = Math.min(srcIdxFloor + 1, samples.length - 1);
      const t = srcIdx - srcIdxFloor;
      resampled[i] = samples[srcIdxFloor] * (1 - t) + samples[srcIdxCeil] * t;
    }
    return resampled;
  }
  
  return samples;
}

/**
 * Pad or trim audio to target length
 * @param {Float32Array} y - Audio samples
 * @param {number} targetLen - Target length in samples (default: 4s at 16kHz = 64000)
 * @returns {Float32Array} - Padded or trimmed audio
 */
export function padOrTrim(y, targetLen = 4.0 * 16000) {
  if (y.length < targetLen) {
    const padded = new Float32Array(targetLen);
    padded.set(y);
    return padded;
  }
  return y.slice(0, targetLen);
}

/**
 * Extract mel-spectrogram from audio
 * @param {Float32Array} y - Audio samples
 * @param {number} sr - Sample rate
 * @param {number} nMels - Number of mel bins (default: 128)
 * @returns {tf.Tensor} - Mel-spectrogram in dB
 */
export function extractMel(y, sr, nMels = 128) {
  // Convert to tensor
  const audioTensor = tf.tensor1d(Array.from(y));
  
  // Compute STFT
  const frameLength = 2048;
  const frameStep = 512;
  const fftLength = 2048;
  
  const stft = tf.signal.stft(
    audioTensor,
    frameLength,
    frameStep,
    fftLength,
    tf.signal.hannWindow
  );
  
  // Get magnitude
  const magnitude = tf.abs(stft);
  
  // Create mel filter bank
  const numSpectrumBins = fftLength / 2 + 1;
  const linearToMelMatrix = tf.signal.linearToMelWeight(
    nMels,
    numSpectrumBins,
    sr,
    0,
    sr / 2
  );
  
  // Apply mel filter bank
  const melSpec = tf.matMul(magnitude, linearToMelMatrix);
  
  // Convert to dB scale
  const melPower = tf.square(melSpec);
  const melDb = tf.mul(
    10.0,
    tf.log(tf.add(melPower, 1e-10)).div(tf.log(10.0))
  );
  
  audioTensor.dispose();
  magnitude.dispose();
  melSpec.dispose();
  melPower.dispose();
  
  return melDb;
}

/**
 * Extract handcrafted features from audio
 * @param {Float32Array} y - Audio samples
 * @param {number} sr - Sample rate
 * @returns {Float32Array} - Handcrafted features [zcr, centroid, bandwidth]
 */
export function extractHandcrafted(y, sr) {
  // Convert to array for Meyda
  const audioArray = Array.from(y);
  
  // Configure Meyda
  const features = Meyda.extract(['zcr', 'spectralCentroid', 'spectralBandwidth'], audioArray);
  
  // Simple averaging approach for demonstration
  // In production, compute over frames
  const zcr = features.zcr || 0;
  const centroid = features.spectralCentroid || 0;
  const bandwidth = features.spectralBandwidth || 0;
  
  return new Float32Array([zcr, centroid, bandwidth]);
}

/**
 * Extract MFCC features from audio
 * @param {Float32Array} y - Audio samples
 * @param {number} sr - Sample rate
 * @param {number} nMfcc - Number of MFCC coefficients (default: 20)
 * @returns {Array} - MFCC coefficients
 */
export function extractMfcc(y, sr, nMfcc = 20) {
  // Configure Meyda for MFCC
  const audioArray = Array.from(y);
  const mfcc = Meyda.extract('mfcc', audioArray);
  
  return mfcc || new Array(nMfcc).fill(0);
}
