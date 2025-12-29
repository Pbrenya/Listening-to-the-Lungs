#!/usr/bin/env node

/**
 * Example: Run inference on a WAV file
 * Usage: node js/examples/inference.js path/to/file.wav
 */

import * as tf from '@tensorflow/tfjs-node';
import { loadWav, padOrTrim, extractMel, extractHandcrafted } from '../features.js';
import { CFG } from '../data.js';
import path from 'path';
import fs from 'fs';

async function main(wavPath) {
  if (!fs.existsSync(wavPath)) {
    console.error(`File not found: ${wavPath}`);
    process.exit(1);
  }

  console.log(`Loading audio from ${wavPath}...`);
  
  // Load and preprocess audio
  const y = loadWav(wavPath, CFG.sampleRate);
  const yPadded = padOrTrim(y);
  
  // Extract features
  const melTensor = extractMel(yPadded, CFG.sampleRate, CFG.nMels);
  const hand = extractHandcrafted(yPadded, CFG.sampleRate);
  
  // Reshape for model input
  const melInput = melTensor.expandDims(0).expandDims(-1);
  const handInput = tf.tensor2d([Array.from(hand)]);
  
  // Load model
  const modelPath = path.join(CFG.workDir, 'model_js');
  
  if (!fs.existsSync(path.join(modelPath, 'model.json'))) {
    console.error(`No trained model found at ${modelPath}`);
    console.log('Please run training first');
    process.exit(1);
  }
  
  console.log('Loading model...');
  const model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
  
  // Predict
  console.log('Running inference...');
  const pred = model.predict([melInput, handInput]);
  const predArray = await pred.array();
  const classIdx = pred.argMax(-1).dataSync()[0];
  const prob = predArray[0][classIdx];
  
  console.log('\n=== Prediction Results ===');
  console.log(`Predicted: ${CFG.classes[classIdx]}`);
  console.log(`Confidence: ${(prob * 100).toFixed(2)}%`);
  console.log('\nAll probabilities:');
  CFG.classes.forEach((cls, idx) => {
    const p = predArray[0][idx];
    const bar = '█'.repeat(Math.round(p * 50));
    console.log(`  ${cls.padEnd(12)} ${(p * 100).toFixed(2)}% ${bar}`);
  });
  
  // Clean up
  melTensor.dispose();
  melInput.dispose();
  handInput.dispose();
  pred.dispose();
}

// Get WAV path from command line
const wavPath = process.argv[2];

if (!wavPath) {
  console.log('Usage: node js/examples/inference.js path/to/file.wav');
  process.exit(1);
}

main(wavPath).catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
