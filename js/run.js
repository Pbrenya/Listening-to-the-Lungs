#!/usr/bin/env node

/**
 * run.js - CLI entrypoint for common tasks:
 *   - node js/run.js train    (runs a smoke training run)
 *   - node js/run.js xai      (runs a Grad-CAM demo)
 *   - node js/run.js infer --wav path/to.wav  (demo inference)
 */

import { Command } from 'commander';
import * as tf from '@tensorflow/tfjs-node';
import { buildModel } from './model.js';
import { gradCAM } from './xai.js';
import { loadWav, padOrTrim, extractMel, extractHandcrafted } from './features.js';
import { CFG } from './data.js';
import fs from 'fs';
import path from 'path';

const program = new Command();

/**
 * Create dummy dataset for smoke testing
 */
function makeDummyDataset(batchSize = 4, timeSteps = 64, handDim = 70, numClasses = 5) {
  const melData = tf.randomNormal([batchSize, CFG.nMels, timeSteps, 1]);
  const handData = tf.randomNormal([batchSize, handDim]);
  const labels = tf.randomUniform([batchSize], 0, numClasses, 'int32');
  
  return { xs: [melData, handData], ys: labels };
}

/**
 * Train command
 */
program
  .command('train')
  .description('Run smoke training with dummy data')
  .action(async () => {
    console.log('Running smoke training with dummy data...');
    
    const handDim = 70;
    const numClasses = CFG.classes.length;
    
    // Create model
    const model = buildModel(handDim, numClasses, 1e-3);
    console.log('Model created');
    model.summary();
    
    // Create dummy data
    const trainData = makeDummyDataset(4, 64, handDim, numClasses);
    const valData = makeDummyDataset(2, 64, handDim, numClasses);
    
    // Train for a few epochs
    console.log('Training for 2 epochs...');
    await model.fit(trainData.xs, trainData.ys, {
      epochs: 2,
      validationData: [valData.xs, valData.ys],
      verbose: 1
    });
    
    // Save model
    const modelDir = path.join(CFG.workDir, 'model_js');
    await model.save(`file://${modelDir}`);
    console.log(`Model saved to ${modelDir}`);
    
    // Clean up
    trainData.xs[0].dispose();
    trainData.xs[1].dispose();
    trainData.ys.dispose();
    valData.xs[0].dispose();
    valData.xs[1].dispose();
    valData.ys.dispose();
  });

/**
 * XAI demo command
 */
program
  .command('xai')
  .description('Run Grad-CAM demo on random batch')
  .action(async () => {
    console.log('Running Grad-CAM demo on random batch...');
    
    const handDim = 70;
    const numClasses = CFG.classes.length;
    
    // Create model
    const model = buildModel(handDim, numClasses);
    console.log('Model created');
    
    // Create dummy data
    const data = makeDummyDataset(4, 64, handDim, numClasses);
    const [melBatch, handBatch] = data.xs;
    
    // Get predictions
    const preds = model.predict([melBatch, handBatch]);
    console.log('Predictions shape:', preds.shape);
    
    // Run Grad-CAM
    const gcOut = gradCAM(model, melBatch, handBatch);
    console.log('CAM shape:', gcOut.cams.shape);
    
    // Save a sample (would need proper image library for real visualization)
    console.log('Grad-CAM demo completed');
    console.log('Note: For actual visualization, integrate with a plotting library');
    
    // Clean up
    melBatch.dispose();
    handBatch.dispose();
    data.ys.dispose();
    preds.dispose();
    gcOut.cams.dispose();
    gcOut.classIdx.dispose();
    gcOut.preds.dispose();
  });

/**
 * Inference command
 */
program
  .command('infer')
  .description('Run inference on a WAV file')
  .requiredOption('--wav <path>', 'Path to WAV file')
  .action(async (options) => {
    const wavPath = options.wav;
    
    if (!fs.existsSync(wavPath)) {
      console.error(`File not found: ${wavPath}`);
      process.exit(1);
    }
    
    console.log(`Loading audio from ${wavPath}...`);
    
    // Load and preprocess audio
    const y = loadWav(wavPath, CFG.sampleRate);
    const yPadded = padOrTrim(y);
    
    console.log(`Audio loaded: ${yPadded.length} samples`);
    
    // Extract features
    console.log('Extracting features...');
    const melTensor = extractMel(yPadded, CFG.sampleRate, CFG.nMels);
    const hand = extractHandcrafted(yPadded, CFG.sampleRate);
    
    // Reshape for model input
    const melInput = melTensor.expandDims(0).expandDims(-1);
    const handInput = tf.tensor2d([Array.from(hand)]);
    
    // Load model
    const modelPath = path.join(CFG.workDir, 'model_js');
    
    if (!fs.existsSync(path.join(modelPath, 'model.json'))) {
      console.error(`No trained model found at ${modelPath}`);
      console.log('Please run training first or use the Python-trained model');
      process.exit(1);
    }
    
    console.log(`Loading model from ${modelPath}...`);
    const model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
    
    // Predict
    console.log('Running inference...');
    const pred = model.predict([melInput, handInput]);
    const predArray = await pred.array();
    const classIdx = pred.argMax(-1).dataSync()[0];
    const prob = predArray[0][classIdx];
    
    console.log('\n--- Prediction Results ---');
    console.log(`Predicted: ${CFG.classes[classIdx]}`);
    console.log(`Confidence: ${(prob * 100).toFixed(2)}%`);
    console.log('\nAll class probabilities:');
    CFG.classes.forEach((cls, idx) => {
      console.log(`  ${cls}: ${(predArray[0][idx] * 100).toFixed(2)}%`);
    });
    
    // Clean up
    melTensor.dispose();
    melInput.dispose();
    handInput.dispose();
    pred.dispose();
  });

// Parse arguments
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}
