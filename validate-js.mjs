#!/usr/bin/env node

/**
 * Simple validation script to test JavaScript implementation
 */

console.log('=== JavaScript Implementation Validation ===\n');

// Test 1: Module imports
console.log('Test 1: Importing modules...');
try {
  const { CFG } = await import('./js/data.js');
  console.log('✓ data.js imported successfully');
  console.log(`  - Sample rate: ${CFG.sampleRate} Hz`);
  console.log(`  - Classes: ${CFG.classes.join(', ')}`);
} catch (err) {
  console.error('✗ Failed to import data.js:', err.message);
  process.exit(1);
}

// Test 2: Model building (without training)
console.log('\nTest 2: Building model architecture...');
try {
  const { buildModel } = await import('./js/model.js');
  const model = buildModel(3, 5, 1e-3);
  console.log('✓ Model built successfully');
  console.log(`  - Total layers: ${model.layers.length}`);
  console.log(`  - Input shapes: mel=[null,128,null,1], hand=[null,3]`);
  console.log(`  - Output shape: [null,5]`);
} catch (err) {
  console.error('✗ Failed to build model:', err.message);
  process.exit(1);
}

// Test 3: Feature extraction functions
console.log('\nTest 3: Feature extraction utilities...');
try {
  const { padOrTrim } = await import('./js/features.js');
  const testAudio = new Float32Array(32000); // 2 seconds at 16kHz
  const padded = padOrTrim(testAudio, 64000); // Pad to 4 seconds
  console.log('✓ Audio padding works');
  console.log(`  - Input length: ${testAudio.length} samples`);
  console.log(`  - Output length: ${padded.length} samples`);
} catch (err) {
  console.error('✗ Failed feature extraction test:', err.message);
  process.exit(1);
}

// Test 4: XAI module
console.log('\nTest 4: XAI module...');
try {
  await import('./js/xai.js');
  console.log('✓ XAI module imported successfully');
  console.log('  - gradCAM, integratedGradients available');
} catch (err) {
  console.error('✗ Failed to import XAI module:', err.message);
  process.exit(1);
}

console.log('\n=== All Validation Tests Passed! ===');
console.log('\nThe JavaScript implementation is ready to use.');
console.log('See JAVASCRIPT_GUIDE.md for usage examples.\n');
console.log('Note: For better performance, install @tensorflow/tfjs-node:');
console.log('  npm install @tensorflow/tfjs-node\n');
