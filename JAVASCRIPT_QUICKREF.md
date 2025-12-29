# Quick Reference: JavaScript Implementation

## Installation

```bash
npm install
```

## Basic Usage

### 1. Run Inference on Audio File

```bash
node js/run.js infer --wav path/to/audio.wav
```

**Example Output:**
```
--- Prediction Results ---
Predicted: Asthma
Confidence: 87.32%

All class probabilities:
  Bronchial: 3.45%
  asthma: 87.32%
  copd: 2.11%
  healthy: 1.02%
  pneumonia: 6.10%
```

### 2. Run Training Demo

```bash
npm run train
# or
node js/run.js train
```

### 3. Run XAI Demo

```bash
npm run xai
# or
node js/run.js xai
```

### 4. Run Example Script

```bash
node js/examples/inference.js path/to/audio.wav
```

### 5. Validate Installation

```bash
node validate-js.mjs
```

## Programmatic Usage

### Simple Inference

```javascript
import { loadWav, padOrTrim, extractMel, extractHandcrafted } from './js/features.js';
import { CFG } from './js/data.js';
import { tf } from './js/features.js';

// Load audio
const audio = loadWav('audio.wav', 16000);
const processed = padOrTrim(audio);

// Extract features
const mel = extractMel(processed, 16000, 128);
const hand = extractHandcrafted(processed, 16000);

// Prepare inputs
const melInput = mel.expandDims(0).expandDims(-1);
const handInput = tf.tensor2d([Array.from(hand)]);

// Load model and predict
const model = await tf.loadLayersModel('file://work_tf/model_js/model.json');
const prediction = model.predict([melInput, handInput]);

// Get result
const probs = await prediction.array();
const classIdx = prediction.argMax(-1).dataSync()[0];
console.log(`Predicted: ${CFG.classes[classIdx]}`);
```

### Build Model

```javascript
import { buildModel } from './js/model.js';

const model = buildModel(
  3,      // handcrafted feature dimension
  5,      // number of classes
  1e-3    // learning rate
);

model.summary();
```

### Explainability

```javascript
import { gradCAM, integratedGradients } from './js/xai.js';

// Grad-CAM
const result = gradCAM(model, melBatch, handBatch);
console.log('CAM shape:', result.cams.shape);

// Integrated Gradients
const attribution = integratedGradients(
  model,
  mel,
  hand,
  targetClass,
  50  // steps
);
```

## File Structure

```
js/
├── data.js              # Configuration (CFG object)
├── features.js          # Audio processing
├── model.js            # Model architecture
├── xai.js              # Explainability methods
├── run.js              # CLI entrypoint
└── examples/
    └── inference.js    # Example script
```

## Configuration

Edit `js/data.js` to change:

```javascript
export const CFG = {
  sampleRate: 16000,    // Audio sample rate
  nMels: 128,           // Mel bins
  workDir: 'work_tf',   // Model directory
  classes: ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia'],
  testSize: 0.15,       // Test split
  valSize: 0.15,        // Validation split
  seed: 42,             // Random seed
  epochs: 10,           // Training epochs
  earlyStopPat: 6       // Early stopping patience
};
```

## Available Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `train` | Run smoke training | `npm run train` |
| `infer` | Run inference | `node js/run.js infer --wav file.wav` |
| `xai` | Run XAI demo | `npm run xai` |
| `--help` | Show help | `node js/run.js --help` |

## NPM Scripts

```json
{
  "train": "node js/run.js train",
  "infer": "node js/run.js infer",
  "xai": "node js/run.js xai",
  "test": "node js/examples/inference.js"
}
```

## Performance Tips

1. **Install native bindings** for 3-5x speedup:
   ```bash
   npm install @tensorflow/tfjs-node
   ```

2. **Batch multiple samples** for efficiency:
   ```javascript
   const batch = tf.stack([mel1, mel2, mel3]);
   const predictions = model.predict([batch, handBatch]);
   ```

3. **Dispose tensors** to free memory:
   ```javascript
   mel.dispose();
   prediction.dispose();
   ```

4. **Run with more memory**:
   ```bash
   node --max-old-space-size=4096 js/run.js infer --wav audio.wav
   ```

## Troubleshooting

### Module not found
```bash
rm -rf node_modules package-lock.json
npm install
```

### Audio file not loading
```bash
# Convert to 16kHz mono WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Out of memory
```javascript
// Dispose tensors after use
tensor.dispose();

// Force garbage collection
if (global.gc) global.gc();
```

## Documentation

- **[JAVASCRIPT_GUIDE.md](JAVASCRIPT_GUIDE.md)** - Comprehensive guide (25KB)
- **[CONVERSION_SUMMARY.md](CONVERSION_SUMMARY.md)** - Conversion details
- **[README.md](README.md)** - Project overview

## API Quick Reference

### Features Module

- `loadWav(path, sr)` - Load audio file
- `padOrTrim(audio, length)` - Pad/trim to length
- `extractMel(audio, sr, nMels)` - Extract mel-spectrogram
- `extractHandcrafted(audio, sr)` - Extract acoustic features
- `extractMfcc(audio, sr, nMfcc)` - Extract MFCC

### Model Module

- `buildModel(handDim, numClasses, lr)` - Build model
- `trainModel(model, trainData, valData, cfg, workDir)` - Train model

### XAI Module

- `gradCAM(model, mel, hand, layerName)` - Grad-CAM visualization
- `integratedGradients(model, mel, hand, class, steps)` - IG attribution

### Data Module

- `CFG` - Configuration object
- `prepareSplits(data)` - Split dataset

## Requirements

- Node.js >= 18.0.0
- npm or yarn
- Audio files in WAV format (16kHz mono recommended)

## Next Steps

1. ✅ Installation complete
2. ✅ Run validation: `node validate-js.mjs`
3. ✅ Try inference: `node js/run.js infer --wav audio.wav`
4. 📖 Read full guide: [JAVASCRIPT_GUIDE.md](JAVASCRIPT_GUIDE.md)
5. 🚀 Deploy to your application

## Support

- See [JAVASCRIPT_GUIDE.md](JAVASCRIPT_GUIDE.md) for detailed documentation
- Check [CONVERSION_SUMMARY.md](CONVERSION_SUMMARY.md) for technical details
- Open an issue on GitHub for questions

---

**Happy coding!** 🎉
