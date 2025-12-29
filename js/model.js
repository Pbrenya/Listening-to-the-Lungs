/**
 * Hybrid CNN-BiLSTM-Attention model for lung disease classification
 */

// Try to use tfjs-node for better performance, fall back to pure tfjs
let tf;
try {
  tf = await import('@tensorflow/tfjs-node');
} catch {
  tf = await import('@tensorflow/tfjs');
}

/**
 * Custom Additive Attention Layer
 */
class AdditiveAttention extends tf.layers.Layer {
  constructor(dModel, config = {}) {
    super(config);
    this.dModel = dModel;
    this.w = null;
    this.v = null;
  }

  build(inputShape) {
    this.w = this.addWeight(
      'attention_w',
      [inputShape[inputShape.length - 1], this.dModel],
      'float32',
      tf.initializers.glorotUniform()
    );
    this.v = this.addWeight(
      'attention_v',
      [this.dModel, 1],
      'float32',
      tf.initializers.glorotUniform()
    );
    super.build(inputShape);
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = inputs[0];
      // s = tanh(x * w)
      const s = tf.tanh(tf.matMul(x, this.w.read()));
      // scores = s * v
      const scores = tf.matMul(s, this.v.read());
      // attention weights
      const a = tf.softmax(scores, 1);
      // context vector
      const ctx = tf.sum(tf.mul(x, a), 1);
      return ctx;
    });
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], inputShape[inputShape.length - 1]];
  }

  getConfig() {
    const config = super.getConfig();
    return { ...config, dModel: this.dModel };
  }

  static get className() {
    return 'AdditiveAttention';
  }
}

// Register custom layer
tf.serialization.registerClass(AdditiveAttention);

/**
 * Create a convolutional block
 * @param {tf.SymbolicTensor} x - Input tensor
 * @param {number} filters - Number of filters
 * @returns {tf.SymbolicTensor} - Output tensor
 */
function convBlock(x, filters) {
  let y = tf.layers.conv2d({
    filters: filters,
    kernelSize: [3, 3],
    padding: 'same',
  }).apply(x);
  
  y = tf.layers.batchNormalization().apply(y);
  y = tf.layers.reLU().apply(y);
  
  y = tf.layers.conv2d({
    filters: filters,
    kernelSize: [3, 3],
    padding: 'same',
  }).apply(y);
  
  y = tf.layers.batchNormalization().apply(y);
  y = tf.layers.reLU().apply(y);
  
  y = tf.layers.maxPooling2d({
    poolSize: [2, 2],
  }).apply(y);
  
  y = tf.layers.dropout({ rate: 0.2 }).apply(y);
  
  return y;
}

/**
 * Build the hybrid CNN-BiLSTM-Attention model
 * @param {number} handDim - Dimension of handcrafted features
 * @param {number} numClasses - Number of output classes
 * @param {number} lr - Learning rate (default: 3e-4)
 * @returns {tf.LayersModel} - Compiled model
 */
export function buildModel(handDim, numClasses, lr = 3e-4) {
  // Mel-spectrogram input: (nMels, time, 1)
  const melInput = tf.input({ shape: [128, null, 1], name: 'mel' });
  
  // Handcrafted features input
  const handInput = tf.input({ shape: [handDim], name: 'handcrafted' });
  
  // Mel-spectrogram branch
  let x = convBlock(melInput, 32);
  x = convBlock(x, 64);
  x = convBlock(x, 128);
  
  // Identity layer for Grad-CAM (last conv output)
  // Note: Lambda is not available in pure tfjs, using activation with 'linear'
  const convTailIdentity = tf.layers.activation({ 
    activation: 'linear',
    name: 'conv_tail_identity'
  }).apply(x);
  x = convTailIdentity;
  
  // Reshape for LSTM: permute to (batch, time, channels)
  // Current shape: (batch, mels', time', channels)
  // We need: (batch, time', features)
  x = tf.layers.permute({ dims: [2, 1, 3] }).apply(x);
  
  // Flatten spatial dimensions for each time step
  x = tf.layers.timeDistributed({
    layer: tf.layers.flatten()
  }).apply(x);
  
  // Bidirectional LSTM
  x = tf.layers.bidirectional({
    layer: tf.layers.lstm({ units: 128, returnSequences: true }),
    mergeMode: 'concat'
  }).apply(x);
  
  // Additive Attention
  const attentionLayer = new AdditiveAttention(256, { name: 'temporal_attention' });
  const ctx = attentionLayer.apply(x);
  
  let melEmb = tf.layers.dropout({ rate: 0.3 }).apply(ctx);
  
  // Handcrafted features branch
  let h = tf.layers.dense({ units: 256 }).apply(handInput);
  h = tf.layers.batchNormalization().apply(h);
  h = tf.layers.reLU().apply(h);
  h = tf.layers.dropout({ rate: 0.3 }).apply(h);
  h = tf.layers.dense({ units: 128, activation: 'relu' }).apply(h);
  
  // Fusion
  let z = tf.layers.concatenate().apply([melEmb, h]);
  z = tf.layers.dense({ units: 256, activation: 'relu' }).apply(z);
  z = tf.layers.dropout({ rate: 0.3 }).apply(z);
  
  // Output layer
  const output = tf.layers.dense({
    units: numClasses,
    activation: 'softmax',
    dtype: 'float32'
  }).apply(z);
  
  // Create and compile model
  const model = tf.model({
    inputs: [melInput, handInput],
    outputs: output
  });
  
  const optimizer = tf.train.adam(lr);
  
  model.compile({
    optimizer: optimizer,
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

/**
 * Train the model
 * @param {tf.LayersModel} model - Model to train
 * @param {Object} trainData - Training data {xs: [melTensor, handTensor], ys: labelTensor}
 * @param {Object} valData - Validation data
 * @param {Object} cfg - Configuration object
 * @param {string} workDir - Directory to save model
 * @returns {Promise<Object>} - Training history
 */
export async function trainModel(model, trainData, valData, cfg, workDir = 'work_tf') {
  const callbacks = [
    tf.callbacks.earlyStopping({
      monitor: 'val_acc',
      patience: cfg.earlyStopPat,
      restoreBestWeights: true
    })
  ];
  
  const history = await model.fit(trainData.xs, trainData.ys, {
    epochs: cfg.epochs,
    validationData: [valData.xs, valData.ys],
    callbacks: callbacks,
    verbose: 1
  });
  
  // Save model
  await model.save(`file://${workDir}/model`);
  console.log(`Model saved to ${workDir}/model`);
  
  return history;
}

export { AdditiveAttention, tf };
