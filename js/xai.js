/**
 * Explainable AI (XAI) methods: Grad-CAM, Integrated Gradients
 */

// Try to use tfjs-node for better performance, fall back to pure tfjs
let tf;
try {
  tf = await import('@tensorflow/tfjs-node');
} catch {
  tf = await import('@tensorflow/tfjs');
}

/**
 * Compute Grad-CAM for the model
 * @param {tf.LayersModel} model - Trained model
 * @param {tf.Tensor} melBatch - Mel-spectrogram batch
 * @param {tf.Tensor} handBatch - Handcrafted features batch
 * @param {string} convLayerName - Name of convolutional layer (default: 'conv_tail_identity')
 * @returns {Object} - {cams, classIdx, preds}
 */
export function gradCAM(model, melBatch, handBatch, convLayerName = 'conv_tail_identity') {
  return tf.tidy(() => {
    // Get the conv layer and output
    const convLayer = model.getLayer(convLayerName);
    
    // Create a model that outputs both conv features and predictions
    const gradModel = tf.model({
      inputs: model.inputs,
      outputs: [convLayer.output, model.output]
    });
    
    // Forward pass with gradient tape
    const grads = tf.variableGrads(() => {
      const [convOut, preds] = gradModel.predict([melBatch, handBatch]);
      
      // Get predicted classes
      const classIdx = tf.argMax(preds, 1);
      
      // Create one-hot encoding
      const oneHot = tf.oneHot(classIdx, preds.shape[1]);
      
      // Loss = sum of predicted class scores
      const loss = tf.sum(tf.mul(preds, oneHot));
      
      return loss;
    });
    
    // Get predictions
    const [convOut, preds] = gradModel.predict([melBatch, handBatch]);
    const classIdx = tf.argMax(preds, 1);
    
    // Compute gradients of loss w.r.t. conv output
    const convGrads = grads.grads[convOut.name] || grads.grads[Object.keys(grads.grads)[0]];
    
    // Global average pooling on gradients
    const weights = tf.mean(convGrads, [1, 2]);
    
    // Weighted sum of feature maps
    const cam = tf.sum(
      tf.mul(
        tf.expandDims(tf.expandDims(weights, 1), 1),
        convOut
      ),
      3
    );
    
    // Apply ReLU
    const camRelu = tf.relu(cam);
    
    // Normalize
    const camNorm = tf.div(
      camRelu,
      tf.add(tf.max(camRelu, [1, 2], true), 1e-8)
    );
    
    return {
      cams: camNorm,
      classIdx: classIdx,
      preds: preds
    };
  });
}

/**
 * Compute Integrated Gradients for the model
 * @param {tf.LayersModel} model - Trained model
 * @param {tf.Tensor} mel - Mel-spectrogram (single sample)
 * @param {tf.Tensor} hand - Handcrafted features (single sample)
 * @param {number} targetClass - Target class index
 * @param {number} steps - Number of integration steps (default: 50)
 * @returns {tf.Tensor} - Attribution map
 */
export function integratedGradients(model, mel, hand, targetClass, steps = 50) {
  return tf.tidy(() => {
    // Create baseline (zeros)
    const baseline = tf.zeros(mel.shape);
    
    // Generate alphas
    const alphas = tf.linspace(0.0, 1.0, steps);
    const alphasArray = Array.from(alphas.dataSync());
    
    // Accumulate gradients
    let integrated = tf.zeros(mel.shape);
    
    for (const alpha of alphasArray) {
      const x = tf.add(baseline, tf.mul(alpha, tf.sub(mel, baseline)));
      
      const grads = tf.variableGrads(() => {
        const preds = model.predict([x, hand]);
        const loss = preds.gather([targetClass], 1);
        return tf.sum(loss);
      });
      
      // Get gradient w.r.t. mel input
      const melGrad = grads.grads[x.name] || grads.grads[Object.keys(grads.grads)[0]];
      integrated = tf.add(integrated, melGrad);
    }
    
    // Average gradients
    const avgGrads = tf.div(integrated, steps);
    
    // Multiply by (input - baseline)
    const attribution = tf.mul(tf.sub(mel, baseline), avgGrads);
    
    alphas.dispose();
    
    return attribution;
  });
}

/**
 * Visualize CAM on mel-spectrogram
 * Note: This is a placeholder - in browser/node you'd use a canvas or image library
 * @param {Array} mel - Mel-spectrogram data
 * @param {Array} cam - CAM data
 * @param {string} title - Plot title
 */
export function visualizeCAMOnMel(mel, cam, title = 'Grad-CAM') {
  console.log(`Visualization: ${title}`);
  console.log('Mel shape:', mel.length, 'x', mel[0]?.length || 0);
  console.log('CAM shape:', cam.length, 'x', cam[0]?.length || 0);
  console.log('Note: Use a plotting library like plotly.js or chart.js for actual visualization');
}

/**
 * SHAP explanation for handcrafted features
 * Note: Full SHAP implementation requires the SHAP library
 * This is a simplified placeholder
 * @param {tf.LayersModel} model - Trained model
 * @param {tf.Tensor} mel - Mel-spectrogram
 * @param {tf.Tensor} hand - Handcrafted features
 * @param {number} nSamples - Number of samples for approximation
 * @returns {Object} - Placeholder SHAP values
 */
export function shapExplain(model, mel, hand, nSamples = 50) {
  console.warn('SHAP explanation is not fully implemented in JavaScript version.');
  console.warn('For full SHAP support, consider using the Python version or a JS SHAP library.');
  
  // Return a simple placeholder structure
  return {
    values: [null, tf.zeros(hand.shape)],
    error: 'SHAP not fully implemented in JS version'
  };
}
