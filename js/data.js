/**
 * Configuration and data handling utilities
 */

export const CFG = {
  sampleRate: 16000,
  nMels: 128,
  workDir: 'work_tf',
  classes: ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia'],
  testSize: 0.15,
  valSize: 0.15,
  seed: 42,
  // training defaults
  epochs: 10,
  earlyStopPat: 6,
};

/**
 * Split dataset into train/val/test sets
 * @param {Array} dataArray - Array of data objects with label_id
 * @returns {Object} - {train, val, test}
 */
export function prepareSplits(dataArray) {
  // Simple split logic for demonstration
  // In production, use proper stratified splitting
  const shuffled = [...dataArray].sort(() => Math.random() - 0.5);
  const testCount = Math.floor(shuffled.length * CFG.testSize);
  const valCount = Math.floor(shuffled.length * CFG.valSize);
  
  const test = shuffled.slice(0, testCount);
  const val = shuffled.slice(testCount, testCount + valCount);
  const train = shuffled.slice(testCount + valCount);
  
  return { train, val, test };
}

export default CFG;
