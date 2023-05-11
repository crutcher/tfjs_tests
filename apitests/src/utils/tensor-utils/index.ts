import * as tf from "@tensorflow/tfjs-core";

export type BasicType = number | string | boolean;

/* --- Loops over all values in tensor */
export function forEachTensorValue<T>(
  t: tf.Tensor,
  callback: (val: number) => T
): T | void {
  const unpacked = tf.unstack(tf.reshape(t, [-1]));
  for (const el of unpacked) {
    const elAsArray = el.dataSync();
    if (elAsArray.length !== 1) return;
    const item = elAsArray[0];
    // callback on item
    return callback(item);
  }
}

export function isTrueForEach(
  t: tf.Tensor,
  callback: (val: number) => boolean
): boolean {
  forEachTensorValue(t, (val) => {
    if (!callback(val)) {
      return false;
    }
  });
  return true;
}

export function isFilledWith(t: tf.Tensor, val: number): boolean {
  forEachTensorValue(t, (item) => {
    if (item !== val) {
      return false;
    }
  });
  return true;
}

export function isAllZeros(t: tf.Tensor): boolean {
  return isTrueForEach(t, (val) => val === 0);
}

export function isAllOnes(t: tf.Tensor): boolean {
  return isTrueForEach(t, (val) => val === 1);
}

export default {
  forEachTensorValue,
  isFilledWith,
  isTrueForEach,
  isAllZeros,
  isAllOnes,
};
