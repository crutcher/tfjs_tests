import * as tf from "@tensorflow/tfjs-core";

export type BasicType = number | string | boolean;

/* --- Returns true tensor is filled with a single value, n */
export function isFilledWith(n: BasicType, t: tf.Tensor): boolean {
  const unpacked = tf.unstack(tf.reshape(t, [-1]));
  for (const el of unpacked) {
    const elAsArray = el.dataSync();
    if (elAsArray.length == 0) return true;
    if (elAsArray.length !== 1 || elAsArray[0] !== n) return false;
  }
  return true;
}

export const isAllOnes = isFilledWith.bind(null, 1);

/* --- Loops over all values in tensor */
export function forEachTensorValue(
  t: tf.Tensor,
  callback: (val: number) => void
): void {
  const unpacked = tf.unstack(tf.reshape(t, [-1]));
  for (const el of unpacked) {
    const elAsArray = el.dataSync();
    if (elAsArray.length !== 1) return;
    const item = elAsArray[0];
    // callback on item
    callback(item);
  }
}

export default {
  isFilledWith,
  isAllOnes,
  forEachTensorValue,
};
