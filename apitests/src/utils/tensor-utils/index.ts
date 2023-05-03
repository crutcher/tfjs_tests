import * as tf from "@tensorflow/tfjs-core";

/* --- Returns true tensor is filled with a single value, n */
export function isFilledWith(
  n: number | string | boolean,
  t: tf.Tensor
): boolean {
  const unpacked = tf.unstack(tf.reshape(t, [-1]));
  for (const el of unpacked) {
    const elAsArray = el.dataSync();
    if (elAsArray.length == 0) return true;
    if (elAsArray.length !== 1 || elAsArray[0] !== n) return false;
  }
  return true;
}

export const isAllOnes = isFilledWith.bind(null, 1);

export default {
  isFilledWith,
  isAllOnes,
};
