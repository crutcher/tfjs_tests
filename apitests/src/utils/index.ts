import type tfTypes from "@tensorflow/tfjs-core";
import * as tf from "@tensorflow/tfjs-core";

/* --- Returns true if all elements in a tensor are 1 */
export function isAllOnes(t: tfTypes.Tensor): boolean {
  const result = true;
  const unpacked = tf.unstack(tf.reshape(t, [-1]));
  for (const el of unpacked) {
    const elAsArray = el.dataSync();
    if (elAsArray.length !== 1 || elAsArray[0] !== 1) return false;
  }
  return result;
}
