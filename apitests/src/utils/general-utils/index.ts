import * as tf from "@tensorflow/tfjs-core";

export type BasicType = number | string | boolean;

/* --- Loops over all values in tensor */
export function average(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b) / arr.length;
}

export default {
  average,
};
