import React from "react";
import * as tf from "@tensorflow/tfjs-node";

// See: https://js.tensorflow.org/api/latest/#tensor
test("tf.tensor(): default options", () => {
  const t: tf.Tensor<tf.Rank.R1> = tf.tensor([2, 3]);

  expect(t.dtype).toEqual("float32");
  expect(t.shape).toEqual([2]);
  expect(t.arraySync()).toEqual([2.0, 3.0]);
});

test("tf.tensor(): shapes", () => {
  const t: tf.Tensor<tf.Rank.R2> = tf.tensor([2, 3, 4, 5], [2, 2]);

  expect(t.dtype).toEqual("float32");
  expect(t.shape).toEqual([2, 2]);
  expect(t.arraySync()).toEqual([
    [2.0, 3.0],
    [4.0, 5.0],
  ]);
});

test("tf.tensor(): dtypes", () => {
  const t: tf.Tensor<tf.Rank.R2> = tf.tensor(
    [
      [2, 3],
      [4, 5],
    ],
    undefined,
    "int32"
  );

  expect(t.dtype).toEqual("int32");
  expect(t.shape).toEqual([2, 2]);
  expect(t.arraySync()).toEqual([
    [2, 3],
    [4, 5],
  ]);
});

test("tf.scalar(): basic", () => {
  const t: tf.Scalar = tf.scalar(2);

  expect(t.dtype).toEqual("float32");
  expect(t.shape).toEqual([]);
  expect(t.arraySync()).toEqual(2);
});

test("tf.scalar(): dtypes", () => {
  const t: tf.Scalar = tf.scalar(true, "bool");

  expect(t.dtype).toEqual("bool");
  expect(t.shape).toEqual([]);
  expect(t.arraySync()).toEqual(1);
});

test("tf.tensor1d(): bad values", () => {
  expect(() => tf.tensor1d([[1], [2]] as any)).toThrow(
    "requires values to be a flat/TypedArray"
  );
});

test("tf.tensor1d(): basic", () => {
  const t: tf.Tensor1D = tf.tensor1d([1, 2, 3, 4]);

  expect(t.dtype).toEqual("float32");
  expect(t.shape).toEqual([4]);
  expect(t.arraySync()).toEqual([1, 2, 3, 4]);
});

test("tf.tensor2d(): basic", () => {
  const t: tf.Tensor2D = tf.tensor2d([
    [1, 2],
    [3, 4],
  ]);
  expect(t.dtype).toEqual("float32");
  expect(t.shape).toEqual([2, 2]);
  expect(t.arraySync()).toEqual([
    [1, 2],
    [3, 4],
  ]);

  const t2: tf.Tensor2D = tf.tensor2d([1, 2, 3, 4], [2, 2]);
  expect(t2.dtype).toEqual("float32");
  expect(t2.shape).toEqual([2, 2]);
  expect(t2.arraySync()).toEqual([
    [1, 2],
    [3, 4],
  ]);
});
