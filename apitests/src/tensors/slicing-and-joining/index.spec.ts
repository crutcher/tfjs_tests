import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as booleanMaskAsync from "./booleanMaskAsync";
import * as concat from "./concat";
import * as gather from "./gather";
import * as reverse from "./reverse";
import * as slice from "./slice";
import * as split from "./split";
import * as stack from "./stack";
import * as tile from "./tile";
import * as unstack from "./unstack";

/* ---- Tensors - slicing and joining:----
  TensorFlow.js provides several operations to slice or extract parts of a tensor,
  or join multiple tensors together.
*/

describe("**** TENSORS: Slicing and Joining ****", () => {
  /* ---- tf.booleanMaskAsync (tensor, mask, axis?)---- *
    Apply boolean mask to tensor.
  */
  describe(
    "tf.booleanMaskAsync (tensor, mask, axis?) : slicing and joining",
    booleanMaskAsync.run
  );

  /* ---- tf.concat (tensors, axis?)---- *
    Concatenates a list of tf.Tensors along a given axis.
    The tensors ranks and types must match, and their sizes must match in all dimensions except axis.
    Also available are stricter rank-specific methods that assert that tensors are of the given rank:
    tf.concat1d, tf.concat2d, tf.concat3d, tf.concat4d
  */
  describe("tf.concat (tensors, axis?) : slicing and joining", concat.run);

  /* ---- tf.gather (x, indices, axis?, batchDims?)---- *
    Gather slices from tensor x's axis axis according to indices.
  */
  describe(
    "tf.gather (x, indices, axis?, batchDims?) : slicing and joining",
    gather.run
  );

  /* ---- tf.reverse (x, axis?)---- *
    Reverses a tf.Tensor along a specified axis.
    Also available are stricter rank-specific methods that assert that x is of the given rank:
    tf.reverse1d, tf.reverse2d, tf.reverse3d, tf.reverse4d
  */
  describe("tf.reverse (x, axis?) : slicing and joining", reverse.run);

  /* ---- tf.slice (x, begin, size?)---- *
    Extracts a slice from a tf.Tensor starting at coordinates begin and is of size size.
    Also available are stricter rank-specific methods with the same signature as this method that assert that x is of the given rank:
    tf.slice1d, tf.slice2d, tf.slice3d, tf.slice4d
  */
  describe("tf.slice (x, begin, size?) : slicing and joining", slice.run);

  /* ---- tf.split (x, numOrSizeSplits, axis?)---- *
    Splits a tf.Tensor into sub tensors.
    If numOrSizeSplits is a number, splits x along dimension axis into numOrSizeSplits smaller tensors.
    Requires that numOrSizeSplits evenly divides x.shape[axis].
    If numOrSizeSplits is a number array, splits x into numOrSizeSplits.length pieces.
    The shape of the i-th piece has the same size as x except along dimension axis where the size is numOrSizeSplits[i].
  */
  describe(
    "tf.split (x, numOrSizeSplits, axis?) : slicing and joining",
    split.run
  );

  /* ---- tf.stack (tensors, axis?)---- *
    Stacks a list of rank-R tf.Tensors into one rank-(R+1) tf.Tensor.
  */
  describe("tf.stack (tensors, axis?) : slicing and joining", stack.run);

  /* ---- tf.tile (x, reps)---- *
    Construct a tensor by repeating it the number of times given by reps.
    This operation creates a new tensor by replicating input reps times.
    The output tensor's ith dimension has input.shape[i] * reps[i] elements,
    and the values of input are replicated reps[i] times along the ith dimension.
    For example, tiling [a, b, c, d] by [2] produces [a, b, c, d, a, b, c, d].
  */
  describe("tf.tile (x, reps) : slicing and joining", tile.run);

  /* ---- tf.unstack (x, axis?)---- *
    Unstacks a tf.Tensor of rank-R into a list of rank-(R-1) tf.Tensors.
  */
  describe("tf.unstack (x, axis?)", unstack.run);
});
