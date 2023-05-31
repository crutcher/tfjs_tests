import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as booleanMaskAsync from "./booleanMaskAsync";
import * as concat from "./concat";
import * as gather from "./gather";
import * as reverse from "./reverse";

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
});
