import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as batchToSpaceND from "./batchToSpaceND";
import * as broadcastTo from "./broadcastTo";
import * as broadcastArgs from "./broadcastArgs";
import * as cast from "./cast";
import * as depthToSpace from "./depthToSpace";

/* ---- Tensors - transformations: This section describes some common Tensor transformations for reshaping and type-casting.---- */
describe("**** TENSORS: Transformation Methods ****", () => {
  /* ---- tf.batchToSpaceND(x, blockShape, crops)---- *
    This operation reshapes the "batch" dimension 0 into M + 1 dimensions of shape blockShape + [batch], interleaves these blocks back into the grid defined by the spatial dimensions [1, ..., M], to obtain a result with the same rank as the input.
    The spatial dimensions of this intermediate result are then optionally cropped according to crops to produce the output.
  */
  describe(
    "tf.batchToSpaceND(x, blockShape, crops) : transformation",
    batchToSpaceND.run
  );

  /* ---- tf.broadcastTo (x, shape) ---- *
    Broadcast an array to a compatible shape NumPy-style.
    The tensor's shape is compared to the broadcast shape from end to beginning.
    Ones are prepended to the tensor's shape until it has the same length as the broadcast shape.
    If input.shape[i]==shape[i], the (i+1)-th axis is already broadcast-compatible.
    If input.shape[i]==1 and shape[i]==N, then the input tensor is tiled N times along that axis (using tf.tile).
  */
  describe("tf.broadcastTo (x, shape) : transformation", broadcastTo.run);

  /* ---- tf.broadcastTo (x, shape) ---- *
    Return the shape of s0 op s1 with broadcast.
    compute r0, the broadcasted shape as a tensor. s0, s1 and r0 are all integer vectors.
    This function returns the shape of the result of an operation between two tensors of size s0 and s1 performed with broadcast.
  */
  describe("tf.broadcastArgs (s0, s1) : transformation", broadcastArgs.run);

  /* ---- tf.broadcastTo (x, shape) ---- *
    Casts a tf.Tensor to a new dtype.
  */
  describe("tf.cast (x, dtype) : transformation", cast.run);

  /* ---- tf.broadcastTo (x, shape) ---- *
    Rearranges data from depth into blocks of spatial data.
    More specifically, this op outputs a copy of the input tensor where values from the depth dimension are moved
    in spatial blocks to the height and width dimensions.
    The attr blockSize indicates the input block size and how the data is moved.
  */
  describe(
    "tf.depthToSpace (x, blockSize, dataFormat?) : transformation",
    depthToSpace.run
  );
});
