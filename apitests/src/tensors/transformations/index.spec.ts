import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as batchToSpaceND from "./batchToSpaceND";
import * as broadcastTo from "./broadcastTo";

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
});
