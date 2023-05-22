import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as batchToSpaceND from "./batchToSpaceND";

/* ---- Tensors: This section describes some common Tensor transformations for reshaping and type-casting.---- */
describe("**** TENSORS: Transformation Methods ****", () => {
  /* ---- tf.batchToSpaceND(x, blockShape, crops)---- *
    This operation reshapes the "batch" dimension 0 into M + 1 dimensions of shape blockShape + [batch], interleaves these blocks back into the grid defined by the spatial dimensions [1, ..., M], to obtain a result with the same rank as the input.
    The spatial dimensions of this intermediate result are then optionally cropped according to crops to produce the output.
  */
  describe(
    "tf.batchToSpaceND(x, blockShape, crops) : transformation",
    batchToSpaceND.run
  );
});
