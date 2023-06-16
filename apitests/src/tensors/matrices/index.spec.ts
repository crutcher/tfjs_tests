import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as einsum from "./einsum";

/* ---- Tensors - matrices:---- */

describe("**** TENSORS: Matrices ****", () => {
  /* ---- tf.einsum (equation, ...tensors)---- *
    Tensor contraction over specified indices and outer product.
    einsum allows defining Tensors by defining their element-wise computation.
    This computation is based on Einstein summation.
  */
  describe("tf.einsum (equation, ...tensors) : matrices", einsum.run);
});
