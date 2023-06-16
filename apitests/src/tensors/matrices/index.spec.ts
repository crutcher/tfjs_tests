import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as einsum from "./einsum";

/* ---- Tensors - matrices:---- */

describe("**** TENSORS: Matrices ****", () => {
  /* ---- tf.booleanMaskAsync (tensor, mask, axis?)---- *
    Apply boolean mask to tensor.
  */
  describe("tf.einsum (equation, ...tensors) : matrices", einsum.run);
});
