import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as booleanMaskAsync from "./booleanMaskAsync";

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
});
