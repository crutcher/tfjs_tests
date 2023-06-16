import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as multinomial from "./multinomial";
import * as rand from "./rand";
import * as randomGamma from "./randomGamma";

/* ---- Tensors - random:---- */

describe("**** TENSORS: Random ****", () => {
  /* ---- tf.multinomial (logits, numSamples, seed?, normalized?) ---- *
    Creates a tf.Tensor with values drawn from a multinomial distribution.
  */
  describe(
    "tf.multinomial (logits, numSamples, seed?, normalized?)",
    multinomial.run
  );

  /* ---- tf.rand (shape, randFunction, dtype?) ---- *
    Creates a tf.Tensor with values sampled from a random number generator function defined by the user.
  */
  describe("tf.rand (shape, randFunction, dtype?)", rand.run);

  /* ---- tf.randomGamma (shape, alpha, beta?, dtype?, seed?) ---- *
    Creates a tf.Tensor with values sampled from a gamma distribution.
  */
  describe(
    "tf.randomGamma (shape, alpha, beta?, dtype?, seed?)",
    randomGamma.run
  );
});
