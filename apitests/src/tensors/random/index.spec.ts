import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);

import * as multinomial from "./multinomial";
import * as rand from "./rand";
import * as randomGamma from "./randomGamma";
import * as randomNormal from "./randomNormal";
import * as randomStandardNormal from "./randomStandardNormal";
import * as randomUniform from "./randomUniform";

/* ---- Tensors - random:---- */

describe("**** TENSORS: Random ****", () => {
  /* ---- tf.multinomial (logits, numSamples, seed?, normalized?) ---- *
    Creates a tf.Tensor with values drawn from a multinomial distribution.
  */
  describe(
    "tf.multinomial (logits, numSamples, seed?, normalized?) : random",
    multinomial.run
  );

  /* ---- tf.rand (shape, randFunction, dtype?) ---- *
    Creates a tf.Tensor with values sampled from a random number generator function defined by the user.
  */
  describe("tf.rand (shape, randFunction, dtype?) : random", rand.run);

  /* ---- tf.randomGamma (shape, alpha, beta?, dtype?, seed?) ---- *
    Creates a tf.Tensor with values sampled from a gamma distribution.
  */
  describe(
    "tf.randomGamma (shape, alpha, beta?, dtype?, seed?) : random",
    randomGamma.run
  );

  /* ---- tf.randomNormal (shape, mean?, stdDev?, dtype?, seed?) ---- *
    Creates a tf.Tensor with values sampled from a gamma distribution.
  */
  describe(
    "tf.randomNormal (shape, mean?, stdDev?, dtype?, seed?) : random",
    randomNormal.run
  );

  /* ---- tf.randomStandardNormal (shape, dtype?, seed?) ---- *
    Creates a tf.Tensor with values sampled from a normal distribution.
    The generated values will have mean 0 and standard deviation 1.
  */
  describe(
    "tf.randomStandardNormal (shape, dtype?, seed?) : random",
    randomStandardNormal.run
  );

  /* ---- tf.randomUniform (shape, minval?, maxval?, dtype?, seed?) ---- *
    Creates a tf.Tensor with values sampled from a uniform distribution.
    The generated values follow a uniform distribution in the range [minval, maxval).
    The lower bound minval is included in the range, while the upper bound maxval is excluded.
  */
  describe(
    "tf.randomUniform (shape, minval?, maxval?, dtype?, seed?) : random",
    randomUniform.run
  );

  /* ---- tf.randomUniformInt (shape, minval, maxval, seed?) ---- *
    ! Method not included in tfjs !
  */
});
