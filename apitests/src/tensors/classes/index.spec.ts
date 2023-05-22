import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);
import * as Tensor from "./Tensor";
import * as Variable from "./Variable";

/* ---- Creating Tensors: tf classes related to tensors ---- */
describe("**** TENSORS: Classes ****", () => {
  /* ---- tf.Tensor class---- *
  A tf.Tensor object represents an immutable, multidimensional array of numbers that has a shape and a data type.
  */
  describe("tf.Tensor : class", Tensor.run);
  /* ---- tf.Variable class---- *
  A mutable tf.Tensor, useful for persisting state, e.g. for training.
  */
  describe("tf.Variable : class", Variable.run);
});
