import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);
import * as Tensor from "./Tensor";
import * as Variable from "./Variable";
import * as TensorBuffer from "./TensorBuffer";

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
  /* ---- tf.TensorBuffer class---- *
    A mutable object, similar to tf.Tensor, that allows users to set values at locations before converting to an immutable tf.Tensor.
    See tf.buffer() for creating a tensor buffer.
  */
  describe("tf.TensorBuffer : class", TensorBuffer.run);
});
