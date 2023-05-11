import * as chai from "chai";
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);
import * as imag from "./imag";
import * as linspace from "./linspace";
import * as buffer from "./buffer";
import * as clone from "./clone";
import * as complex from "./complex";
import * as diag from "./diag";
import * as eye from "./eye";
import * as fill from "./fill";
import * as oneHot from "./oneHot";
import * as scalar from "./scalar";
import * as tensor from "./tensor";
import * as tensorNd from "./tensorNd";
import * as ones from "./ones";
import * as onesLike from "./onesLike";
import * as print from "./print";
import * as range from "./range";
import * as real from "./real";
import * as truncatedNormal from "./truncatedNormal";
import * as variable from "./variable";
import * as zeros from "./zeros";
import * as zerosLike from "./zerosLike";

/* ---- Creating Tensors ---- */
describe("** CREATION **", () => {
  /* ---- tf.linspace(start, stop, num)---- *
  Return an evenly spaced sequence of numbers (including decimals) over the given interval.
  */
  describe("tf.linspace(start, stop, num):", linspace.run.bind(this));

  /* ---- tf.buffer(start, stop, num)---- *
  Creates an empty tf.TensorBuffer with the specified shape and dtype.
  */
  describe("tf.buffer(): ", buffer.run.bind(this));

  /* ---- tf.clone(tensor)---- *
  Creates a new tensor with the same values and shape as the specified tensor.
  */
  describe("tf.clone(): ", clone.run.bind(this));

  /* ---- tf.complex(real, imag)---- *
  Converts two real numbers to a complex number.
  */
  describe("tf.complex(): ", complex.run.bind(this));

  /* ---- tf.diag ----
    Creates new tensor from argument tensor
      -- each row is a copy of the argument tensor
      -- expect all values are 0 except for the diagonal
  */
  describe("tf.diag(tensor): ", diag.run.bind(this));

  /* ---- tf.eye ----
    Create an identity matrix
  */
  describe(
    "tf.eye(numRows, numColumns?, batchShape?, dtype?): ",
    eye.run.bind(this)
  );

  /* ---- tf.fill ----
    Creates a tf.Tensor filled with a scalar value.
  */
  describe("tf.fill(shape, value, dtype?): ", fill.run.bind(this));

  /* ---- tf.imag ---- *
    Returns the imaginary part of a complex (or real) tensor.
  */
  describe("tf.imag(complexTensor): ", imag.run.bind(this));

  /* ---- oneHot ---- *
    1. The values represented by indices take on onValue (1) while others are offvalue (0)
    2. rank(output) = rank(indices) + 1
    3. Create a number of "rows" equal to indices.length
    4. For each "row": set column value to 1 if column index is in indices, else 0
  */
  describe(
    "tf.oneHot(indices, depth, onValue?, offValue?, dtype?): ",
    oneHot.run.bind(this)
  );

  /* ---- scalar ---- *
    Creates rank-0 tf.Tensor (scalar) with the provided value and dtype.
  */
  describe("tf.scalar(): ", scalar.run.bind(this));

  /* ---- tensor ---- *
    Creates a tf.Tensor with the provided values, shape and dtype.
  */
  describe("tf.tensor(): ", tensor.run.bind(this));

  /* ---- tensor 1D-6D ---- *
   */
  describe("tf.tensorNd() : n∈{1, 2, 3, 4, 5, 6}", tensorNd.run.bind(this));

  /* ---- ones ---- *
    Creates a tf.Tensor with all elements set to 1.
   */
  describe("tf.ones(shape, dytpe?)", ones.run.bind(this));

  /* ---- onesLike ---- *
    Creates a tf.Tensor with all elements set to 1 with the same shape as the given tensor.
   */
  describe("tf.oneLike(tensor)", onesLike.run.bind(this));

  /* ---- print ---- *
    Prints information about the tf.Tensor including its data
   */
  describe("tf.print(t, verbose?)", print.run.bind(this));

  /* ---- range ---- *
    Creates a new tf.Tensor1D filled with the numbers in the range provided.
    The tensor is a half-open interval meaning it includes start, but excludes stop. Decrementing ranges and negative step values are also supported.
   */
  describe("tf.range(start, stop, step?, dtype?)", range.run.bind(this));

  /* ---- real ---- *
    Returns the real part of a complex (or real) tensor.
    Given a tensor input, this operation returns a tensor of type float that is the real part of each element in input considered as a complex number.
    If the input is real, it simply makes a clone.
  */
  describe("tf.real(tensor)", real.run.bind(this));

  /* ---- truncatedNormal ---- *
    Creates a tf.Tensor with values sampled from a truncated normal distribution.
  */
  describe(
    "tf.truncatedNormal(shape, mean?, stdDev?, dtype?, seed?)",
    truncatedNormal.run.bind(this)
  );
  /* ---- variable ---- *
    Creates a new variable with the provided initial value.
  */
  describe(
    "tf.variable(initialValue, trainable?, name?, dtype?)",
    variable.run.bind(this)
  );

  /* ---- zeros ---- *
    Creates a tf.Tensor with all elements set to 0.
  */
  describe("tf.zeros(shape, dtype?)", zeros.run.bind(this));

  /* ---- zerosLike ---- *
    Creates a tf.Tensor with all elements set to 0 with the same shape as the given tensor.
  */
  describe("tf.zerosLike(tensor?)", zerosLike.run.bind(this));
});
