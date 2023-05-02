import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../load-tf";

let tf: loader.TFModule;

export function run() {
  // CONSTANTS
  const SHAPES_RESULTS = [
    {
      shape: [2, 3],
      result: [
        [1, 1, 1],
        [1, 1, 1],
      ],
    },
    {
      shape: [3, 1],
      result: [[1], [1], [1]],
    },
    {
      shape: [2, 2, 3],
      result: [
        [
          [1, 1, 1],
          [1, 1, 1],
        ],

        [
          [1, 1, 1],
          [1, 1, 1],
        ],
      ],
    },
  ];
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  it("  -- default options", () => {
    SHAPES_RESULTS.forEach(({ shape, result }) => {
      const t: tfTypes.Tensor = tf.ones(shape);
      expect(t).to.lookLike(result);
      expect(_isAllOnes(t)).to.be.true;
    });
  });
  it("  -- dtypes", () => {
    const { shape } = SHAPES_RESULTS[0];
    const t: tfTypes.Tensor = tf.ones(shape, "int32");
    expect(t).to.haveDtype("int32");
  });
}

function _isAllOnes(t: tfTypes.Tensor): boolean {
  const result = true;
  const unpacked = tf.unstack(tf.reshape(t, [-1]));
  for (const el of unpacked) {
    const elAsArray = el.dataSync();
    if (elAsArray.length !== 1 || elAsArray[0] !== 1) return false;
  }
  return result;
}
