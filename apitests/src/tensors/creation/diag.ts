import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

let tf: loader.TFModule;

//CONSTANTS
const ARR_2D = [
  [
    [
      [1, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],

    [
      [0, 2],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
  ],

  [
    [
      [0, 0],
      [3, 0],
      [0, 0],
      [0, 0],
    ],

    [
      [0, 0],
      [0, 4],
      [0, 0],
      [0, 0],
    ],
  ],

  [
    [
      [0, 0],
      [0, 0],
      [5, 0],
      [0, 0],
    ],

    [
      [0, 0],
      [0, 0],
      [0, 6],
      [0, 0],
    ],
  ],

  [
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [7, 0],
    ],

    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 8],
    ],
  ],
];

/* ---- Main ---- */
export function run() {
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  it("  -- 1d", async () => {
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    const t = tf.diag(x);
    const expected = [
      [1, 0, 0, 0],
      [0, 2, 0, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 4],
    ];
    expect(t).to.haveShape([4, 4]);
    expect(t).to.lookLike(expected);
  });
  it("  -- 2d", async () => {
    const x: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
    const t = tf.diag(x);
    expect(t).to.haveShape([4, 2, 4, 2]);
    const expected = ARR_2D;
    expect(t).to.lookLike(expected);
  });
}
