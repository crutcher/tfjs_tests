import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as loader from "../load-tf";
import type tfTypes from "@tensorflow/tfjs-core";

const arr2d = [
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

/* ---- tf.diag ----
  Creates new tensor from argument tensor
    -- each row is a copy of the argument tensor
    -- expect all values are 0 except for the diagonal
*/
describe("tf.diag(tensor): ", () => {
  it("  -- 1d", async () => {
    const tf: loader.TFModule = await loader.load();
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
    const tf: loader.TFModule = await loader.load();
    const x: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
    const t = tf.diag(x);
    expect(t).to.haveShape([4, 2, 4, 2]);
    const expected = arr2d;
    expect(t).to.lookLike(expected);
  });
});
