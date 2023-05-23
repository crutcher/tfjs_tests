import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.mirrorPad (x, paddings, mode)-- */
export function run() {
  /* HOOKS: */
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });

  /* CONSTANTS: */
  const INPUT = [
    [
      [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
      ],
    ],
  ];

  /* TESTS: */

  it("  -- reflect: 3rd dimension", () => {
    // input shape = [ 1, 1, 3, 3 ]
    const outputShape = [1, 1, 7, 3];
    const output = [
      [
        [
          [6, 7, 8],
          [3, 4, 5],
          [0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
          [3, 4, 5],
          [0, 1, 2],
        ],
      ],
    ];
    const x: tfTypes.Tensor4D = tf.tensor4d(INPUT);
    const y = x.mirrorPad(
      [
        [0, 0],
        [0, 0],
        [2, 2],
        [0, 0],
      ],
      "reflect"
    );
    expect(y).to.haveShape(outputShape);
    expect(y).to.lookLike(output);
  });
  it("  -- reflect: 4th dimension", () => {
    // input shape = [ 1, 1, 3, 3 ]
    const outputShape = [1, 1, 3, 7];
    const output = [
      [
        [
          [2, 1, 0, 1, 2, 1, 0],
          [5, 4, 3, 4, 5, 4, 3],
          [8, 7, 6, 7, 8, 7, 6],
        ],
      ],
    ];
    const x: tfTypes.Tensor4D = tf.tensor4d(INPUT);
    const y = x.mirrorPad(
      [
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 2],
      ],
      "reflect"
    );
    expect(y).to.haveShape(outputShape);
    expect(y).to.lookLike(output);
  });
  it("  -- reflect: 3rd and 4th dimensions", () => {
    // input shape = [ 1, 1, 3, 3 ]
    const outputShape = [1, 1, 7, 7];
    const output = [
      [
        [
          [8, 7, 6, 7, 8, 7, 6],
          [5, 4, 3, 4, 5, 4, 3],
          [2, 1, 0, 1, 2, 1, 0],
          [5, 4, 3, 4, 5, 4, 3],
          [8, 7, 6, 7, 8, 7, 6],
          [5, 4, 3, 4, 5, 4, 3],
          [2, 1, 0, 1, 2, 1, 0],
        ],
      ],
    ];
    const x: tfTypes.Tensor4D = tf.tensor4d(INPUT);
    const y = x.mirrorPad(
      [
        [0, 0],
        [0, 0],
        [2, 2],
        [2, 2],
      ],
      "reflect"
    );
    expect(y).to.haveShape(outputShape);
    expect(y).to.lookLike(output);
  });
  it("  -- bad reflect: paddings out of range", () => {
    // input shape = [ 1, 1, 3, 3 ]
    const x: tfTypes.Tensor4D = tf.tensor4d(INPUT);
    const paddings: [number, number][] = [
      [0, 0],
      [0, 0],
      [3, 3],
      [0, 0],
    ];
    expect(() => x.mirrorPad(paddings, "reflect")).to.throw(
      `Padding in dimension 2 cannot be greater than or equal to 2 or less than 0 for input of shape 1,1,3,3`
    );
  });
  it("  -- symmetric : valid padding up to [3, 3]", () => {
    // input shape = [ 1, 1, 3, 3 ]
    const outputShape = [1, 1, 9, 3];
    const output = [
      [
        [
          [6, 7, 8],
          [3, 4, 5],
          [0, 1, 2],
          [0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
          [6, 7, 8],
          [3, 4, 5],
          [0, 1, 2],
        ],
      ],
    ];
    const x: tfTypes.Tensor4D = tf.tensor4d(INPUT);
    const y = x.mirrorPad(
      [
        [0, 0],
        [0, 0],
        [3, 3],
        [0, 0],
      ],
      "symmetric"
    );
    expect(y).to.haveShape(outputShape);
    expect(y).to.lookLike(output);
  });
}
