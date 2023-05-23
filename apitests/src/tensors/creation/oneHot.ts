import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

let tf: loader.TFModule;

export function run() {
  // CONSTANTS
  const INDICES = [0, 1];
  const DEPTH = 3;
  const INDICES_RESULTS = [
    {
      indices: [0, 1],
      expected: [
        [1, 0, 0],
        [0, 1, 0],
      ],
    },
    {
      indices: [1, 2, 3],
      expected: [
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
      ],
    },
    {
      indices: [
        [0, 1],
        [1, 2],
      ],
      expected: [
        [
          [1, 0, 0],
          [0, 1, 0],
        ],

        [
          [0, 1, 0],
          [0, 0, 1],
        ],
      ],
    },
  ];

  // TESTS
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  it("  -- basic", async () => {
    INDICES_RESULTS.forEach(({ indices, expected }) => {
      const x: tfTypes.Tensor = tf.tensor(indices, undefined, "int32");
      const xShape: number[] = x.shape;
      const t: tfTypes.Tensor = tf.oneHot(x, DEPTH);
      const expectedShape = [...xShape, DEPTH];
      expect(t).to.haveShape(expectedShape);
      expect(t).to.lookLike(expected);
    });
  });
  it("  -- onValue, offValue", async () => {
    //CONSTANTS
    const ON_VALUE = 8;
    const OFF_VALUE = 1;
    const RESULT = [
      [8, 1, 1],
      [1, 8, 1],
    ];
    // TESTS
    const x: tfTypes.Tensor = tf.tensor(INDICES, undefined, "int32");
    const xShape: number[] = x.shape;
    const t: tfTypes.Tensor = tf.oneHot(x, DEPTH, ON_VALUE, OFF_VALUE);
    const expectedShape = [...xShape, DEPTH];
    expect(t).to.haveShape(expectedShape);
    expect(t).to.lookLike(RESULT);
  });
  it("  -- dType", async () => {
    //CONSTANTS
    const RESULT = [
      [true, false, false],
      [false, true, false],
    ];
    // TESTS
    const x: tfTypes.Tensor = tf.tensor(INDICES, undefined, "int32");
    const xShape: number[] = x.shape;
    const t: tfTypes.Tensor = tf.oneHot(x, DEPTH, undefined, undefined, "bool");
    const expectedShape = [...xShape, DEPTH];
    expect(t).to.haveShape(expectedShape);
    expect(t).to.lookLike(RESULT);
  });
}
