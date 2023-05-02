import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as loader from "../load-tf";
import type tfTypes from "@tensorflow/tfjs-core";

/* ---- tf.oneHot(indices, depth, onValue?, offValue?, dtype?)---- *
  1. The values represented by indices take on onValue (1) while others are offvalue (0)
  2. rank(output) = rank(indices) + 1
  3. Create a number of "rows" equal to indices.length
  4. For each "row": set column value to 1 if column index is in indices, else 0
*/
describe("tf.oneHot(indices, depth, onValue?, offValue?, dtype?): ", () => {
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
  it("  -- basic", async () => {
    const tf: loader.TFModule = await loader.load();

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
    const tf: loader.TFModule = await loader.load();
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
    const tf: loader.TFModule = await loader.load();
    //CONSTANTS
    const RESULT = [
      [1, 0, 0],
      [0, 1, 0],
    ];
    // TESTS
    const x: tfTypes.Tensor = tf.tensor(INDICES, undefined, "int32");
    const xShape: number[] = x.shape;
    const t: tfTypes.Tensor = tf.oneHot(x, DEPTH, undefined, undefined, "bool");
    const expectedShape = [...xShape, DEPTH];
    expect(t).to.haveShape(expectedShape);
    expect(t).to.lookLike(RESULT);
  });
});
