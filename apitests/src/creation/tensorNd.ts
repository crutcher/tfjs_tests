import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../load-tf";

let tf: loader.TFModule;

export function run() {
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  it("  -- tf.tensor1d()", () => {
    const t: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    expect(t).to.haveDtype("float32");
    expect(t).to.haveShape([4]);
    expect(t).to.lookLike([1, 2, 3, 4]);
  });
  it("  -- tf.tensor2d()", () => {
    const t: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    expect(t).to.haveDtype("float32");
    expect(t).to.haveShape([2, 2]);
    expect(t.arraySync()).to.eql([
      [1, 2],
      [3, 4],
    ]);
  });
  it("  -- tf.tensor3d() : with dtype included", () => {
    const array = [
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ];

    const t: tfTypes.Tensor3D = tf.tensor3d(array, undefined, "int32");
    expect(t).to.haveDtype("int32");
    expect(t).to.haveShape([2, 2, 2]);
    expect(t.arraySync()).to.eql(array);
  });
  it("  -- tf.tensor4d()", () => {
    const array = [
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
      [
        [
          [9, 10],
          [11, 12],
        ],
        [
          [13, 14],
          [15, 16],
        ],
      ],
    ];

    const t: tfTypes.Tensor4D = tf.tensor4d(array, undefined, "int32");
    expect(t).to.haveDtype("int32");
    expect(t).to.haveShape([2, 2, 2, 2]);
    expect(t.arraySync()).to.eql(array);
  });
  it("  -- tf.tensor5d()", () => {
    const array = [
      [
        [
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
          [
            [7, 8, 9],
            [10, 11, 12],
          ],
        ],
        [
          [
            [13, 14, 15],
            [16, 17, 18],
          ],
          [
            [19, 20, 21],
            [22, 23, 24],
          ],
        ],
      ],
      [
        [
          [
            [25, 26, 27],
            [28, 29, 30],
          ],
          [
            [31, 32, 33],
            [34, 35, 36],
          ],
        ],
        [
          [
            [37, 38, 39],
            [40, 41, 42],
          ],
          [
            [43, 44, 45],
            [46, 47, 48],
          ],
        ],
      ],
    ];

    const t: tfTypes.Tensor5D = tf.tensor5d(array);
    expect(t).to.haveDtype("float32");
    expect(t).to.haveShape([2, 2, 2, 2, 3]);
    expect(t.arraySync()).to.eql(array);
  });
  it("  -- tf.tensor6d()", () => {
    const array = [
      [
        [
          [
            [
              [1, 2],
              [3, 4],
            ],
            [
              [5, 6],
              [7, 8],
            ],
          ],
          [
            [
              [9, 10],
              [11, 12],
            ],
            [
              [13, 14],
              [15, 16],
            ],
          ],
        ],
        [
          [
            [
              [17, 18],
              [19, 20],
            ],
            [
              [21, 22],
              [23, 24],
            ],
          ],
          [
            [
              [25, 26],
              [27, 28],
            ],
            [
              [29, 30],
              [31, 32],
            ],
          ],
        ],
      ],
      [
        [
          [
            [
              [33, 34],
              [35, 36],
            ],
            [
              [37, 38],
              [39, 40],
            ],
          ],
          [
            [
              [41, 42],
              [43, 44],
            ],
            [
              [45, 46],
              [47, 48],
            ],
          ],
        ],
        [
          [
            [
              [49, 50],
              [51, 52],
            ],
            [
              [53, 54],
              [55, 56],
            ],
          ],
          [
            [
              [57, 58],
              [59, 60],
            ],
            [
              [61, 62],
              [63, 64],
            ],
          ],
        ],
      ],
    ];

    const t: tfTypes.Tensor = tf.tensor6d(array);
    expect(t).to.haveDtype("float32");
    expect(t).to.haveShape([2, 2, 2, 2, 2, 2]);
    expect(t.arraySync()).to.eql(array);
  });
}
