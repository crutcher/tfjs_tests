/* To Import: "import * from load-tf" */

/* -- Types -- */
// private
type GPUModule = typeof import("@tensorflow/tfjs-node-gpu");
type JSModule = typeof import("@tensorflow/tfjs");
// public
export type TFModule = GPUModule | JSModule;

/* -- Functions: -- */

// load(): Loads tensorflow library dynamically based on OS
export async function load(): Promise<TFModule> {
  try {
    if (process.env.OS === "linux") {
      return await import("@tensorflow/tfjs-node-gpu");
    } else {
      return await import("@tensorflow/tfjs");
    }
  } catch (err) {
    throw new Error("Failed to load TensorFlow library");
  }
}
