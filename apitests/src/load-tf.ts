import Os from "os";
import * as dotenv from "dotenv";
dotenv.config();
/* 
Loads a tensorflow module based on environment variable TF_MODULE
EXPORTS: load() function, TFModule type
( To Import this module elsewhere: "import * from load-tf" ) 
*/

/* -- Types -- */
// private
type GPUModule = typeof import("@tensorflow/tfjs-node-gpu");
type JSModule = typeof import("@tensorflow/tfjs");
type NodeModule = typeof import("@tensorflow/tfjs-node");
// public
export type TFModule = GPUModule | JSModule | NodeModule;

/* -- Constants -- */

// posible values of TF_MODULE. TF_MODULE must match a tensorflowjs module name
const MODULES = {
  tfjs: "tfjs",
  "tfjs-node-gpu": "tfjs-node-gpu",
  "tfjs-node": "tfjs-node",
};
const DEFAULT_MODULE = MODULES.tfjs;

/* -- Functions: -- */

/* load(): Loads tensorflow library dynamically based on OS */
export async function load(): Promise<TFModule> {
  try {
    let module = process.env.TF_MODULE;
    if (module === MODULES["tfjs-node-gpu"]) {
      if (!isLinux()) throw new Error("GPU not supported on this OS");
    } else if (module === MODULES["tfjs-node"]) {
      if (!isArm64()) throw new Error("Node not supported on this CPU");
    } else {
      module = DEFAULT_MODULE;
    }
    //return: Promise<GPUModule | JSModule | NodeModule>
    return await import(`@tensorflow/${module}`);
  } catch (err) {
    // Error handling
    let message = "Unknown Error";
    if (err instanceof Error) message = err.message;
    console.error(message);
    throw new Error("Failed to load TensorFlow library");
  }
}

function isLinux(): boolean {
  return Os.platform() === "linux";
}

function isArm64(): boolean {
  return process.arch === "arm64";
}
