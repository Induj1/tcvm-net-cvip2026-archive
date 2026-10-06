import { readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Graphviz } from "@hpcc-js/wasm/graphviz";

const [, , inputPath, outputPath] = process.argv;

if (!inputPath || !outputPath) {
  console.error("Usage: node render_graphviz.mjs input.dot output.svg");
  process.exit(2);
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const graphviz = await Graphviz.load();
const dot = await readFile(inputPath, "utf8");
const svg = graphviz.layout(dot, "svg", "dot");
await writeFile(outputPath, svg, "utf8");
