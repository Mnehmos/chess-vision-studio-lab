// Regenerate src/generated/schemas.ts from the canonical JSON Schema.
// Run after `python -m cvslab schema export --out schemas/cvslab.schema.json`.
import { compile } from "json-schema-to-typescript";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const doc = JSON.parse(readFileSync(path.join(here, "../../schemas/cvslab.schema.json"), "utf8"));

// One compile of the root: shared $defs are hoisted to named exports exactly once.
const out = await compile(doc, "CVSLab", {
  additionalProperties: false,
  bannerComment: "// Generated from schemas/cvslab.schema.json by `npm run gen:types` — do not edit.",
});
const target = path.join(here, "../src/generated/schemas.ts");
mkdirSync(path.dirname(target), { recursive: true });
writeFileSync(target, out);

const exportsCount = (out.match(/^export /gm) || []).length;
const duplicates = out
  .split("\n")
  .filter((line) => /^(export (interface|type)) /.test(line))
  .map((line) => line.replace(/^export (interface|type) (\w+).*/, "$2"))
  .filter((name, index, all) => all.indexOf(name) !== index);
if (duplicates.length) {
  throw new Error(`duplicate generated declarations: ${[...new Set(duplicates)].join(", ")}`);
}
console.log(`wrote ${path.relative(process.cwd(), target)} (${exportsCount} exports, ${Object.keys(doc.$defs).length} schema defs)`);
