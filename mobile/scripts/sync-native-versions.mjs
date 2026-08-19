import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const appJsonPath = path.join(projectRoot, "app.json");
const projectPath = path.join(projectRoot, "ios", "Vigorly.xcodeproj", "project.pbxproj");
const infoPlistPath = path.join(projectRoot, "ios", "Vigorly", "Info.plist");

function replaceRequired(source, expression, replacement, label) {
  const matches = source.match(expression);
  if (!matches?.length) {
    throw new Error(`Could not find ${label}; native version sync was not applied.`);
  }
  return source.replace(expression, replacement);
}

export async function syncNativeVersions() {
  const appConfig = JSON.parse(await readFile(appJsonPath, "utf8"));
  const version = String(appConfig?.expo?.version ?? "").trim();
  const buildNumber = String(appConfig?.expo?.ios?.buildNumber ?? "").trim();

  if (!/^\d+(?:\.\d+){1,2}$/.test(version)) {
    throw new Error(`Invalid expo.version: ${JSON.stringify(version)}`);
  }
  if (!/^\d+$/.test(buildNumber)) {
    throw new Error(`Invalid expo.ios.buildNumber: ${JSON.stringify(buildNumber)}`);
  }

  const originalProject = await readFile(projectPath, "utf8");
  let nextProject = replaceRequired(
    originalProject,
    /CURRENT_PROJECT_VERSION = [^;]+;/g,
    `CURRENT_PROJECT_VERSION = ${buildNumber};`,
    "CURRENT_PROJECT_VERSION build settings",
  );
  nextProject = replaceRequired(
    nextProject,
    /MARKETING_VERSION = [^;]+;/g,
    `MARKETING_VERSION = ${version};`,
    "MARKETING_VERSION build settings",
  );

  const originalInfoPlist = await readFile(infoPlistPath, "utf8");
  let nextInfoPlist = replaceRequired(
    originalInfoPlist,
    /(<key>CFBundleShortVersionString<\/key>\s*<string>)[^<]+(<\/string>)/,
    `$1${version}$2`,
    "CFBundleShortVersionString",
  );
  nextInfoPlist = replaceRequired(
    nextInfoPlist,
    /(<key>CFBundleVersion<\/key>\s*<string>)[^<]+(<\/string>)/,
    `$1${buildNumber}$2`,
    "CFBundleVersion",
  );

  if (nextProject !== originalProject) {
    await writeFile(projectPath, nextProject);
  }
  if (nextInfoPlist !== originalInfoPlist) {
    await writeFile(infoPlistPath, nextInfoPlist);
  }

  return { version, buildNumber };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = await syncNativeVersions();
  console.log(`Native iPhone and Watch versions synced to ${result.version} (${result.buildNumber}).`);
}
