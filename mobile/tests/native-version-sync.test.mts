import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const appConfig = JSON.parse(readFileSync('app.json', 'utf8'));
const packageJson = JSON.parse(readFileSync('package.json', 'utf8'));
const project = readFileSync('ios/Vigorly.xcodeproj/project.pbxproj', 'utf8');
const infoPlist = readFileSync('ios/Vigorly/Info.plist', 'utf8');

function plistValue(key: string) {
  return infoPlist.match(new RegExp(`<key>${key}</key>\\s*<string>([^<]+)</string>`))?.[1];
}

test('phone and Watch native targets use the Expo marketing version and build number', () => {
  const version = String(appConfig.expo.version);
  const buildNumber = String(appConfig.expo.ios.buildNumber);
  const nativeVersions = [...project.matchAll(/MARKETING_VERSION = ([^;]+);/g)].map((match) => match[1]);
  const nativeBuilds = [...project.matchAll(/CURRENT_PROJECT_VERSION = ([^;]+);/g)].map((match) => match[1]);

  assert.ok(nativeVersions.length >= 4, 'expected Debug and Release settings for phone and Watch');
  assert.ok(nativeBuilds.length >= 4, 'expected Debug and Release build numbers for phone and Watch');
  assert.deepEqual([...new Set(nativeVersions)], [version]);
  assert.deepEqual([...new Set(nativeBuilds)], [buildNumber]);
  assert.equal(plistValue('CFBundleShortVersionString'), version);
  assert.equal(plistValue('CFBundleVersion'), buildNumber);
});

test('EAS and local iOS builds synchronize native versions before compiling', () => {
  assert.equal(packageJson.scripts.preios, 'npm run sync:native-versions');
  assert.equal(packageJson.scripts['eas-build-post-install'], 'npm run sync:native-versions');
});
