# Calorie Lens mobile

The native iOS and Android client for Calorie Lens, built with Expo SDK 54 and
React Native.

## Included

- Today dashboard for calories, protein, water, steps, sleep, and meal rhythm
- Voice-first logging with typed input as a fallback
- Evidence-backed food estimates with visible ranges, confidence, and calculation basis
- A reviewed offline catalog sourced from USDA FoodData Central
- Breakfast, lunch, evening snack, and dinner logs
- Weight-personalized workout burn from the 2024 Adult Compendium of Physical Activities
- Follow-up voice questions when a portion, bowl size, duration, or weight is missing
- Workout, water, steps, sleep, and weight logging
- SQLite-backed local persistence
- Native Apple Health/Apple Watch sync on iOS
- Native Health Connect sync on Android
- Secure account signup, login, password recovery, and deletion
- Device-encrypted offline storage plus encrypted cross-device cloud sync
- Portable backup export and restore
- A personal AI coach with editable long-term memory and offline fallback
- One shared design system and codebase for both platforms

## Local setup

```bash
npm install
cp .env.example .env
```

Start the AI API from the repository root:

```bash
pip install -r requirements.txt
GOOGLE_API_KEY=your_key uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Set `EXPO_PUBLIC_API_URL` in `.env` to the API address reachable by the device.
For a physical phone, use your computer's LAN IP rather than `localhost`.

Account sessions and recovery codes use iOS Keychain or Android Keystore via
Expo SecureStore. Fitness records are encrypted before being written to the
local SQLite key-value store. When signed in, changes are queued offline and
automatically synchronized when the API is reachable.

The AI service extracts facts from speech; it does not supply calories. Food
and exercise numbers are calculated after transcription by the deterministic
reference engine.

## Accuracy setup

Open **Profile & goals** once and add:

- your current body weight, used in active-energy calculations;
- the capacity of your usual bowl in ml.

To measure a bowl, fill it with water and read the volume in a measuring jug.
The app intentionally asks for this value instead of treating every Indian
katori or bowl as identical. Its default cup measure is 200 ml.

For best results, say the cooked weight or package serving:

- `Lunch: 2 rotis and one 200 ml bowl rajma`
- `100 grams grilled chicken and 150 grams cooked rice`
- `30 minute brisk walk, moderate effort`

## Run native builds

HealthKit and Health Connect are native modules, so this app uses an Expo
development client instead of Expo Go.

```bash
npm run prebuild
npm run ios
# or
npm run android
```

The first Health sync asks for operating-system permissions. Apple Watch data
already written to Apple Health is read through HealthKit. Android reads
compatible health and wearable data through Health Connect.

## Quality checks

```bash
npm run typecheck
npm run lint
cd ..
python -m unittest discover -s tests -v
```

## Build with EAS

```bash
npx eas build --profile development --platform ios
npx eas build --profile development --platform android
```

App identifiers are configured as `com.sahilbharti.calorielens`. Store release
builds still require your Apple Developer and Google Play accounts, signing
credentials, privacy disclosures, and Health Connect declaration.

See [`../ACCURACY.md`](../ACCURACY.md) for calculation rules, source records,
benchmark cases, and limitations. Nutrition and exercise calories are
estimates, not medical advice.
