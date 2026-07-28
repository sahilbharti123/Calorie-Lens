# Calorie Lens mobile

The native iOS and Android client for Calorie Lens, built with Expo SDK 54 and
React Native.

## Included

- Today dashboard for calories, protein, water, steps, sleep, and meal rhythm
- Natural-language meal estimates with a useful offline food reference
- Breakfast, lunch, evening snack, and dinner logs
- Workout, water, steps, sleep, and weight logging
- Microphone capture and review-before-save voice flow
- SQLite-backed local persistence
- Native Apple Health/Apple Watch sync on iOS
- Native Health Connect sync on Android
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
```

## Build with EAS

```bash
npx eas build --profile development --platform ios
npx eas build --profile development --platform android
```

App identifiers are configured as `com.sahilbharti.calorielens`. Store release
builds still require your Apple Developer and Google Play accounts, signing
credentials, privacy disclosures, and Health Connect declaration.

Nutrition and exercise calories are estimates, not medical advice.
