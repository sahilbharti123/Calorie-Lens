# Vigorly product operating plan

## Product promise

Vigorly is the private fitness ledger that makes uncertain estimates inspectable. A user should be able to understand today in two seconds, record something in under ten seconds, and always know where imported or estimated data came from.

The operating principle is **glance first, evidence always**:

- Glance first: show the current state and one useful next action before detail.
- Evidence always: show source, freshness, range, assumptions, and reversibility wherever data can be delayed or uncertain.

## What we borrow from StepsApp

We borrow product mechanics, not visual cloning:

- One dominant daily state instead of a dashboard of competing modules.
- Progress that is understandable without opening a detail screen.
- Adjustable goals and calm, non-judgmental encouragement.
- A small completion moment for real behavior.
- Useful day/week/month continuity.
- Widgets and watch surfaces only after the underlying data is trustworthy.
- Fast, human support paths for permissions, wearable sync, restore, and migration.

## What its reviews warn us not to ship

- **Opaque freshness:** every Health value shows provider and last successful refresh.
- **Silent reconciliation:** imported values never appear to change mysteriously; the app explains that the provider merged sources and offers retry or correction.
- **Punitive streaks:** Vigorly celebrates logging and training consistency, never eating less or staying under calories.
- **Fragile wearable chains:** Apple Health and Health Connect are the explicit sources; unsupported indirect chains are described honestly.
- **Paywall-before-value:** the complete offline ledger remains useful without an account or subscription.
- **Entitlement ambiguity:** if monetization arrives, purchased capabilities and future additions must have explicit boundaries.
- **Ads and tracking:** no advertising inside the private health ledger.

## Competitor evidence snapshot — 1 August 2026

The strongest positive pattern is clarity plus motivation. StepsApp's US App Store listing shows a 4.8 rating across roughly 287,000 ratings and repeatedly highlights automatic tracking, a dominant daily goal, trends, achievements, widgets, Watch support, accessibility, and customization. A featured reviewer calls it simple, motivating, and better than their prior Fitbit experience, then specifically praises persistent human support after a Watch regression was fixed. Sources: [US App Store listing and reviews](https://apps.apple.com/us/app/stepsapp-pedometer/id1037595083), [independent visual review](https://www.macworld.com/article/3109586/stepsapp-review-a-gorgeous-way-to-visualize-apple-health-data.html).

The recurring negative pattern is not “more features needed”; it is uncertainty and broken trust:

- Users report delayed totals, Watch/phone disagreement, temporary crashes, and totals decreasing after sync. The developer explains that Apple Health merges sources and removes duplicates, which means the product must explain reconciliation before users interpret it as data loss. Source: [App Store Watch reviews](https://apps.apple.com/us/app/stepsapp-pedometer/id1037595083?platform=watch&see-all=reviews).
- Some international reviews complain about frequent ads, paid-feature pressure, workout-mode bugs, and difficult recovery after migration. Source: [App Store review collection](https://apps.apple.com/ua/app/stepsapp-pedometer/id1037595083?platform=iphone&see-all=reviews).
- Android review aggregation surfaces background reliability, reopening the app to resume counting, ads, and aggressive premium prompts. Treat the aggregation as directional rather than equivalent to first-party store data. Source: [Google Play negative-review aggregation](https://unstar.app/app/com.stepsappgmbh.stepsapp?country=en-US&platform=android).
- Watch-face freshness is partly constrained by operating-system update budgets, so Vigorly must distinguish platform latency from an app failure and never promise a live complication it cannot deliver. Sources: [StepsApp support on refresh behavior](https://steps.app/support/pedometer/ios/troubleshooting/how-does-stepsapp-update-my-steps-and-other-records), [user discussion of complication update limits](https://www.reddit.com/r/AppleWatch/comments/1eowyhb).

Product conclusion: borrow the glanceable goal, continuity, calm motivation, and visual craft. Improve on it with explicit provenance/freshness, honest platform limits, optional accounts, a useful free core, reversible data actions, and no ad-funded health ledger.

## Core experience

### First run

1. Explain the trust promise in one screen.
2. Ask the goal.
3. Ask only age, height, weight, and energy-equation option.
4. Show a starter plan and enter the offline vault immediately.

Activity, pace, training preferences, diet style, coaching style, bowl calibration, Health connection, and account creation are progressive setup tasks. They can improve the plan later but cannot block first value.

### Today

The screen answers, in order:

1. Is my imported data current?
2. Where do calories and protein stand?
3. What is the next useful action?
4. Can I repeat something instead of describing it again?
5. What are water, steps, and sleep at a glance?
6. What did I log today?

There is one primary action at a time. Cards are reserved for actionable groups; aligned rows and whitespace carry the rest.

### Logging

New or uncertain input follows capture → clarify → review → confirm. Known repeated input follows repeat → confirmation, with an undo path. Recent meal groups and saved meals retain their original range, confidence, basis, and source.

### Health

Every Health surface exposes:

- Provider: Apple Health or Health Connect.
- Last successful refresh time.
- Current state: not connected, syncing, current, no shared samples, or needs attention.
- Categories read.
- Plain-language expected behavior: provider data may reconcile phone and wearable sources.
- Retry and a route to platform permission settings when available.

Manual logging remains possible when Health is unavailable.

### Progress and motivation

- Celebrate goal completion briefly and respect reduced motion.
- Streaks describe logging or training consistency only.
- A missed day is neutral historical information, not failure.
- Trends distinguish missing data from zero.
- Day, week, month, and longer views use the same metric vocabulary.

## Information architecture

- **Today:** current state and next action.
- **Food:** estimate quality, meal history, recents, and saved meals.
- **Train:** resume/start workout first; routines and history second; templates/settings deeper.
- **Progress:** trends, Health connection, and data-quality context.
- **Coach:** one recommendation supported by arithmetic from the user's own ledger.

No additional primary tab is required. Widgets and watch surfaces extend Today instead of adding navigation.

## Interaction and accessibility contract

- Minimum 44 × 44 point interactive area.
- Reduced-motion users receive instant or short crossfade state changes.
- Product screens do not orchestrate page-load reveals.
- Body and control copy remain readable under Dynamic Type.
- Screen-reader labels state value, goal, source, freshness, and selected state where relevant.
- Color is never the only status signal.
- Destructive local actions prefer undo; irreversible account deletion remains explicit.

## Data and trust contract

- Unknown foods are never assigned generic nutrition values.
- Mobile, API, and companion surfaces share canonical catalog data or golden fixtures.
- Health sync never silently overwrites provenance.
- Local records remain encrypted with a device-only key.
- Cloud sync remains optional and conflict-aware.
- Account creation is offered after value, not before it.

## Success measures

- Median time from first launch to first confirmed log.
- Percentage of first sessions that reach a confirmed log.
- Median interactions required to repeat a known meal.
- Clarification rate and correction rate for estimated meals.
- Percentage of Health users who can identify provider and freshness.
- Sync retry success and permission-related support rate.
- Seven-day return rate without using guilt-based notifications.
- Crash-free active workouts and successful recovery after backgrounding.

## Delivery order

1. Progressive first run and offline-first entry.
2. Today hierarchy, recent meals, saved meals, and repeat action.
3. Health freshness and source contract.
4. Non-fabricating companion behavior and shared accuracy tests.
5. Reduced motion, touch targets, readable type, and quieter surfaces.
6. Widgets and watch/lock-screen extensions after data reliability is measured.

## Definition of done for this implementation goal

- The high-priority behaviors above exist in production source, not only in mockups.
- Existing encrypted data normalizes safely after schema additions.
- Typecheck, lint, Python tests, mobile domain tests, and Python compilation pass.
- The changed flows have empty, error, loading, success, and offline states.
- No unrecognized food path invents a calorie estimate.
- No primary interaction is smaller than a 44-point hit area.
- Reduced motion is honored by shared motion primitives.
