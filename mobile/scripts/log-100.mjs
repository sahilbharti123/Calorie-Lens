/**
 * Logs a hundred realistic entries and prints what each one actually produced,
 * so the numbers can be read rather than assumed.
 *
 * The phrasing sweep proves an entry appears. This proves the entry is right:
 * the food matched, the portion, the energy, and whether the macros reconcile
 * with the calories at 4/4/9 kcal per gram. A catalog row with the wrong fat
 * figure passes every structural test ever written and still tells the user
 * something false, so the arithmetic is checked here on every single item.
 */
import { parseFitnessCommand } from '../src/lib/nutrition.ts';

const CTX = { cupMl: 200, bowlMl: 250, weightKg: 72 };

const ENTRIES = [
  // ---- Indian breakfast
  '3 idlis and sambar', '2 dosas', 'a masala dosa', 'upma and coffee',
  '2 parathas with curd', 'poha for breakfast', '2 boiled eggs and toast',
  'omelette and 2 toast',
  // ---- Indian mains
  '2 rotis and 1 bowl dal', '3 rotis with sabzi', 'rajma chawal',
  '1 bowl dal and 1 cup rice', 'chole bhature', '2 rotis and paneer curry',
  'butter chicken with 2 naan', 'chicken curry and rice', 'biryani',
  'khichdi for dinner', 'sambar rice', 'dal tadka and 2 rotis',
  'palak paneer with roti', 'egg curry and rice', 'fish curry with rice',
  'mutton curry and 2 rotis', 'pulao and raita',
  // ---- western
  '2 slices of pizza', 'a burger and fries', 'a chicken sandwich',
  'pasta with tomato sauce', 'a bowl of pasta', 'caesar salad',
  '200 g grilled chicken and salad', 'scrambled eggs on toast',
  'a bowl of soup', 'chicken nuggets and fries',
  // ---- weighed / precise
  '150 g paneer', '200 g chicken breast', '100 g rice', '250 g curd',
  '50 g almonds', '30 g whey protein', '2 tbsp peanut butter', '100 g oats',
  '0.2 kg chicken', '1 kg watermelon',
  // ---- snacks and sweets
  'a banana', '2 bananas', 'an apple', 'a bar of dark chocolate',
  'a packet of chips', '4 biscuits', 'a samosa', '2 samosas',
  'a gulab jamun', 'a slice of cake', 'a bowl of ice cream', 'a handful of almonds',
  'a protein bar', 'popcorn',
  // ---- drinks
  'a coffee', 'a latte', '2 cups of chai', 'a glass of milk',
  'a beer', 'two beers', '500 ml of beer', 'a pint of beer',
  '2 pegs of whisky', 'a glass of red wine', 'a coke', 'a diet coke',
  'orange juice', 'a mango lassi', 'a protein shake', 'coconut water',
  'a can of red bull',
  // ---- water
  'I drank 2 litres of water', '3 glasses of water', '500 ml water',
  // ---- hinglish
  'maine 2 roti aur dal khaya', 'ek katori rajma khaya', 'do anda khaya',
  'thoda paneer khaya', 'aadha bowl rice khaya', 'ek glass doodh piya',
  '2 idli aur sambar khaya',
  // ---- body and activity
  'my weight is 78.4', 'I weigh 80 kg', '10000 steps', 'I slept 7.5 hours',
  '30 minute brisk walk', 'I ran for 20 minutes hard', '45 min moderate cycling',
  'gym for 1 hour moderate', 'yoga for 45 minutes light', 'I swam for 30 minutes moderate',
  // ---- messy real speech
  'so i had like 2 rotis and some dal', 'um a beer',
  'i had 200 grams of chicken breast grilled', 'today i had 3 idlis and a coffee',
  'i had a one bowl of italy. i had kne bowl of idli',
  'lunch was rice dal and sabzi', 'dinner: 2 rotis and chicken curry',
  'had a coffee and 2 biscuits', 'i ate 2 eggs and drank a glass of milk',
];

/** Energy implied by the macros, at 4/4/9 kcal per gram. */
const macroKcal = (item) => item.protein * 4 + item.carbs * 4 + item.fat * 9;

let entryNumber = 0;
const problems = [];

for (const say of ENTRIES) {
  entryNumber += 1;
  const parsed = await parseFitnessCommand(say, undefined, CTX);
  const label = `${String(entryNumber).padStart(3)}. ${say}`;

  if (parsed.clarification) {
    console.log(`${label}\n     ASKS: ${parsed.clarification.question}`);
    problems.push(`${say} — asks instead of logging`);
    continue;
  }

  const lines = [];
  for (const operation of parsed.operations) {
    if (operation.type === 'meal') {
      for (const item of operation.items) {
        const implied = macroKcal(item);
        // Alcohol carries most of its energy as ethanol at 7 kcal/g, which is
        // not one of the three macros, so beer and wine legitimately fail this
        // reconciliation. Very small totals are excluded too: a 7 kcal coffee
        // cannot be checked to within 30% of anything.
        const alcoholic = /beer|wine|whisky|vodka|gin|rum|cocktail|spirit/i.test(item.name);
        const off = implied > 0 && item.calories >= 25 && !alcoholic
          && Math.abs(implied - item.calories) / item.calories > 0.3;
        if (off) {
          problems.push(
            `${say} — ${item.name}: ${item.calories} kcal but macros imply ${Math.round(implied)}`,
          );
        }
        lines.push(
          `     ${item.calories.toString().padStart(4)} kcal  ${item.name} · ${item.quantity}`
          + `  (${item.calorieLow}–${item.calorieHigh}, ${item.confidence})${off ? '   << MACROS DISAGREE' : ''}`,
        );
      }
    } else if (operation.type === 'workout') {
      lines.push(`     ${operation.calories.toString().padStart(4)} kcal burnt  ${operation.name} · ${operation.durationMin} min ${operation.intensity}`);
    } else {
      lines.push(`     ${String(operation.amount).padStart(4)}  ${operation.type}`);
    }
  }
  console.log(`${label}\n${lines.join('\n')}`);
}

console.log(`\n${'='.repeat(70)}`);
if (problems.length) {
  console.log(`${problems.length} problems:\n`);
  for (const problem of problems) console.log(`  · ${problem}`);
} else {
  console.log('All 100 entries logged with self-consistent numbers.');
}
process.exit(problems.length ? 1 : 0);
