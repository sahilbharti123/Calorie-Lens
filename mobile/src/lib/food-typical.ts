import type { FoodReference } from '@/src/lib/food-catalog';

/**
 * Typical-composition reference values.
 *
 * These are NOT USDA records and must never be presented as though they were.
 * They are ordinary published figures for a category — a pint of lager, a slice
 * of cheese pizza, a bar of milk chocolate — and every one of them carries a
 * deliberately wide `calorieVariance`, because a category is exactly the thing
 * that varies. Two lagers differ by a fifth; two restaurant curries by half.
 *
 * The point of having them is that refusing to answer is the worst outcome
 * available. The app's promise is a range and its assumptions instead of false
 * precision — so meeting an unrecognised beer with "I have no reference" breaks
 * that promise in the one situation it exists for. A wide, honestly-labelled
 * range beats silence, and it beats a confident single number even more.
 *
 * Where a real USDA record exists for a food, it belongs in `food-catalog.ts`
 * instead. That file is the high-confidence tier; this one is the fallback.
 */
const t = (
  name: string,
  aliases: string[],
  calories: number,
  protein: number,
  carbs: number,
  fat: number,
  extra: Partial<FoodReference> = {},
): FoodReference => ({
  name,
  aliases,
  calories,
  protein,
  carbs,
  fat,
  tier: 'typical',
  // ±22% unless the category is unusually tight or unusually wild.
  calorieVariance: 0.22,
  ...extra,
});

/** A drink measured by volume; density is close enough to water. */
const drink = (
  name: string,
  aliases: string[],
  calories: number,
  protein: number,
  carbs: number,
  fat: number,
  extra: Partial<FoodReference> = {},
) => t(name, aliases, calories, protein, carbs, fat, {
  density: 1.0,
  densityVariance: 0.02,
  // What "a beer" or "two coffees" means when no amount is given.
  servingMl: 330,
  ...extra,
});

export const TYPICAL_FOODS: FoodReference[] = [
  /* ----------------------------------------------------------- alcohol ---- */
  drink('Beer (regular, ~5%)', ['beer', 'lager', 'pint', 'hoegaarden', 'kingfisher', 'budweiser', 'heineken', 'corona'], 43, 0.5, 3.6, 0),
  drink('Beer (strong, ~8%)', ['strong beer', 'strong lager'], 65, 0.5, 5.5, 0),
  drink('Light beer', ['light beer', 'lite beer'], 29, 0.2, 1.6, 0),
  drink('Wine (red)', ['red wine'], 85, 0.1, 2.6, 0, { servingMl: 150, density: 0.99 }),
  drink('Wine (white)', ['white wine', 'wine'], 82, 0.1, 2.6, 0, { servingMl: 150, density: 0.99 }),
  drink('Whisky, vodka, gin, rum (40%)', ['whisky', 'whiskey', 'vodka', 'gin', 'rum', 'tequila', 'brandy', 'spirit'], 231, 0, 0, 0, { servingMl: 30, density: 0.95 }),
  drink('Cocktail (mixed, sweet)', ['cocktail', 'mojito', 'margarita', 'pina colada', 'long island'], 120, 0.1, 14, 0, { calorieVariance: 0.4 }),

  /* --------------------------------------------------------- hot drinks --- */
  drink('Tea with milk and sugar', ['chai', 'milk tea', 'masala chai'], 45, 1.4, 6.5, 1.4),
  drink('Black tea, no sugar', ['black tea', 'green tea', 'tea'], 1, 0, 0.2, 0, { calorieVariance: 0.5 }),
  drink('Black coffee', ['black coffee', 'americano', 'espresso', 'coffee'], 2, 0.1, 0, 0, { calorieVariance: 0.5 }),
  drink('Latte / cappuccino (whole milk)', ['latte', 'cappuccino', 'flat white', 'cafe latte'], 55, 3, 5.3, 2.4),

  /* -------------------------------------------------------- cold drinks --- */
  drink('Cola / soft drink (regular)', ['cola', 'coke', 'pepsi', 'soft drink', 'soda', 'fizzy drink', 'sprite', 'fanta', 'thums up'], 42, 0, 10.6, 0, { servingMl: 200, density: 1.04 }),
  drink('Diet soft drink', ['diet coke', 'coke zero', 'diet soda', 'pepsi max'], 0.4, 0, 0.1, 0, { calorieVariance: 0.5 }),
  drink('Orange juice', ['orange juice', 'juice', 'mosambi juice'], 45, 0.7, 10.4, 0.2, { servingMl: 200, density: 1.05 }),
  drink('Apple juice', ['apple juice'], 46, 0.1, 11.3, 0.1, { servingMl: 200, density: 1.05 }),
  drink('Mango shake / lassi (sweet)', ['mango shake', 'lassi', 'sweet lassi', 'milkshake', 'thickshake'], 95, 2.6, 15, 2.6, { servingMl: 200, density: 1.05 }),
  drink('Buttermilk / chaas', ['buttermilk', 'chaas', 'chhach'], 30, 1.6, 3.2, 1, { servingMl: 240, density: 1.03 }),
  drink('Coconut water', ['coconut water', 'nariyal pani'], 19, 0.7, 3.7, 0.2, { servingMl: 200, density: 1.01 }),
  drink('Energy drink', ['energy drink', 'red bull', 'monster'], 45, 0, 11, 0, { servingMl: 200, density: 1.04 }),
  drink('Sports drink', ['gatorade', 'sports drink', 'electrolyte drink'], 25, 0, 6, 0, { servingMl: 300, density: 1.03 }),
  drink('Protein shake (whey, water)', ['protein shake', 'whey shake', 'protein drink'], 45, 8.5, 1.8, 0.7, { servingMl: 200, density: 1.02, calorieVariance: 0.3 }),

  /* -------------------------------------------------- western mains ------- */
  t('Pizza (cheese, thin base)', ['pizza', 'margherita'], 266, 11, 33, 10, { pieceG: 107, pieceVariance: 0.3, calorieVariance: 0.28 }),
  t('Burger (beef, with bun)', ['burger', 'hamburger', 'cheeseburger', 'whopper'], 250, 13, 20, 12, { pieceG: 180, pieceVariance: 0.3, calorieVariance: 0.3 }),
  t('Chicken burger', ['chicken burger', 'chicken sandwich', 'mcchicken'], 230, 14, 22, 10, { pieceG: 170, pieceVariance: 0.3, calorieVariance: 0.3 }),
  t('Sandwich (filled)', ['sandwich', 'sub', 'panini'], 230, 10, 26, 9, { pieceG: 200, pieceVariance: 0.35, calorieVariance: 0.35 }),
  t('Pasta with tomato sauce (cooked)', ['pasta', 'spaghetti', 'penne', 'macaroni'], 130, 4.5, 22, 2.6, { servingMl: 250, density: 0.7, densityVariance: 0.15 }),
  t('Pasta with cream sauce (cooked)', ['white sauce pasta', 'alfredo', 'carbonara', 'creamy pasta'], 190, 6, 20, 9.5, { servingMl: 250, density: 0.72, densityVariance: 0.15, calorieVariance: 0.3 }),
  t('Instant noodles (prepared)', ['maggi', 'instant noodles', 'ramen', 'cup noodles'], 145, 3.5, 19, 6, { servingMl: 500, density: 0.6, densityVariance: 0.2 }),
  t('Hakka / stir-fried noodles', ['hakka noodles', 'chow mein', 'noodles', 'chowmein'], 165, 5, 24, 5.5, { servingMl: 400, density: 0.6, densityVariance: 0.2 }),
  t('Fried rice', ['fried rice', 'egg fried rice'], 165, 4.5, 25, 5, { servingMl: 330, density: 0.8, densityVariance: 0.15 }),
  t('French fries', ['french fries', 'fries', 'chips (fried)', 'wedges'], 310, 3.4, 41, 15, { servingMl: 330, density: 0.5, densityVariance: 0.2 }),
  t('Chicken nuggets', ['nuggets', 'chicken nuggets'], 290, 15, 18, 18, { pieceG: 17, pieceVariance: 0.2 }),
  t('Fried chicken (coated)', ['fried chicken', 'kfc', 'broasted chicken'], 260, 20, 10, 16, { pieceG: 120, pieceVariance: 0.3 }),
  t('Omelette (2 eggs, oil)', ['omelette', 'omelet'], 165, 11, 1.2, 13, { pieceG: 120, pieceVariance: 0.25 }),
  t('Scrambled eggs', ['scrambled eggs', 'scrambled egg', 'bhurji'], 160, 11, 2, 12),
  t('Soup (clear / vegetable)', ['soup', 'clear soup', 'tomato soup'], 40, 1.5, 6, 1.2, { servingMl: 330, density: 1.0, densityVariance: 0.05, calorieVariance: 0.35 }),
  t('Salad with dressing', ['salad', 'caesar salad', 'green salad'], 90, 2.5, 7, 6, { servingMl: 330, density: 0.4, densityVariance: 0.25, calorieVariance: 0.4 }),

  /* --------------------------------------------------- indian mains ------- */
  t('Paneer curry (butter / makhani)', ['paneer butter masala', 'butter paneer', 'shahi paneer', 'paneer curry', 'kadai paneer'], 220, 8, 9, 17, { servingMl: 330, density: 0.95, densityVariance: 0.12, calorieVariance: 0.3 }),
  t('Palak paneer', ['palak paneer', 'saag paneer'], 165, 8, 7, 12, { density: 0.95, densityVariance: 0.12, calorieVariance: 0.3 }),
  t('Paneer tikka (dry)', ['paneer tikka'], 235, 15, 6, 17, { calorieVariance: 0.28 }),
  t('Butter chicken', ['butter chicken', 'murgh makhani', 'chicken tikka masala', 'tikka masala'], 195, 12, 6, 13, { density: 0.95, densityVariance: 0.12, calorieVariance: 0.3 }),
  t('Chicken tikka (dry)', ['chicken tikka', 'tandoori chicken'], 195, 24, 3, 9.5, { calorieVariance: 0.25 }),
  t('Mutton curry', ['mutton curry', 'mutton', 'lamb curry', 'rogan josh'], 210, 15, 5, 15, { density: 0.95, densityVariance: 0.12, calorieVariance: 0.3 }),
  t('Fish curry', ['fish curry', 'fish gravy'], 130, 13, 4, 7, { density: 0.95, densityVariance: 0.12, calorieVariance: 0.3 }),
  t('Egg curry', ['egg curry', 'anda curry'], 150, 8, 6, 11, { density: 0.95, densityVariance: 0.12, calorieVariance: 0.3 }),
  t('Mixed vegetable sabzi', ['sabzi', 'mixed vegetable', 'veg curry', 'bhindi', 'aloo gobi', 'baingan bharta'], 115, 3, 11, 7, { density: 0.8, densityVariance: 0.15, calorieVariance: 0.3 }),
  t('Sambar', ['sambar', 'sambhar'], 65, 3, 9, 2, { density: 0.95, densityVariance: 0.1 }),
  t('Rasam', ['rasam'], 35, 1.5, 5, 1, { density: 1.0, densityVariance: 0.05 }),
  t('Kadhi', ['kadhi', 'kadi'], 95, 3.5, 8, 5.5, { density: 0.95, densityVariance: 0.1 }),
  t('Rajma / chole gravy (restaurant)', ['rajma masala', 'chole masala', 'chana masala'], 145, 6, 18, 5.5, { density: 0.85, densityVariance: 0.15, calorieVariance: 0.3 }),
  t('Dal makhani', ['dal makhani', 'daal makhani'], 165, 6, 14, 9.5, { density: 0.9, densityVariance: 0.12, calorieVariance: 0.3 }),
  t('Dal tadka', ['dal tadka', 'dal fry', 'tadka dal'], 120, 6, 15, 4, { density: 0.9, densityVariance: 0.12 }),
  t('Pav bhaji', ['pav bhaji'], 165, 4, 20, 8, { calorieVariance: 0.3 }),
  t('Chole bhature', ['chole bhature', 'chole bhatura'], 285, 8, 33, 14, { calorieVariance: 0.32 }),
  t('Masala dosa', ['masala dosa'], 170, 4, 26, 6, { pieceG: 200, pieceVariance: 0.3, calorieVariance: 0.3 }),
  t('Uttapam', ['uttapam', 'uthappam'], 165, 4.5, 26, 5, { pieceG: 140, pieceVariance: 0.25 }),
  t('Vada (medu)', ['vada', 'medu vada'], 300, 7, 32, 16, { pieceG: 50, pieceVariance: 0.25 }),
  t('Upma', ['upma'], 145, 3.5, 22, 5, { density: 0.75, densityVariance: 0.15 }),
  t('Poha', ['poha'], 130, 2.5, 22, 4, { density: 0.6, densityVariance: 0.15 }),
  t('Khichdi', ['khichdi', 'khichri'], 120, 4.5, 19, 3, { density: 0.85, densityVariance: 0.15 }),
  t('Pulao / veg biryani', ['pulao', 'pulav', 'veg biryani', 'jeera rice'], 165, 3.5, 26, 5.5, { density: 0.8, densityVariance: 0.15 }),
  t('Naan', ['naan', 'butter naan'], 310, 9, 50, 8, { pieceG: 90, pieceVariance: 0.25 }),
  t('Tandoori roti', ['tandoori roti'], 275, 8, 52, 3.5, { pieceG: 60, pieceVariance: 0.2 }),
  t('Puri', ['puri', 'poori'], 385, 7, 43, 20, { pieceG: 25, pieceVariance: 0.25 }),
  t('Idli sambar (plate)', ['idli sambar'], 105, 3.5, 19, 1.5, { calorieVariance: 0.28 }),
  t('Pakora / bhaji (fried)', ['pakora', 'pakoda', 'bhajji', 'bhaji'], 315, 7, 30, 19, { pieceG: 25, pieceVariance: 0.3 }),
  t('Spring roll (fried)', ['spring roll'], 250, 5, 30, 12, { pieceG: 60, pieceVariance: 0.25 }),
  t('Momos (steamed)', ['momo', 'momos', 'dumpling', 'dumplings'], 185, 7, 26, 5.5, { pieceG: 30, pieceVariance: 0.2 }),
  t('Kathi roll / frankie', ['kathi roll', 'frankie', 'wrap', 'shawarma'], 215, 9, 24, 9, { pieceG: 200, pieceVariance: 0.3, calorieVariance: 0.3 }),
  t('Vada pav', ['vada pav'], 265, 6, 38, 10, { pieceG: 120, pieceVariance: 0.2 }),
  t('Poha / upma plate', ['breakfast plate'], 140, 3.5, 22, 4.5, { calorieVariance: 0.35 }),

  /* ------------------------------------------------------------ sweets ---- */
  t('Milk chocolate', ['chocolate', 'milk chocolate', 'dairy milk', 'cadbury'], 535, 7.6, 59, 30, { pieceG: 25, pieceVariance: 0.3 }),
  t('Dark chocolate (70%)', ['dark chocolate'], 600, 7.8, 46, 43, { pieceG: 25, pieceVariance: 0.3 }),
  t('Biscuits / cookies', ['biscuit', 'biscuits', 'cookie', 'cookies', 'parle g', 'oreo'], 480, 6, 68, 20, { pieceG: 12, pieceVariance: 0.35 }),
  t('Cake (frosted)', ['cake', 'pastry', 'cupcake'], 375, 4, 52, 17, { pieceG: 90, pieceVariance: 0.35, calorieVariance: 0.3 }),
  t('Doughnut', ['doughnut', 'donut'], 420, 5, 48, 23, { pieceG: 60, pieceVariance: 0.25 }),
  t('Ice cream (vanilla)', ['ice cream', 'icecream', 'gelato'], 205, 3.5, 24, 11, { density: 0.55, densityVariance: 0.15 }),
  t('Gulab jamun', ['gulab jamun'], 330, 4, 48, 14, { pieceG: 40, pieceVariance: 0.25 }),
  t('Rasgulla', ['rasgulla', 'rosogolla'], 190, 4, 38, 2.5, { pieceG: 45, pieceVariance: 0.25 }),
  t('Jalebi', ['jalebi'], 420, 2.5, 65, 17, { pieceG: 30, pieceVariance: 0.3 }),
  t('Laddu', ['laddu', 'ladoo', 'besan laddu'], 425, 7, 55, 20, { pieceG: 40, pieceVariance: 0.25 }),
  t('Barfi / kalakand', ['barfi', 'burfi', 'kalakand'], 400, 8, 45, 21, { pieceG: 30, pieceVariance: 0.25 }),
  t('Halwa', ['halwa', 'halva', 'sheera'], 330, 4, 45, 15, { density: 1.0, densityVariance: 0.15 }),
  t('Kheer / payasam', ['kheer', 'payasam', 'rice pudding'], 130, 3.5, 20, 4, { density: 1.0, densityVariance: 0.1 }),
  t('Shahi tukda / bread pudding', ['shahi tukda', 'bread pudding', 'double ka meetha'], 330, 6, 40, 16, { calorieVariance: 0.3 }),
  t('Sugar', ['sugar', 'cheeni'], 387, 0, 100, 0, { density: 0.85, densityVariance: 0.05, calorieVariance: 0.05 }),
  t('Honey', ['honey', 'shahad'], 304, 0.3, 82, 0, { density: 1.42, densityVariance: 0.05, calorieVariance: 0.05 }),
  t('Jam', ['jam', 'marmalade'], 260, 0.4, 65, 0.1, { density: 1.3, densityVariance: 0.08 }),

  /* ------------------------------------------------------------ snacks ---- */
  t('Potato crisps', ['crisps', 'potato chips', 'lays', 'kurkure'], 545, 6, 53, 34, { calorieVariance: 0.18 }),
  t('Namkeen / bhujia', ['namkeen', 'bhujia', 'sev', 'mixture'], 540, 12, 45, 34),
  t('Popcorn (buttered)', ['popcorn'], 420, 8, 55, 20, { density: 0.15, densityVariance: 0.25 }),
  t('Roasted peanuts', ['peanuts', 'moongphali', 'groundnut'], 585, 24, 21, 50, { density: 0.6, densityVariance: 0.1 }),
  t('Cashews', ['cashew', 'cashews', 'kaju'], 555, 18, 30, 44, { density: 0.55, densityVariance: 0.1 }),
  t('Walnuts', ['walnut', 'walnuts', 'akhrot'], 654, 15, 14, 65, { density: 0.5, densityVariance: 0.1 }),
  t('Protein bar', ['protein bar'], 375, 30, 38, 11, { pieceG: 60, pieceVariance: 0.2 }),
  t('Granola / muesli', ['granola', 'muesli'], 450, 10, 64, 17, { density: 0.45, densityVariance: 0.15 }),
  t('Breakfast cereal (flakes)', ['cornflakes', 'cereal', 'chocos'], 380, 7, 84, 1.5, { density: 0.35, densityVariance: 0.2 }),

  /* ------------------------------------------------------------- dairy ---- */
  t('Skimmed milk', ['skimmed milk', 'skim milk', 'toned milk', 'low fat milk'], 35, 3.4, 5, 0.1, { density: 1.03, densityVariance: 0.02, calorieVariance: 0.12 }),
  t('Greek yogurt (plain)', ['greek yogurt', 'hung curd'], 59, 10, 3.6, 0.4, { density: 1.03, densityVariance: 0.03, calorieVariance: 0.15 }),
  t('Cheddar / processed cheese', ['cheese', 'cheddar', 'cheese slice'], 400, 25, 1.3, 33, { pieceG: 20, pieceVariance: 0.2 }),
  t('Mozzarella', ['mozzarella'], 300, 22, 2.2, 22),
  t('Cream', ['cream', 'malai', 'fresh cream'], 340, 2.1, 2.8, 36, { density: 1.0, densityVariance: 0.05 }),
  t('Ghee', ['ghee', 'clarified butter'], 900, 0, 0, 100, { density: 0.91, densityVariance: 0.02, calorieVariance: 0.05 }),

  /* ------------------------------------------------ meat, fish, protein --- */
  t('Chicken thigh (cooked)', ['chicken thigh', 'chicken leg'], 210, 26, 0, 11, { calorieVariance: 0.18 }),
  t('Mutton / lamb (cooked)', ['lamb', 'mutton meat', 'goat meat'], 265, 25, 0, 18, { calorieVariance: 0.2 }),
  t('Pork (cooked)', ['pork', 'bacon', 'ham'], 300, 25, 0, 22, { calorieVariance: 0.25 }),
  t('Beef (cooked, lean)', ['beef', 'steak'], 250, 26, 0, 16, { calorieVariance: 0.22 }),
  t('Fish (white, cooked)', ['fish', 'tilapia', 'cod', 'pomfret', 'rohu'], 130, 26, 0, 2.5, { calorieVariance: 0.2 }),
  t('Salmon (cooked)', ['salmon'], 208, 22, 0, 13, { calorieVariance: 0.15 }),
  t('Prawns / shrimp (cooked)', ['prawn', 'prawns', 'shrimp'], 99, 21, 0.2, 1.4, { calorieVariance: 0.15 }),
  t('Soya chunks (cooked)', ['soya chunks', 'soya', 'meal maker'], 105, 15, 8, 0.5, { density: 0.6, densityVariance: 0.15 }),
  t('Whey protein powder', ['whey protein', 'protein powder'], 400, 80, 8, 5, { pieceG: 30, pieceVariance: 0.1, calorieVariance: 0.12 }),

  /* -------------------------------------------------------------- carbs --- */
  t('Bread (white)', ['white bread'], 265, 9, 49, 3.2, { pieceG: 28, pieceVariance: 0.15 }),
  t('Chapati with ghee', ['ghee roti', 'ghee chapati'], 340, 8, 46, 14, { pieceG: 45, pieceVariance: 0.2 }),
  t('Cooked quinoa', ['quinoa'], 120, 4.4, 21, 1.9, { density: 0.75, densityVariance: 0.12, calorieVariance: 0.12 }),
  t('Cooked pasta (plain)', ['plain pasta', 'boiled pasta'], 158, 5.8, 31, 0.9, { density: 0.7, densityVariance: 0.12, calorieVariance: 0.12 }),
  t('Corn (sweet, cooked)', ['corn', 'sweetcorn', 'bhutta'], 96, 3.4, 21, 1.5, { density: 0.7, densityVariance: 0.1, calorieVariance: 0.12 }),

  /* ---------------------------------------------------- fruit & veg ------- */
  t('Orange', ['orange', 'santra'], 47, 0.9, 12, 0.1, { pieceG: 130, pieceVariance: 0.25, calorieVariance: 0.12 }),
  t('Mango', ['mango', 'aam'], 60, 0.8, 15, 0.4, { pieceG: 200, pieceVariance: 0.3, calorieVariance: 0.12 }),
  t('Grapes', ['grapes', 'angoor'], 69, 0.7, 18, 0.2, { density: 0.6, densityVariance: 0.1, calorieVariance: 0.12 }),
  t('Papaya', ['papaya', 'papita'], 43, 0.5, 11, 0.3, { density: 0.6, densityVariance: 0.1, calorieVariance: 0.12 }),
  t('Watermelon', ['watermelon', 'tarbooj'], 30, 0.6, 7.6, 0.2, { density: 0.6, densityVariance: 0.1, calorieVariance: 0.12 }),
  t('Pomegranate', ['pomegranate', 'anar'], 83, 1.7, 19, 1.2, { density: 0.6, densityVariance: 0.1, calorieVariance: 0.12 }),
  t('Guava', ['guava', 'amrood'], 68, 2.6, 14, 1, { pieceG: 120, pieceVariance: 0.25, calorieVariance: 0.12 }),
  t('Dates', ['dates', 'khajoor'], 282, 2.5, 75, 0.4, { pieceG: 8, pieceVariance: 0.2, calorieVariance: 0.12 }),
  t('Avocado', ['avocado'], 160, 2, 8.5, 15, { pieceG: 150, pieceVariance: 0.25, calorieVariance: 0.15 }),
  t('Cucumber', ['cucumber', 'kheera'], 15, 0.7, 3.6, 0.1, { density: 0.6, densityVariance: 0.1, calorieVariance: 0.15 }),
  t('Tomato', ['tomato', 'tamatar'], 18, 0.9, 3.9, 0.2, { pieceG: 100, pieceVariance: 0.25, calorieVariance: 0.15 }),
  t('Carrot', ['carrot', 'gajar'], 41, 0.9, 9.6, 0.2, { density: 0.6, densityVariance: 0.1, calorieVariance: 0.12 }),
  t('Spinach (cooked)', ['spinach', 'palak'], 23, 2.9, 3.6, 0.4, { density: 0.7, densityVariance: 0.15, calorieVariance: 0.15 }),
  t('Broccoli (cooked)', ['broccoli'], 35, 2.4, 7.2, 0.4, { density: 0.6, densityVariance: 0.12, calorieVariance: 0.15 }),
  t('Onion', ['onion', 'pyaz'], 40, 1.1, 9.3, 0.1, { pieceG: 110, pieceVariance: 0.25, calorieVariance: 0.15 }),
  t('Mixed salad leaves', ['lettuce', 'salad leaves'], 15, 1.4, 2.9, 0.2, { density: 0.3, densityVariance: 0.2, calorieVariance: 0.2 }),

  /* --------------------------------------------------------- condiments --- */
  t('Mayonnaise', ['mayonnaise', 'mayo'], 680, 1, 0.6, 75, { density: 0.95, densityVariance: 0.05 }),
  t('Tomato ketchup', ['ketchup', 'tomato sauce'], 100, 1.2, 26, 0.2, { density: 1.15, densityVariance: 0.05 }),
  t('Chutney (coconut)', ['coconut chutney', 'chutney'], 165, 3, 8, 14, { density: 0.9, densityVariance: 0.15, calorieVariance: 0.3 }),
  t('Pickle (oil-based)', ['pickle', 'achar'], 190, 1, 10, 16, { density: 0.9, densityVariance: 0.15, calorieVariance: 0.3 }),
  t('Raita', ['raita'], 60, 2.6, 5, 3.2, { density: 1.0, densityVariance: 0.1 }),
];
