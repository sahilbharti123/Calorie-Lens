/**
 * Reviewed on-device food catalog.
 *
 * Every entry is a USDA FoodData Central record (SR Legacy or FNDDS survey
 * data) verified against the published per-100 g values; the FDC ID is shown
 * to the user as the source. Piece weights and densities are documented
 * portion assumptions with an explicit variance that widens the shown range.
 *
 * The Python engine (calorie_engine/catalog.py) mirrors this list — keep the
 * two in sync when adding foods.
 */

export type FoodReference = {
  name: string;
  aliases: string[];
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fdcId: number;
  pieceG?: number;
  pieceVariance?: number;
  density?: number;
  densityVariance?: number;
};

export const FOODS: FoodReference[] = [
  // ------------------------------------------------------- pulses & legumes
  { name: 'Cooked kidney beans', aliases: ['kidney beans', 'kidney bean', 'rajma'], calories: 127, protein: 8.67, carbs: 22.8, fat: 0.5, fdcId: 175194, density: 0.75, densityVariance: 0.18 },
  { name: 'Cooked lentils', aliases: ['lentils', 'lentil', 'dhal', 'daal', 'dal'], calories: 116, protein: 9.02, carbs: 20.1, fat: 0.38, fdcId: 172421, density: 0.75, densityVariance: 0.18 },
  { name: 'Boiled chickpeas', aliases: ['chickpeas', 'chickpea', 'chana', 'chole', 'garbanzo'], calories: 164, protein: 8.86, carbs: 27.4, fat: 2.59, fdcId: 173757, density: 0.66, densityVariance: 0.15 },
  { name: 'Tofu (regular)', aliases: ['tofu'], calories: 76, protein: 8.1, carbs: 1.9, fat: 4.8, fdcId: 172476, density: 1.04, densityVariance: 0.08 },

  // ------------------------------------------------------------------ grains
  { name: 'Cooked white rice', aliases: ['white rice', 'plain rice', 'cooked rice', 'basmati rice', 'rice'], calories: 130, protein: 2.69, carbs: 28.2, fat: 0.28, fdcId: 168878, density: 0.79, densityVariance: 0.12 },
  { name: 'Cooked brown rice', aliases: ['brown rice'], calories: 123, protein: 2.74, carbs: 25.6, fat: 0.97, fdcId: 169704, density: 0.78, densityVariance: 0.12 },
  { name: 'Rolled oats (dry)', aliases: ['rolled oats', 'oatmeal', 'oats'], calories: 379, protein: 13.2, carbs: 67.7, fat: 6.5, fdcId: 173904, density: 0.35, densityVariance: 0.15 },
  { name: 'Whole-wheat roti', aliases: ['chapati', 'chappati', 'phulka', 'roti'], calories: 299, protein: 7.85, carbs: 46.1, fat: 9.2, fdcId: 174075, pieceG: 40, pieceVariance: 0.2 },
  { name: 'Whole-wheat paratha', aliases: ['paratha', 'parantha'], calories: 326, protein: 6.4, carbs: 45.4, fat: 13.2, fdcId: 174076, pieceG: 80, pieceVariance: 0.25 },
  { name: 'Whole-wheat bread', aliases: ['whole wheat bread', 'brown bread', 'bread', 'toast'], calories: 252, protein: 12.4, carbs: 42.7, fat: 3.5, fdcId: 172688, pieceG: 28, pieceVariance: 0.18 },
  { name: 'Idli', aliases: ['idli', 'idlis'], calories: 128, protein: 6.36, carbs: 24.98, fat: 0.35, fdcId: 2708346, pieceG: 50, pieceVariance: 0.22 },
  { name: 'Plain dosa', aliases: ['plain dosa', 'dosa'], calories: 210, protein: 5.7, carbs: 37.04, fat: 4.05, fdcId: 2708347, pieceG: 100, pieceVariance: 0.3 },

  // --------------------------------------------------------- protein sources
  { name: 'Roasted chicken breast', aliases: ['chicken breast', 'grilled chicken', 'roasted chicken', 'chicken'], calories: 165, protein: 31, carbs: 0, fat: 3.57, fdcId: 171477 },
  { name: 'Boiled egg', aliases: ['hard boiled egg', 'boiled egg', 'anda', 'eggs', 'egg'], calories: 155, protein: 12.6, carbs: 1.12, fat: 10.6, fdcId: 173424, pieceG: 50, pieceVariance: 0.12 },
  { name: 'Egg white', aliases: ['egg whites', 'egg white'], calories: 52, protein: 10.9, carbs: 0.73, fat: 0.17, fdcId: 172183, pieceG: 33, pieceVariance: 0.12, density: 1.03, densityVariance: 0.05 },
  { name: 'Paneer (fresh cheese)', aliases: ['paneer', 'cottage cheese cube'], calories: 310, protein: 20.4, carbs: 2.5, fat: 24.3, fdcId: 172224 },

  // -------------------------------------------------------------- dairy etc.
  { name: 'Whole milk', aliases: ['whole milk', 'full fat milk', 'doodh', 'milk'], calories: 60, protein: 3.27, carbs: 4.63, fat: 3.2, fdcId: 746782, density: 1.03, densityVariance: 0.03 },
  { name: 'Plain whole-milk yogurt', aliases: ['plain yogurt', 'yogurt', 'curd', 'dahi'], calories: 61, protein: 3.47, carbs: 4.66, fat: 3.25, fdcId: 171284, density: 1.03, densityVariance: 0.06 },
  { name: 'Smooth peanut butter', aliases: ['peanut butter'], calories: 598, protein: 22.2, carbs: 22.3, fat: 51.4, fdcId: 172470, density: 1.07, densityVariance: 0.08 },
  { name: 'Butter', aliases: ['makhan', 'makkhan', 'butter'], calories: 717, protein: 0.85, carbs: 0.06, fat: 81.1, fdcId: 173430, density: 0.96, densityVariance: 0.03, pieceG: 14, pieceVariance: 0.25 },
  { name: 'Olive oil', aliases: ['olive oil', 'cooking oil', 'oil'], calories: 884, protein: 0, carbs: 0, fat: 100, fdcId: 171413, density: 0.91, densityVariance: 0.02 },
  { name: 'Almonds', aliases: ['almonds', 'almond', 'badam'], calories: 579, protein: 21.15, carbs: 21.55, fat: 49.93, fdcId: 170567, pieceG: 1.2, pieceVariance: 0.15, density: 0.55, densityVariance: 0.15 },

  // ------------------------------------------------------ vegetables & fruit
  { name: 'Boiled potato', aliases: ['boiled potato', 'potato', 'potatoes', 'aloo'], calories: 87, protein: 1.9, carbs: 20.1, fat: 0.1, fdcId: 170438, pieceG: 170, pieceVariance: 0.25 },
  { name: 'Baked sweet potato', aliases: ['sweet potato', 'shakarkandi'], calories: 90, protein: 2, carbs: 20.7, fat: 0.15, fdcId: 168483, pieceG: 150, pieceVariance: 0.25 },
  { name: 'Banana', aliases: ['banana', 'kela'], calories: 89, protein: 1.09, carbs: 22.8, fat: 0.33, fdcId: 173944, pieceG: 118, pieceVariance: 0.18 },
  { name: 'Apple', aliases: ['apple', 'seb'], calories: 52, protein: 0.26, carbs: 13.8, fat: 0.17, fdcId: 171688, pieceG: 182, pieceVariance: 0.2 },

  // -------------------------------------------------- mixed dishes (FNDDS)
  { name: 'Samosa', aliases: ['samosa', 'samosas'], calories: 309, protein: 5.1, carbs: 33.1, fat: 17.4, fdcId: 2344214, pieceG: 100, pieceVariance: 0.3 },
  { name: 'Biryani with meat', aliases: ['chicken biryani', 'mutton biryani', 'biryani'], calories: 144, protein: 8.4, carbs: 12.1, fat: 6.8, fdcId: 2341916, density: 0.85, densityVariance: 0.18 },
  { name: 'Chicken curry', aliases: ['chicken curry', 'chicken gravy'], calories: 82, protein: 5.7, carbs: 6.7, fat: 3.9, fdcId: 2341861, density: 0.95, densityVariance: 0.15 },
];
