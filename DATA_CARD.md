# Dataset Card (`DATA_CARD.md`)

## 1. Summary Statistics
- **Total valid image-JSON pairs:** 979
- **Images missing JSON:** 0
- **Corrupt/unreadable files:** 0
- **Damaged images (has >=1 damage annotation):** 294 (30.0%)
- **Undamaged images (only car parts annotated):** 685 (70.0%)
- **Derived Severity Breakdown:** {'undamaged': np.int64(685), 'minor': np.int64(111), 'severe': np.int64(94), 'moderate': np.int64(89)}
- **Median Resolution:** 1280x720
- **Near-duplicate pairs found (pHash distance <= 8):** 6 pairs across 973 unique car groups

## 2. Damage Class Distribution
| Damage Class | Category | Count |
|---|---|---|
| `scratch` | Minor | 789 |
| `bumper crack` | Moderate | 169 |
| `panel gap` | Moderate | 122 |
| `replace` | Severe | 102 |
| `dent minor` | Minor | 80 |
| `bumper replace` | Severe | 58 |
| `dent medium` | Moderate | 56 |
| `crack` | Moderate | 43 |
| `bumper dent` | Moderate | 31 |
| `bent` | Moderate | 18 |
| `dent major` | Severe | 17 |

## 3. Car Part Class Distribution
| Car Part | Count |
|---|---|
| `rim` | 673 |
| `tyre` | 663 |
| `rear bumper` | 587 |
| `number plate` | 576 |
| `rear windshield` | 506 |
| `boot` | 497 |
| `left orvm` | 402 |
| `left tail lamp` | 390 |
| `right orvm` | 363 |
| `bonnet` | 362 |
| `right tail lamp` | 345 |
| `front bumper` | 341 |
| `front windshield` | 320 |
| `front grill` | 317 |
| `left headlamp` | 250 |
| `right headlamp` | 248 |
| `left quarter panel` | 242 |
| `left front door` | 228 |
| `left rear door` | 218 |
| `right front door` | 199 |
| `right quarter panel` | 197 |
| `right rear door` | 194 |
| `left fender` | 122 |
| `right fender` | 110 |
| `left running board` | 44 |
| `right running board` | 36 |
| `right indicator` | 10 |
| `left indicator` | 9 |