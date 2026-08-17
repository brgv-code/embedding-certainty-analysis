"""
make_synthetic_dataset.py — writes synthetic_sentences.csv for the distribution/scatter plots.

Sentences are grouped into clearly distinct semantic clusters so the plots separate visibly.
The three "bank" groups are the key demo: same word, different meaning —
  bank_money      -> financial sense (should cluster with finance)
  bank_river      -> riverside sense (should cluster away from money)
  bank_ambiguous  -> no resolving context (should land between the two)
Plus far-apart topics (jordan facts, space, cooking) and a nonsense group that should scatter.

Run:  python make_synthetic_dataset.py   ->  synthetic_sentences.csv
Then: python plot_distributions.py --responses synthetic_sentences.csv
"""

import csv

GROUPS = {
    "bank_money": [
        "I deposited my paycheck at the bank and checked my savings account balance.",
        "The bank approved our mortgage after reviewing three years of tax returns.",
        "Interest rates set by the central bank affect every commercial loan.",
        "She waited in line at the bank to withdraw cash from the teller.",
        "Online banking lets me transfer funds between checking and savings instantly.",
        "The investment bank underwrote the company's initial public offering.",
        "My debit card was declined, so I called the bank's fraud department.",
        "They opened a joint account and applied for a small business loan.",
    ],
    "bank_river": [
        "We sat on the grassy bank of the river and watched the current drift by.",
        "The flood eroded the muddy bank and swept away the wooden footbridge.",
        "Fishermen lined the far bank casting their lines into the slow water.",
        "Reeds and willows grew thick along the steep bank of the stream.",
        "The canoe slid ashore onto the sandy bank below the waterfall.",
        "Otters burrow into the soft earth of the riverbank near the bend.",
        "At dawn a cold mist rose off the water and clung to the reedy bank.",
        "The trail follows the northern bank of the creek through the pine forest.",
    ],
    "bank_ambiguous": [
        "I am standing next to the bank.",
        "She walked slowly toward the bank.",
        "They met near the bank yesterday.",
        "He left something at the bank.",
        "We should go to the bank later.",
        "The bank was very quiet this morning.",
        "There is a bank around the corner.",
        "I will wait for you by the bank.",
    ],
    "jordan_facts": [
        "Jordan is a kingdom ruled by a monarch, not a president.",
        "King Hussein reigned over Jordan from 1952 until his death in 1999.",
        "Amman is the capital and largest city of Jordan.",
        "In 1985 Jordan was governed by King Hussein bin Talal.",
        "The Hashemite dynasty has ruled Jordan since its independence.",
        "King Abdullah II succeeded his father as Jordan's monarch in 1999.",
        "Petra, the ancient rock-carved city, lies in southern Jordan.",
        "Jordan shares long desert borders with Saudi Arabia and Iraq.",
    ],
    "space": [
        "Apollo 11 landed the first humans on the Moon in July 1969.",
        "The Saturn V rocket propelled the Apollo missions out of Earth's orbit.",
        "Apollo 20 was cancelled and never flew to the lunar surface.",
        "Astronauts collected lunar rock samples during the Apollo landings.",
        "The command module orbited the Moon while the lander descended.",
        "NASA ended the Apollo program after Apollo 17 in 1972.",
        "Powerful telescopes reveal distant galaxies billions of light years away.",
        "The spacecraft re-entered the atmosphere and splashed down in the Pacific.",
    ],
    "cooking": [
        "Whisk the eggs and sugar until the batter turns pale and fluffy.",
        "Simmer the tomato sauce with garlic, basil, and a pinch of salt.",
        "Knead the dough for ten minutes, then let it rise until doubled.",
        "Sear the steak on high heat and let it rest before slicing.",
        "Fold the flour gently into the meringue to keep it airy.",
        "Roast the vegetables with olive oil until the edges caramelize.",
        "Bring the stock to a boil, then add the noodles and vegetables.",
        "Grate fresh parmesan over the pasta just before serving.",
    ],
    "nonsense": [
        "Colorless green ideas sleep furiously beneath the humming xylophone.",
        "Banana sqrt purple the the running sideways lamp of cheese.",
        "Quantum toaster whispered violet equations to the sleepy staircase.",
        "The the of and running jumped seven o'clock umbrella backwards.",
        "Fluorescent pickle orbited the melancholy adjective on a Tuesday.",
        "Sputtering moonlight ate the rectangular silence with plastic verbs.",
        "Wobble the frantic teaspoon until the ceiling forgets its socks.",
        "Zigzag helium giggled where the number blue tastes of Thursday.",
    ],
}


def main(path="synthetic_sentences.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "model", "response"])
        for group, sentences in GROUPS.items():
            for i, s in enumerate(sentences):
                w.writerow([group, f"s{i}", s])
    n = sum(len(v) for v in GROUPS.values())
    print(f"Wrote {path}: {len(GROUPS)} groups, {n} sentences")


if __name__ == "__main__":
    main()
