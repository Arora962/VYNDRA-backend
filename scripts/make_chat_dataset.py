#!/usr/bin/env python3
"""Create deterministic, balanced synthetic chat and realistic-review datasets."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FOODS = ["biryani", "dosa", "idli", "paneer", "pasta", "pizza", "dal", "samosa", "salad", "noodles", "poha", "paratha", "rajma", "chole", "upma", "curd rice", "khichdi", "pav bhaji", "sandwich", "sushi"]
INTENT = ["I'm eating {food} rn 😋", "should I order {food}?", "just cooked {food} lol", "aaj {food} khayi", "who wants {food} with me", "photo of my {food} incoming 📸", "craving {food} so bad", "had {food} for lunch yaar", "is {food} healthy?", "{food} for dinner tonight"]
DISCUSSION = ["what do you think about {food}?", "best place for {food} near college?", "my mom makes amazing {food}", "is homemade {food} better?", "tell me a recipe for {food}", "why is {food} so popular", "rating {food} out of ten", "we discussed {food} in class", "{food} is overrated tbh", "anyone tried vegan {food}?"]
NOT_FOOD = ["exam was a piece of cake", "I'm starving for attention", "this homework is a nightmare", "meeting at {time} tomorrow", "send me the lecture notes", "my laptop battery is dead", "that movie was so good", "I have a headache today", "the bus is running late", "please call me after class", "I'm cooked for tomorrow's exam", "this task is a hot mess", "happy birthday 🎉", "did you submit the assignment?", "my phone needs charging"]
SLANG = ["", " lol", " yaar", " tbh", " fr", " 😭", " omg", " pls", " bro"]
CONTEXTS = ["", " today", " rn", " after class", " later", " this week", " btw", " honestly", " I guess", " not gonna lie", " somehow", " for real", " as usual", " haha", " I think", " what now", " if possible", " please", " seriously", " just saying", " maybe", " though", " for once", " at least", " apparently", " in my opinion", " no joke", " anyway", " lowkey", " highkey", " on campus", " at home", " with friends", " before class", " after lunch", " this evening", " tomorrow", " last night", " tmrw", " rn pls", " lol ok"]

def unique_examples(rng: random.Random, label: str, target: int) -> list[str]:
    rows, seen = [], set()
    templates = INTENT if label == "food_intent" else DISCUSSION if label == "food_discussion" else NOT_FOOD
    attempts = 0
    while len(rows) < target:
        attempts += 1
        if attempts > target * 100:
            raise RuntimeError(f"Could only create {len(rows)} unique messages for {label}; increase template/context variety.")
        food = rng.choice(FOODS)
        time = rng.choice(["9am", "noon", "6pm", "after class", "tomorrow"])
        text = rng.choice(templates).format(food=food, time=time)
        if rng.random() < 0.55:
            text += rng.choice(CONTEXTS)
        if rng.random() < 0.28:
            text += rng.choice(SLANG)
        if rng.random() < 0.12:
            text = text.replace("biryani", "biriyani").replace("dinner", "diner")
        if rng.random() < 0.12:
            text = text.replace(" ", "  ", 1)
        text = text.strip()
        if text not in seen:
            seen.add(text); rows.append(text)
    return rows

def build_dataset() -> pd.DataFrame:
    rng = random.Random(SEED)
    per_class = 1200
    records = []
    for label in ("food_intent", "food_discussion", "not_food"):
        for text in unique_examples(rng, label, per_class):
            sentiment = "positive" if any(x in text.lower() for x in ["amazing", "good", "healthy", "😋", "love", "craving"]) else "negative" if any(x in text.lower() for x in ["nightmare", "dead", "headache", "overrated", "😭", "starving"]) else "neutral"
            records.append({"text": text, "label": label, "sentiment_label": sentiment})
    rng.shuffle(records)
    df = pd.DataFrame(records)
    # Split after unique text generation, so text cannot leak across partitions.
    df["split"] = "train"
    for label in df.label.unique():
        indices = df.index[df.label.eq(label)].tolist(); rng.shuffle(indices)
        n = len(indices); n_val = int(n * .15); n_test = int(n * .15)
        df.loc[indices[:n_val], "split"] = "validation"
        df.loc[indices[n_val:n_val+n_test], "split"] = "test"
    assert not df.text.duplicated().any()
    return df.sort_values(["split", "label", "text"]).reset_index(drop=True)

def realistic_review() -> pd.DataFrame:
    """Return 150 individually authored realistic examples for human label review."""
    intent = [
      ("I'm grabbing biryani from the place near the metro, want anything?", "positive", "biryani"),
      ("aaj dosa khayi and honestly it slapped 😭", "positive", "dosa"),
      ("Can someone save me two idlis? running late lol", "neutral", "idli"),
      ("I packed paneer wraps for the train ride", "neutral", "paneer"),
      ("ordering pizza now, drop your toppings fast", "neutral", "pizza"),
      ("Mumma made rajma chawal, I'm sending a pic", "positive", "rajma"),
      ("I'm eating poha before the lab, anyone want some?", "neutral", "poha"),
      ("craving chole bhature so bad rn", "positive", "chole"),
      ("made a tiny bowl of khichdi, finally something warm", "positive", "khichdi"),
      ("should I get sushi or just make noodles at home?", "neutral", "sushi"),
      ("bro the samosa filling is actually amazing today", "positive", "samosa"),
      ("I have leftover paratha if anyone's hungry", "neutral", "paratha"),
      ("curd rice for lunch because this heat is unreal", "neutral", "curd rice"),
      ("just ordered dal makhani and naan, no regrets", "positive", "dal makhani"),
      ("I forgot lunch so I'm getting a sandwich rn", "neutral", "sandwich"),
      ("kya khau yaar, maybe masala dosa?", "neutral", "dosa"),
      ("made pasta with way too much cheese lol", "positive", "pasta"),
      ("can you add one more plate of pav bhaji to the order", "neutral", "pav bhaji"),
      ("my dinner is aloo paratha and dahi tonight", "neutral", "paratha"),
      ("sending you my homemade salad, tell me if it looks okay", "positive", "salad"),
      ("I need chai and pakoda immediately after this meeting", "neutral", "chai"),
      ("had biriyani for lunch, spelling aside it was fire", "positive", "biryani"),
      ("we're getting momos after class, come along?", "positive", "momos"),
      ("I'm making upma, do you want a portion?", "neutral", "upma"),
      ("the dosa batter fermented perfectly this time 😌", "positive", "dosa"),
      ("ordering chappati and dal for my flatmates", "neutral", "chapati"),
      ("I could demolish a plate of pani puri rn", "positive", "pani puri"),
      ("lunch break = leftover biryani, elite behaviour", "positive", "biryani"),
      ("got a bowl of noodles, will send the menu photo", "neutral", "noodles"),
      ("I made tofu bhurji because paneer was finished", "neutral", "paneer"),
      ("anyone want the last slice of pizza?", "neutral", "pizza"),
      ("I'm trying that new dosa stall by the library today", "neutral", "dosa"),
      ("finally eating something, just got a veg thali", "positive", "thali"),
      ("aaj ghar pe aloo gobi bani hai, full comfort food", "positive", "aloo gobi"),
      ("I packed idly and chutney, don't let me forget it", "neutral", "idli"),
      ("can we split a paneer tikka roll?", "neutral", "paneer"),
      ("my biryani order says delivered but it's not here 😭", "negative", "biryani"),
      ("made a smoothie and toast, breakfast is sorted", "positive", "toast"),
      ("I'm picking up gulab jamun for dessert", "positive", "gulab jamun"),
      ("need something quick, maybe poha from the canteen", "neutral", "poha"),
      ("I tried making dosa and it stuck to the pan lol", "negative", "dosa"),
      ("please save me some rajma, I'll be there in ten", "neutral", "rajma"),
      ("just had a really good bowl of ramen", "positive", "ramen"),
      ("I want a photo of the menu before we order", "neutral", ""),
      ("anyone else having chole for dinner?", "neutral", "chole"),
      ("my lunchbox has paratha and pickle today", "neutral", "paratha"),
      ("I am making dal khichdi because I'm exhausted", "neutral", "khichdi"),
      ("send me the link to order that paneer pizza", "neutral", "paneer"),
      ("this samosa is cold but I'm still eating it", "negative", "samosa"),
      ("got sushi for the first time, kinda obsessed", "positive", "sushi"),
    ]
    discussion = [
      ("Is the dosa batter supposed to smell a little sour?", "neutral", "dosa"),
      ("My nani's rajma recipe has black cardamom in it", "positive", "rajma"),
      ("Which café around campus does decent paneer rolls?", "neutral", "paneer"),
      ("I think homemade biryani tastes better the next day", "positive", "biryani"),
      ("Does anyone know why my idli turns out dense?", "neutral", "idli"),
      ("The debate is whether chole needs tea bags for colour", "neutral", "chole"),
      ("What spices do you put in aloo gobi?", "neutral", "aloo gobi"),
      ("My friend says curd rice is only a summer dish, thoughts?", "neutral", "curd rice"),
      ("Is there a good gluten-free alternative to paratha?", "neutral", "paratha"),
      ("The new pizza place uses a really thin crust", "positive", "pizza"),
      ("Why does restaurant dal makhani taste so smoky?", "neutral", "dal makhani"),
      ("I need a reliable recipe for soft chapati", "neutral", "chapati"),
      ("Anyone tried making pani puri water without mint?", "neutral", "pani puri"),
      ("The canteen's poha is better than it has any right to be", "positive", "poha"),
      ("Would you put peanuts in upma or is that weird?", "neutral", "upma"),
      ("My family argues about pineapple on pizza every week", "neutral", "pizza"),
      ("I read that cooling rice changes its resistant starch", "neutral", "rice"),
      ("What's your favourite filling for a dosa?", "positive", "dosa"),
      ("Is paneer tikka usually marinated overnight?", "neutral", "paneer"),
      ("I can't tell if this biriyani recipe has too much oil", "negative", "biryani"),
      ("Does anyone have a good vegan version of kheer?", "neutral", "kheer"),
      ("The difference between chaat masala brands is wild", "neutral", "chaat"),
      ("Would you count a samosa as a snack or a full meal?", "neutral", "samosa"),
      ("I want to learn how to make restaurant-style noodles", "neutral", "noodles"),
      ("Why do some idli recipes add poha to the batter?", "neutral", "idli"),
      ("The best pav bhaji has a little beetroot, apparently", "positive", "pav bhaji"),
      ("Is tofu actually a fair substitute for paneer in curry?", "neutral", "paneer"),
      ("My aunt roasts the spices before grinding them for biryani", "positive", "biryani"),
      ("Which oil gives paratha the best flaky layers?", "neutral", "paratha"),
      ("I saw a recipe for chocolate dosa and I'm confused", "neutral", "dosa"),
      ("Do you wash basmati rice until the water runs clear?", "neutral", "rice"),
      ("What is the least messy way to pack chole bhature?", "neutral", "chole"),
      ("The texture of fresh gulab jamun is completely different", "positive", "gulab jamun"),
      ("My meal prep salad gets soggy by lunchtime, help", "negative", "salad"),
      ("Can I freeze leftover dal without ruining the texture?", "neutral", "dal"),
      ("Aaj kal everyone is rating the same momo place", "neutral", "momos"),
      ("I think lemon rice needs more curry leaves", "neutral", "rice"),
      ("What's the difference between dosa and neer dosa batter?", "neutral", "dosa"),
      ("Is the paneer in packaged wraps usually pre-fried?", "neutral", "paneer"),
      ("The biryani-vs-pulao argument will never end", "neutral", "biryani"),
      ("Would roasted makhana work as a crunchy salad topping?", "neutral", "salad"),
      ("Why does restaurant sambar taste sweeter than mine?", "neutral", "sambar"),
      ("Anyone know if this ramen broth contains shellfish?", "neutral", "ramen"),
      ("The new menu has millet dosa and ragi idli", "positive", "dosa"),
      ("Is reheated pizza better in a pan or the oven?", "neutral", "pizza"),
      ("My recipe says add hing to rajma, is that common?", "neutral", "rajma"),
      ("The cafeteria renamed khichdi as a wellness bowl lol", "neutral", "khichdi"),
      ("How long can cooked paneer curry stay in the fridge?", "neutral", "paneer"),
      ("Does anyone have a low-spice chole recipe?", "neutral", "chole"),
      ("The best part of thali is trying a little of everything", "positive", "thali"),
    ]
    not_food = [
      ("That exam was a piece of cake, I finished early", "positive", ""),
      ("I'm starving for attention, not actual food lol", "negative", ""),
      ("This assignment is cooking my brain", "negative", ""),
      ("I'm cooked for tomorrow's viva 😭", "negative", ""),
      ("Please send the slides before the lecture starts", "neutral", ""),
      ("The new update broke my laptop audio", "negative", ""),
      ("I have a meeting at half past three", "neutral", ""),
      ("That movie was food for thought, honestly", "neutral", ""),
      ("The lecture went over my head completely", "negative", ""),
      ("My phone battery is at two percent again", "negative", ""),
      ("Can you review my resume tonight?", "neutral", ""),
      ("The traffic near the station is ridiculous", "negative", ""),
      ("I'm hungry for a promotion at work", "neutral", ""),
      ("She roasted me in front of the whole class", "negative", ""),
      ("I need a break from staring at this screen", "negative", ""),
      ("That bug was easy to squash, finally", "positive", ""),
      ("The professor gave us a killer question", "neutral", ""),
      ("I'm binge-watching that series tonight", "positive", ""),
      ("My train got delayed by forty minutes", "negative", ""),
      ("Can someone explain this calculus step?", "neutral", ""),
      ("The app crashed the second I opened it", "negative", ""),
      ("I have butterflies before the presentation", "neutral", ""),
      ("That was a sweet gesture from the team", "positive", ""),
      ("The discussion got spicy after the results", "neutral", ""),
      ("I'm craving a vacation, not dinner", "neutral", ""),
      ("Our group chat is absolutely chaotic today", "neutral", ""),
      ("This code is giving me a headache", "negative", ""),
      ("I need to charge my earbuds before leaving", "neutral", ""),
      ("The deadline moved to Friday morning", "neutral", ""),
      ("He has a lot on his plate this semester", "neutral", ""),
      ("My brain is fried after three back-to-back labs", "negative", ""),
      ("That was a bitter defeat for our team", "negative", ""),
      ("We need to break the ice at the workshop", "neutral", ""),
      ("I love this song, it's on repeat", "positive", ""),
      ("The printer refuses to connect to Wi-Fi", "negative", ""),
      ("I'm thirsty for knowledge, send reading links", "neutral", ""),
      ("Her feedback was a bit salty but useful", "neutral", ""),
      ("The presentation template looks clean now", "positive", ""),
      ("I need to sleep before the early train", "neutral", ""),
      ("This task is a hot mess, please help", "negative", ""),
      ("My calendar is packed until next Tuesday", "neutral", ""),
      ("That joke was cheesy but I laughed", "positive", ""),
      ("I can't focus because the neighbours are drilling", "negative", ""),
      ("The exam paper was surprisingly light", "positive", ""),
      ("I have a lot of tabs open and no clue why", "neutral", ""),
      ("The meeting got heated over the budget", "negative", ""),
      ("Please ping me when the build finishes", "neutral", ""),
      ("I'm dying to know the final score", "neutral", ""),
      ("The new keyboard feels nice to type on", "positive", ""),
      ("I need a little space to think this through", "neutral", ""),
    ]
    records = []
    for label, examples in (("food_intent", intent), ("food_discussion", discussion), ("not_food", not_food)):
        for text, sentiment, food in examples:
            records.append({"text": text, "label": label, "sentiment_label": sentiment, "expected_food": food, "review_status": "needs_human_review"})
    frame = pd.DataFrame(records)
    if len(frame) < 150 or frame.text.duplicated().any():
        raise RuntimeError("The hand-authored review set must contain at least 150 unique messages.")
    return frame

def main():
    global SEED
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=SEED); args = parser.parse_args()
    SEED = args.seed
    out = ROOT / "data" / "processed" / "chat_intelligence"; out.mkdir(parents=True, exist_ok=True)
    train = build_dataset(); train.to_csv(out / "chat_messages_synthetic.csv", index=False)
    realistic_review().to_csv(out / "chat_messages_handwritten_review.csv", index=False)
    mention = realistic_review()[["text", "expected_food"]].rename(columns={"expected_food":"food"}).copy()
    # Include common misspellings for the mention extractor; expected labels remain canonical.
    misspellings = {"biryani":"biriyani", "paneer":"panner", "noodles":"noodls", "samosa":"samosaa", "paratha":"parata", "sandwich":"sandwitch", "pizza":"piza", "rajma":"rajmaa", "khichdi":"khichri", "dosa":"dosaa", "idli":"idly", "poha":"pohaa", "sushi":"sushii", "salad":"saladd", "pasta":"pastaa"}
    for idx in range(0, len(mention), 4):
        expected = str(mention.loc[idx, "food"]).strip().lower()
        if expected in misspellings:
            mention.loc[idx, "text"] = str(mention.loc[idx, "text"]).replace(expected, misspellings[expected])
    mention.to_csv(out / "mention_review_template.csv", index=False)
    print(f"Wrote {len(train)} synthetic messages; labels={train.label.value_counts().to_dict()}; split={train.split.value_counts().to_dict()}")
    print(f"Wrote {len(realistic_review())} hand-authored review messages (must be reviewed by a human).")
    print(f"Unique texts: {train.text.nunique()} / {len(train)}")
if __name__ == "__main__": main()
