import json
import pandas as pd
from pandas import json_normalize

with open("C:/Users/raad2/Downloads/Product Matching/data/api_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Normalize
df = json_normalize(
    data["results"],
    record_path="results",
    meta=["alias"],
    sep="_"
)


names_list = [
    "ducray exomega control emollient cream 200 ml",
    "votum 10 mg tablet 28",
    "umatone 30 sachet",
    "norocarmena 0033 mg 21 tablet",
    "starville vitamin c serum 30 ml",
    "johnson s vita rich pomegranate brightening lotion 250 ml",
    "avent 618m night time blue silicone soother 2 scf 17662",
    "signal whitening paste 100 ml",
    "candicure 1 % cream 30g",
    "zomegipral 40 mg iv vial",
    "tetra glow sensitive areas whitening cream 50g",
    "femigiene nipple care cream 50 ml",
    "voltic d50 mg tablet 20",
    "farlin 0m standard neck bottle 120 ml",
    "unical 120 mg capsule 30",
    "rispons 1 mg ml oral solution 100 ml",
    "rebo zeal cream 30g",
    "no rash tube 100g",
    "ginexin f40 mg tablet 40",
    "colgate classic deep clean toothbrush medium",
    "tresemme biotin repair 7 shampoo 700 ml",
    "pravafen 40160 mg capsule 30",
    "oprexa 10 mg tablet 30",
    "aptamil 1 milk 400g",
    "sulfozinc 10 mg 5 ml syrup 80 ml",
    "tresemme perfectly un done shampoo 500 ml",
    "tobrason 03 % 01 % eye cream 5g",
    "zyrtec 01 % syrup 100 ml",
    "veradent sensitive toothbrush",
    "al ghad dhea ultimate 50 mg capsule 60",
    "parachute gold thick strong oil 100 ml",
    "aftamed forte mouth ulcer gel 8 ml",
    "seroxat cr 125 mg tablet 30",
    "gillette fusion 5 refillable men razor",
    "dibi face lift creator 31 eye contour gel 15 ml",
    "vichy ideal moisturising milk serum 200 ml",
    "oramin f capsule 30",
    "glysolid classic lotion 250 ml",
    "heliocare 360 spf 50 invisible sunblock spray 200 ml",
    "thiopental 05g vial 25"
]

repeated = []
for name in names_list:
    repeated.extend([name] * 30)

df["my_product"] = repeated[:len(df)]

df.to_excel("C:/Users/raad2/Downloads/Product Matching/data/output_api_test.xlsx", index=False, engine="openpyxl")