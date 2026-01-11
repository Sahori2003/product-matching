import numpy as np
import pandas as pd
import re

# Unit replacements and lists
unit_replacements = {
    r'\b(mg\.?|milligram(s)?)\b': 'mg',
    r'\b(mcg\.?|μg|microgram(s)?)\b': 'mcg',
    r'\b(g\.?|gram(s)?)\b': 'g',
    r'\b(kg\.?|kilogram(s)?)\b': 'kg',
    r'\b(ml\.?|milliliter(s)?)\b': 'ml',
    r'\b(l\.?|liter(s)?)\b': 'l',
    r'\b(meq|mEq)\b': 'meq',
    r'\b(iu|u|unit(s)?)\b': 'iu',
    r'\b(percent|pct)\b': '%',
    r'\b(box|pack|pk)\b': 'pk'
}

unit = ['mg', 'spf', 'mcg', 'g', 'kg', 'ml', 'l', 'mg/ml', 'meq', 'iu', '%', 'pk']

FORMULATIONS = [
    'tablet', 'capsule', 'syrup', 'suspension', 'solution', 'cream', 'gel', 'lotion', 'paste', 'drops',
    'sachet', 'powder', 'granule', 'patch', 'spray', 'film', 'lozenge', 'injection',
    'ampoule', 'vial', 'softgel', 'inhaler', 'suppository', 'ovule', 'pessary', 'shampoo',
    'conditioner', 'serum', 'mask', 'oil', 'balm', 'soap', 'wash', 'condom', 'hand sanitizer',
    'milk', 'spry', 'salt','effervescent tablet', 'foam', 'wax', 'sanitizer', 'wipes', 
    'pad', 'diaper', 'toothbrush', 'toothpaste', 'thermometer', 'nebulizer', 'glucometer', 
    'razor', 'bottle', 'kit', 'device', 'makeup', 'lip makeup', 'eye makeup', 'nail product', 
    'makeup remover', 'cleaner', 'fragrance', 'medical accessory','strip'
]

formulation_replacements = {
    r'\b(tablet(s)?|tab(s)?|tables|t)\b': 'tablet',
    r'\b(capsule(s)?|cap(s)?|capsul)\b': 'capsule',
    r'\b(syrup)\b': 'syrup',
    r'\b(suspension|susp)\b': 'suspension',
    r'\b(solution)\b': 'solution',
    r'\b(cream|ointment|lipocream)\b': 'cream',
    r'\b(gel(s)?)\b': 'gel',
    r'\b(lotion)\b': 'lotion',
    r'\b(paste|toothpaste)\b': 'paste',
    r'\b(drop(s)?)\b': 'drops',
    r'\b(sach|sachet(s)?)\b': 'sachet',
    r'\b(powder)\b': 'powder',
    r'\b(granule(s)?)\b': 'granule',
    r'\b(patch)\b': 'patch',
    r'\b(spray)\b': 'spray',
    r'\b(film)\b': 'film',
    r'\b(lozenge(s)?)\b': 'lozenge',
    r'\b(injection)\b': 'injection',
    r'\b(ampoule)\b': 'ampoule',
    r'\b(vial(s)?)\b': 'vial',
    r'\b(softgel(s)?)\b': 'softgel',
    r'\b(inhaler)\b': 'inhaler',
    r'\b(suppository)\b': 'suppository',
    r'\b(ovule)\b': 'ovule',
    r'\b(pessary)\b': 'pessary',
    r'\b(shampoo)\b': 'shampoo',
    r'\b(conditioner)\b': 'conditioner',
    r'\b(serum)\b': 'serum',
    r'\b(mask)\b': 'mask',
    r'\b(oil|oilfusion)\b': 'oil',
    r'\b(balm)\b': 'balm',
    r'\b(soap)\b': 'soap',
    r'\b(wash|cleanser)\b': 'wash',
    r'\b(condom)\b': 'condom',
    r'\b(salt(s)?)\b': 'salt',
    r'\b(spry(s)?)\b': 'spry',
    r'\b(milk(s)?)\b': 'milk',

    r'\b(tablet(s)?|tab(s)?|tabs|t)\b': 'tablet',
    r'\b(capsule(s)?|cap(s)?|capsul|softgel(s)?|soft gels|caplet)\b': 'capsule',
    r'\b(syrup)\b': 'syrup',
    r'\b(suspension|susp)\b': 'suspension',
    r'\b(solution|soln)\b': 'solution',
    r'\b(drop(s)?|nasal drop(s)?|eye drop(s)?)\b': 'drops',
    r'\b(sachet(s)?)\b': 'sachet',
    r'\b(vial(s)?|ampoule(s)?|ampul(e)?|amp(s)?)\b': 'vial',
    r'\b(injection|injectable)\b': 'injection',
    r'\b(lozenge(s)?)\b': 'lozenge',
    r'\b(chewable tablet(s)?)\b': 'tablet',
    r'\b(effervescent)\b': 'effervescent tablet',
    r'\b(suppository|ovule|pessary)\b': 'suppository',
    r'\b(inhaler|nasal inhaler)\b': 'inhaler',

    # Topical / Skin Forms
    r'\b(gel(s)?|cooling gel|soothing gel|healing gel|anti-inflammatory gel|pain relief gel|joint gel|muscle gel|drinking water gel)\b': 'gel',
    r'\b(cream(s)?|ointment(s)?|lipocream|balm|e-ointment|antifungal cream|antibacterial cream|antiseptic cream|rash cream|eczema cream|hand cream|body cream|face cream|foot cream|moisturizing cream|massage cream)\b': 'cream',
    r'\b(lotion(s)?|spray lotion|after sun lotion|body lotion)\b': 'lotion',
    r'\b(spray(s)?|spray-on|aerosol(s)?|foam spray|nasal spray|body spray|deodorant|deo)\b': 'spray',
    r'\b(foam(s)?|mousse)\b': 'foam',
    r'\b(paste|toothpaste)\b': 'paste',
    r'\b(patch(es)?)\b': 'patch',
    r'\b(powder(s)?|tooth powder|whitening powder)\b': 'powder',
    r'\b(film)\b': 'film',

    # Hair Care
    r'\b(shampoo)\b': 'shampoo',
    r'\b(conditioner)\b': 'conditioner',
    r'\b(hair mask|mask)\b': 'mask',
    r'\b(hair oil|oilfusion|oil(s)?)\b': 'oil',
    r'\b(hair cream|hair lotion|hair serum|serum)\b': 'serum',
    r'\b(hair spray|tonic)\b': 'spray',
    r'\b(hair wax|hair clay|hair putty|hair pomade)\b': 'wax',

    # Body & Hygiene
    r'\b(soap|bar soap|liquid soap)\b': 'soap',
    r'\b(wash|cleanser|body wash|hand wash|vaginal wash|daily wash)\b': 'wash',
    r'\b(sanitizer|hand sanitizer)\b': 'sanitizer',
    r'\b(wipes|tissue(s)?)\b': 'wipes',
    r'\b(pad(s)?|napkin(s)?|panty liner(s)?|sanitary towel(s)?|menstrual cup|period panties)\b': 'pad',
    r'\b(diaper(s)?|under wear diaper)\b': 'diaper',
    r'\b(condom(s)?)\b': 'condom',

    # Devices / Tools
    r'\b(toothbrush|tooth brush|electric toothbrush|chargable toothbrush)\b': 'toothbrush',
    r'\b(toothpaste)\b': 'toothpaste',
    r'\b(thermometer)\b': 'thermometer',
    r'\b(nebulizer)\b': 'nebulizer',
    r'\b(glucometer)\b': 'glucometer',
    r'\b(razor(s)?|electric shaver)\b': 'razor',
    r'\b(bottle(s)?|feeding bottle|sterilizer)\b': 'bottle',
    r'\b(kit|first aid kit|makeup kit)\b': 'kit',
    r'\b(device|monitor|machine|ventilator|cpap machine|blood pressure monitor|defibrillator|sphygmomanometer|stethoscope)\b': 'device',

    # Makeup & Beauty
    r'\b(foundation|concealer|powder|blush|bronzer|highlighter|bb cream|cc cream|primer|cushion)\b': 'makeup',
    r'\b(lipstick|lip gloss|lip liner)\b': 'lip makeup',
    r'\b(eyeliner|eyeshadow|eye pencil|mascara)\b': 'eye makeup',
    r'\b(nail polish|nail remover|nail solution)\b': 'nail product',
    r'\b(makeup remover|setting spray)\b': 'makeup remover',

    # Home & Cleaning
    r'\b(cleaner|disinfectant|repellent|bleach|detergent|sanitizer|dish soap|dishwashing liquid|fabric softener|toilet cleaner|floor cleaner|air freshener|mosquito repellent|destroyer)\b': 'cleaner',

    # Miscellaneous
    r'\b(salt(s)?)\b': 'salt',
    r'\b(milk|body milk)\b': 'milk',
    r'\b(water|drinking water)\b': 'water',
    r'\b(ampule|ampul|ampoule)\b': 'ampoule',

    r'\b(perfume|fragrance|cologne|body mist)\b': 'fragrance',
    r'\b(bandage|plaster|gauze|brace|support)\b': 'medical accessory',
}

stopwords = ['the', 'a', 'an', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'with']



def clean_text(df, column_name):
    """Clean and normalize text in the specified DataFrame column"""

    # Remove Arabic characters
    df[column_name] = df[column_name].str.replace(r'[\u0600-\u06FF]+', ' ', regex=True)

    # Convert to lowercase and trim spaces
    df[column_name] = df[column_name].str.lower().str.strip()

    # Remove sequences of multiple asterisks (** or more)
    df[column_name] = df[column_name].str.replace(r'\*{2,}', '', regex=True)

    # Remove non-alphanumeric characters (keep: / + % *)
    df[column_name] = df[column_name].str.replace(r'[^\w\s\%\*]', ' ', regex=True)

    # Remove numbers with 5 digits or more (likely IDs)
    df[column_name] = df[column_name].str.replace(r'\b\d{5,}\b', ' ', regex=True)

    # Add space between numbers and letters (e.g., 20mg → 20 mg)
    df[column_name] = df[column_name].str.replace(r'(?<=\d)(?=[A-Za-z%])', ' ', regex=True)

    # Add space between letters and numbers (e.g., ml50 → ml 50)
    df[column_name] = df[column_name].str.replace(r'(?<=[A-Za-z%])(?=\d)', ' ', regex=True)

    # Remove values that are only numbers (e.g., "123" → "")
    df[column_name] = df[column_name].apply(lambda x: '' if str(x).replace(' ', '').isdigit() else x)

    # Apply unit replacements (e.g., "ml." → "ml")
    for pattern, repl in unit_replacements.items():
        df[column_name] = df[column_name].str.replace(pattern, repl, regex=True, flags=re.IGNORECASE)

    # Apply formulation replacements (e.g., "syp" → "syrup")
    for pattern, repl in formulation_replacements.items():
        df[column_name] = df[column_name].str.replace(pattern, repl, regex=True, flags=re.IGNORECASE)

    # Remove stopwords unless the word contains a number
    df[column_name] = df[column_name].apply(
        lambda text: ' '.join(
            word for word in str(text).split()
            if re.search(r'\d', word) or word not in stopwords
        )
    )

    # Remove spaces between numbers (e.g., "20 50" → "2050")
    df[column_name] = df[column_name].str.replace(r'(?<=\d)\s+(?=\d)', '', regex=True)

    # Remove spaces between number and single-letter unit (e.g., "50 m" → "50m")
    df[column_name] = df[column_name].str.replace(r'(?<=\d)\s+(?=[A-Za-z]\b)', '', regex=True)

    # Remove spaces between letter and number (reverse case)
    df[column_name] = df[column_name].str.replace(r'(?<=\b[A-Za-z])\s+(?=\d)', '', regex=True)

    # Normalize multiple spaces to a single space
    df[column_name] = df[column_name].str.replace(r'\s+', ' ', regex=True).str.strip()

    return df


df = pd.read_excel(r"C:/Users/raad2/Downloads/Product Matching/data/Alias_Review_Sample_1000.xlsx")
df = df[["alias", "SKU_Name"]]
df = clean_text(df, 'alias')
df = clean_text(df, 'SKU_Name')

df['alias'] = df['alias'].replace('', np.nan)
df['SKU_Name'] = df['SKU_Name'].replace('', np.nan)

df = df.dropna(subset=['alias', 'SKU_Name'])
df.to_excel(r"C:/Users/raad2/Downloads/Product Matching/data/Cleaned_data.xlsx")