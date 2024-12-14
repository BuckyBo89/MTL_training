import random


homoglyph_map = {
    'a': ['а', 'ᴀ', 'ɑ'],
    'b': ['Ь', 'ƅ', 'ɓ'],
    'c': ['ϲ', 'ᴄ', 'с'],
    'd': ['ԁ', 'ɗ', 'ժ'],
    'e': ['е', 'ҽ', '℮'],
    'f': ['ғ', 'ƒ', 'ſ'],
    'g': ['ɡ', 'ԍ', 'ɠ'],
    'h': ['һ', 'ɦ', 'ʜ'],
    'i': ['і', 'ɩ', 'ι'],
    'j': ['ј', 'ϳ', 'ʝ'],
    'k': ['κ', 'ᴋ', 'ƙ'],
    'l': ['ӏ', 'ⅼ', 'Ɩ'],
    'm': ['м', 'ᴍ', 'ɱ'],
    'n': ['ո', 'ɴ', 'ƞ'],
    'o': ['о', 'ɵ', 'σ'],
    'p': ['р', 'ρ', 'ƿ'],
    'q': ['ԛ', 'ʠ', 'զ'],
    'r': ['ɾ', 'г', 'ʀ'],
    's': ['ѕ', 'ʂ', 'ƨ'],
    't': ['τ', 'ţ', 'ƫ'],
    'u': ['υ', 'ս', 'ᴜ'],
    'v': ['ν', 'ѵ', 'ᴠ'],
    'w': ['ᴡ', 'ɯ', 'ԝ'],
    'x': ['х', 'ҳ', 'χ'],
    'y': ['у', 'ү', 'γ'],
    'z': ['ᴢ', 'ᴢ', 'ʐ'],
}

def apply_homoglyph_attack(text, attack_rate=0.3):
    attacked_text = []
    for char in text:
        if char.lower() in homoglyph_map and random.random() < attack_rate:

            homoglyph = random.choice(homoglyph_map[char.lower()])

            attacked_text.append(homoglyph.upper() if char.isupper() else homoglyph)
        else:
            attacked_text.append(char)
    return ''.join(attacked_text)

# 示例文本
original_text = "This is an example of homoglyph attack."
attacked_text = apply_homoglyph_attack(original_text, attack_rate=0.5)
print("Original text:", original_text)
print("Attacked text:", attacked_text)
