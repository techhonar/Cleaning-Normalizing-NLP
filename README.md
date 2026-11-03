# Persian NLP Cleaning & Normalization Pipeline

[

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)

](https://www.python.org/)
[

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

](https://opensource.org/licenses/MIT)
[

![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Standard%20Library)

-brightgreen.svg)]()
[

![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)

]()

A zero-dependency, modular Python preprocessor for Persian text: cleaning, four-level normalization, and sentence/word tokenization. It resolves common Persian script problems, including Arabic-Persian character unification, invisible directional characters, Zero-Width Non-Joiners (ZWNJ / نیم‌فاصله), repeated-character compression, and PII anonymization.

---

## Architecture Overview

The pipeline is deterministic and runs in three phases. It is designed to shrink vocabulary noise and to be **idempotent** (running it on its own output changes nothing).

```text
Raw Input Text
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  Phase 1: Structural Cleaning                          │
│  • HTML entity decoding & tag stripping                │
│  • Invisible/control character & diacritic removal     │
│  • PII masking: phone (09xx / +98 / 0098) & email      │
│  • URL, @mention and #hashtag handling                 │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2: Four-Level Normalization                     │
│  1. Character: Unicode NFKC & Arabic→Persian map       │
│  2. Format: canonical digits & <NUM> masking           │
│  3. Word: repeated-letter reduction                    │
│  4. Spacing: ZWNJ rules, affixes & punctuation spacing │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  Phase 3: Tokenization                                 │
│  • Sentence tokenization (keeps decimals like ۳.۱۴)    │
│  • Word tokenization (ZWNJ words and <TAGS> stay whole)│
└────────────────────────────────────────────────────────┘
      │
      ▼
Standardized Output + Token/Type Statistics + Idempotency Check
```

### 1. Four Levels of Normalization
* **Character Level:** Converts Arabic glyphs (`ي` → `ی`, `ك` → `ک`), unifies alef variants (`إ/أ/ٱ` → `ا`), and strips diacritics/harakat (`[\u064B-\u065F]`) and tatweel (`\u0640`).
* **Format Level:** Converts English and Arabic-Indic digits to Persian digits (`۰`–`۹`), then replaces numbers with `<NUM>` depending on the task mode.
* **Word Level:** Compresses repeated *letters* (`سلاااام` → `سلام`; digits and punctuation are never compressed), turns `...` into `…`, and collapses duplicate punctuation (`!!!!` → `!`).
* **Spacing Level:** Joins verbal prefixes (`می‌`/`نمی‌`) and suffixes (`ها`/`تر`/`ترین`/…) with ZWNJ (`\u200c`), removes orphaned ZWNJs, and normalizes whitespace around Persian punctuation.

### 2. Task-Dependent Modes (`--mode`)

All modes share the same cleaning, PII masking, character unification, and ZWNJ rules. They differ only in the points below:

| Behavior | `general` | `sentiment` | `topic` | `ner` |
| :--- | :--- | :--- | :--- | :--- |
| Repeated letters | reduced to 1 | reduced to **2** (keeps emphasis) | reduced to 1 | reduced to 1 |
| Emojis | kept | kept | **removed** | kept |
| Punctuation | kept (duplicates collapsed) | kept (duplicates collapsed) | **removed** | kept (duplicates collapsed) |
| Numbers | `<NUM>` | `<NUM>` | `<NUM>` (removed if `use_placeholders=False`) | **kept** (as Persian digits) |
| URL / email / phone / @mention | `<URL>` `<EMAIL>` `<PHONE>` `<USER>` | same | same | same |
| Best for | general-purpose NLP | sentiment analysis | topic classification | named entity recognition |

> In `topic` mode, sentences are split **before** punctuation is removed, so sentence tokenization still works.

#### Before / After Examples

| Input | `general` | `sentiment` | `topic` | `ner` |
| :--- | :--- | :--- | :--- | :--- |
| `سلاااام به همگیییی 😍!!` | `سلام به همگی 😍!` | `سلاام به همگیی 😍!` | `سلام به همگی` | `سلام به همگی 😍!` |
| `نرخ رشد ۳.۱۴ درصد و قیمت 12000 تومان است...` | `نرخ رشد <NUM> درصد و قیمت <NUM> تومان است…` | `نرخ رشد <NUM> درصد و قیمت <NUM> تومان است…` | `نرخ رشد <NUM> درصد و قیمت <NUM> تومان است` | `نرخ رشد ۳.۱۴ درصد و قیمت ۱۲۰۰۰ تومان است…` |
| `تماس: 09123456789 یا info@uni.ac.ir` | `تماس: <PHONE> یا <EMAIL>` | `تماس: <PHONE> یا <EMAIL>` | `تماس: <PHONE> یا <EMAIL>` | `تماس: <PHONE> یا <EMAIL>` |
| `او مي‌رود و كِتاب ها را می خواند` | `او می‌رود و کتاب‌ها را می‌خواند` | *(identical)* | *(identical)* | *(identical)* |

---

## Local Setup & Execution Guide

This project requires **Python 3.9 or higher** and uses only standard modules (`re`, `unicodedata`, `html`, `argparse`, `dataclasses`, `typing`). No third-party packages are needed.

### Step 1: Clone the Repository

```bash
git clone https://github.com/techhonar/Cleaning-Normalizing-NLP.git
cd Cleaning-Normalizing-NLP
```

---

### Step 2: Set Up and Activate the Virtual Environment

#### On Linux (Arch, Ubuntu, Debian, Fedora)
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Verify activation (should point to .venv)
which python
```

#### On macOS
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Verify activation
which python
```

#### On Windows

* **Using PowerShell:**
  ```powershell
  # Create the virtual environment
  python -m venv .venv

  # If script execution is restricted, temporarily bypass for the current process:
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

  # Activate the virtual environment
  .\.venv\Scripts\Activate.ps1

  # Verify activation
  where.exe python
  ```

* **Using Command Prompt (`cmd.exe`):**
  ```cmd
  :: Create the virtual environment
  python -m venv .venv

  :: Activate the virtual environment
  .\.venv\Scripts\activate.bat

  :: Verify activation
  where python
  ```

---

### Step 3: Run the Application

Filenames containing `&` are special in terminal interpreters (background tasks on POSIX shells, command chaining on Windows), so wrap the filename in quotation marks:

#### 1. Default Benchmark Run
Runs the full pipeline on a built-in test string (HTML, emoji, repeated letters, diacritics, URL, email, phone, mention, hashtag, invisible characters):
```bash
python "Cleaning&Normalizing.py"
```

#### 2. Task-Specific Modes
```bash
# Sentiment analysis (emojis kept, repeats reduced to 2)
python "Cleaning&Normalizing.py" --mode sentiment

# Topic classification (emojis and punctuation removed)
python "Cleaning&Normalizing.py" --mode topic

# Named entity recognition (numbers and dates kept)
python "Cleaning&Normalizing.py" --mode ner
```

#### 3. Custom Input String
```bash
python "Cleaning&Normalizing.py" --text "سلام، كِتاب ها در https://site.ir با قیمت 12000 تومان موجود است."
```

#### 4. Pulling Upstream Changes
```bash
git pull origin main
```

#### 5. Deactivation
```bash
deactivate
```

---

## Command-Line Arguments Reference

| Option | Type | Choices | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--mode` | `str` | `general`, `sentiment`, `topic`, `ner` | `general` | Downstream task-dependent processing strategy. |
| `--text` | `str` | *Any string* | Built-in sample | Custom Persian text to process. |
| `-h`, `--help`| `flag`| — | — | Displays the help message. |

---

## Python API

Two extra options exist only in the API (not on the command line):

| `PipelineConfig` field | Default | Effect |
| :--- | :--- | :--- |
| `task_mode` | `"general"` | One of `general`, `sentiment`, `topic`, `ner`. Any other value raises `ValueError`. |
| `use_placeholders` | `True` | `True`: replace URL/email/phone/mention/number with `<URL>`, `<EMAIL>`, … `False`: remove them (numbers are kept, except in `topic`, which removes them). |
| `anonymize_pii` | `True` | `False`: leave emails and phone numbers untouched. URLs and mentions are still masked. |

Because the file name contains `&`, load it with `importlib` (or rename it, for example to `cleaning_normalizing.py`, and use a normal `import`):

```python
import importlib.util

spec = importlib.util.spec_from_file_location("cn", "Cleaning&Normalizing.py")
cn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cn)

pipeline = cn.PersianNLPPipeline(cn.PipelineConfig(task_mode="ner", anonymize_pii=False))
result = pipeline.process("سلام، تماس: 09123456789")

print(result["normalized_text"])   # clean + normalized string
print(result["sentences"])         # list of sentences
print(result["word_tokens"])       # list of tokens
print(result["stats"])             # token/type counts, is_idempotent, ...
```

---

## Verification & Statistical Quality Metrics

Every run prints the raw text, the normalized text, sentence and word tokens, and a statistics block. Real output of the default `general` mode on a benchmark text:

```text
=== ۱. متن خام ورودی (Raw Text) ===
<p>سلاااام به همگیییی 😍!! حال شما چطوره ? </p>
این يك متن آزمايشي است؛ کتاب ها و مقاله هاي علمی در حوزه #یادگیری_ماشین و #هوش_مصنوعی بسیار بزرگ تر شده‌اند. من امروز ساعت 14:30 به دانِشگاه مي‌روم و كِتاب ها را مطالعه می کنم... 
برای ارتباط با @ali_ai به آدرس https://example.com/nlp?id=45 یا ایمیل info@university.ac.ir پیام دهید یا با شماره 09123456789 تماس بگیرید. كشششـــــيده نویسی و نویسه‌های نامرئی نیز حذف می‌شوند.

=== ۲. متن پس از پاک‌سازی و نرمال‌سازی (Mode: general) ===
سلام به همگی 😍! حال شما چطوره؟ این یک متن آزمایشی است؛ کتاب‌ها و مقاله‌های علمی در حوزه یادگیری ماشین و هوش مصنوعی بسیار بزرگ‌تر شده‌اند. من امروز ساعت <NUM>:<NUM> به دانشگاه می‌روم و کتاب‌ها را مطالعه می‌کنم… برای ارتباط با <USER> به آدرس <URL> یا ایمیل <EMAIL> پیام دهید یا با شماره <PHONE> تماس بگیرید. کشیده نویسی و نویسه‌های نامرئی نیز حذف می‌شوند.

=== ۳. توکنایزیشن در سطح جمله (Sentence Tokenization: 6 جمله) ===
  [جمله 1] سلام به همگی 😍!
  [جمله 2] حال شما چطوره؟
  [جمله 3] این یک متن آزمایشی است؛ کتاب‌ها و مقاله‌های علمی در حوزه یادگیری ماشین و هوش مصنوعی بسیار بزرگ‌تر شده‌اند.
  [جمله 4] من امروز ساعت <NUM>:<NUM> به دانشگاه می‌روم و کتاب‌ها را مطالعه می‌کنم…
  [جمله 5] برای ارتباط با <USER> به آدرس <URL> یا ایمیل <EMAIL> پیام دهید یا با شماره <PHONE> تماس بگیرید.
  [جمله 6] کشیده نویسی و نویسه‌های نامرئی نیز حذف می‌شوند.

=== ۴. توکنایزیشن در سطح واژه (Word Tokenization: 73 توکن) ===
['سلام', 'به', 'همگی', '😍', '!', 'حال', 'شما', 'چطوره', '؟', ...]

=== ۵. آمار متن (Token / Type) و تست کیفیت ===
• تعداد کل توکن‌ها (Tokens)     : 69 (با split ساده)  -->  73 (توکنایزر دقیق)
• اندازه واژگان یکتا (Types/V) : 61  -->  62
• تست Idempotency              : ✅ موفق
```

> **Why do the token counts go up?** The tokenizer splits punctuation (`چطوره؟` → `چطوره`, `؟`) and emoji into separate tokens, while a plain `split()` leaves them attached to words. The gain is cleaner, reusable tokens, not a smaller count.

### Quality Assurance Criteria
1. **Idempotency:** Enforces `f(f(x)) = f(x)`. Running the cleaning and normalization again on an already processed output returns an identical string. The check is reported on every run (`is_idempotent`) and holds in all four modes. The rules were also stress-tested with 48,000 randomized adversarial inputs across every mode and config combination.
2. **Surface-form unification:** Variants such as `كِتاب` and `کتاب` collapse into one form, and `کتاب ها` becomes `کتاب‌ها` (ZWNJ-joined), which reduces sparse duplicate types before vectorization.
3. **Lexical isolation:** Suffixes like `ها` or `تر` are joined with a ZWNJ (`\u200c`), not an ASCII space and not full concatenation, so subword tokenizers don't fragment common roots.

### Known Limitations
* The suffix rule (`ام`, `ات`, `اش`, `مان`, `تان`, `شان`, …) is a heuristic and may join a rare standalone word to the previous word.
* Only HTML entities terminated by `;` are decoded (`&amp;` yes, bare `&amp` no).
* In `topic` mode with `anonymize_pii=False`, an email whose domain is purely numeric (for example `x@ali3.14`) is not idempotent. This was the only failure among the 48,000 random stress inputs.

---

## Changelog (Bug Fixes)

* **Idempotency:** `<NUM>`, `<URL>`, … placeholders were removed as HTML tags on a second pass. HTML tag matching now skips placeholders and no longer treats `۳ < ۵ > ۲` as a tag.
* **Repeated characters:** compression no longer runs on digits and punctuation. Previously `۱۰۰۰` became `۱۰` and `...` became `.` (so `…` was never produced).
* **Word tokenizer:** Persian punctuation is no longer glued to words (`خوبه؟` → `خوبه`, `؟`).
* **Phone numbers:** `+98…`, `0098…`, and Arabic-Indic digits are now detected.
* **Mentions:** `@` inside an email is no longer treated as a mention when `anonymize_pii=False`.
* **Topic mode:** sentence tokenization works again (punctuation was removed before splitting, so there was always one sentence).
* **Sentence splitter:** no longer breaks on `example.com` or decimals.
* **ZWNJ:** no dangling ZWNJ after `می` before punctuation or placeholders.
* **Config:** the mutable default argument is gone, and an invalid `task_mode` raises `ValueError`.
* **Smaller fixes:** trailing punctuation is no longer swallowed by URL/email patterns, and `؟!` is no longer split into `؟ !`.

---

## Directory Layout

```
Cleaning-Normalizing-NLP/
├── .gitignore               # Ignores .venv, cache, and editor configs
├── Cleaning&Normalizing.py  # Core preprocessor implementation
├── README.md                # Technical documentation and setup guide
└── requirements.txt         # Environment declaration and Python runtime notes
```

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software for educational, research, or commercial purposes.
