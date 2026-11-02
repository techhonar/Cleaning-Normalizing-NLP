# Persian NLP Cleaning & Normalization Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Standard%20Library)-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

A zero-dependency, modular Python preprocessor engineered for Persian text cleaning and four-level normalization. It directly resolves common Persian script challenges, including Arabic-Persian character unification, invisible directional characters, Zero-Width Non-Joiners (ZWNJ / نیم‌فاصله), repetitive character compression, and PII anonymization.

---

## Architecture Overview

The pipeline executes a deterministic, two-phase process designed to minimize vocabulary explosion and out-of-vocabulary (OOV) tokens while preventing information leakage:

```text
Raw Input Text
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  Phase 1: Structural Cleaning                          │
│  • HTML Entity decoding & tag stripping                │
│  • Invisible & ASCII control character removal         │
│  • PII Anonymization: Phone (09xx) & Email masks       │
│  • Pattern extraction: URLs, mentions, and hashtags    │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2: Four-Level Normalization                     │
│  1. Character Level: Unicode NFKC & Arabic-Persian map │
│  2. Format Level: Canonical digits & token masking     │
│  3. Word Level: Repeated char reduction & punctuation  │
│  4. Spacing Level: Deterministic ZWNJ & affixes        │
└────────────────────────────────────────────────────────┘
      │
      ▼
Standardized Output + Statistical Audit & Idempotency Check
```

### 1. Four Levels of Normalization
* **Character Level:** Converts Arabic presentation glyphs (e.g., `ي` $\rightarrow$ `ی`, `ك` $\rightarrow$ `ک`), unifies variants of alef (`إ/أ/ٱ` $\rightarrow$ `ا`), and strips diacritics/harakat (`[\u064B-\u065F]`) and tatweel کشیده (`\u0640`).
* **Format Level:** Translates English and Eastern-Arabic digits to standard Persian digits (`۰`–`۹`), then converts targeted numbers to `<NUM>` tokens according to the task context.
* **Word Level:** Compresses unnatural character repetitions (e.g., `سلاااام` $\rightarrow$ `سلام`), normalizes ellipsis (`...` $\rightarrow$ `…`), and condenses duplicate punctuation (`!!!!` $\rightarrow$ `!`).
* **Spacing Level:** Enforces strict ZWNJ (`\u200c`) binding for verbal prefixes (`می‌`/`نمی‌`) and nominal suffixes (`ها`/`تر`/`ترین`), while removing orphaned ZWNJs and cleaning whitespace around Persian punctuation.

### 2. Task-Dependent Strategy
Different NLP downstream applications treat noise differently:
* **`general`:** Balanced tokenization with standard placeholders (`<URL>`, `<NUM>`, `<EMAIL>`, `<PHONE>`).
* **`sentiment`:** Preserves sentiment signals, keeping emojis and emotional punctuation (`!`, `؟`) intact while allowing up to two character repeats.
* **`topic`:** Aggressive vocabulary pruning, stripping all emojis, punctuation, and numerical expressions to focus purely on lexical topics.
* **`ner`:** Retains explicit numbers, temporal expressions, and date structures unmasked for named entity boundary detection.

---

## Local Setup & Execution Guide

This project requires **Python 3.9 or higher** and uses only Python standard modules (`re`, `unicodedata`, `html`, `argparse`, `dataclasses`). No third-party packages need to be installed.

### Step 1: Clone the Repository

Clone the repository and enter the directory across any terminal:
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

Because filenames containing the `&` symbol are reserved in terminal interpreters (representing background tasks on POSIX shells or command chaining on Windows), wrap the filename in quotation marks:

#### 1. Default Benchmark Run
Executes the full pipeline against an all-inclusive benchmark test string:
* **Linux / macOS:**
  ```bash
  python "Cleaning&Normalizing.py"
  ```
* **Windows (PowerShell / CMD):**
  ```cmd
  python "Cleaning&Normalizing.py"
  ```

#### 2. Task-Specific Modes
Select downstream task configurations via the `--mode` flag:
```bash
# Sentiment analysis mode (emojis & exclamation marks preserved)
python "Cleaning&Normalizing.py" --mode sentiment

# Topic classification mode (emojis, numbers, and punctuation stripped)
python "Cleaning&Normalizing.py" --mode topic

# Named entity recognition mode (dates, numbers, and symbols preserved)
python "Cleaning&Normalizing.py" --mode ner
```

#### 3. Custom Input String
Pass any arbitrary Persian sentence directly via the `--text` argument:
```bash
python "Cleaning&Normalizing.py" --text "سلام، كِتاب ها در https://site.ir با قیمت 12000 تومان موجود است."
```

#### 4. Pulling Upstream Changes
To pull updates later into your local repository without overwriting your environment:
```bash
git pull origin main
```

#### 5. Deactivation
When finished, exit the virtual environment:
```bash
deactivate
```

---

## Command-Line Arguments Reference

| Option | Type | Choices | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--mode` | `str` | `general`, `sentiment`, `topic`, `ner` | `general` | Downstream task-dependent processing strategy. |
| `--text` | `str` | *Any string* | Built-in sample | Custom Persian text payload to normalize directly. |
| `-h`, `--help`| `flag`| — | — | Displays the command-line help interface. |

---

## Verification & Statistical Quality Metrics

The preprocessor outputs a structured quality audit on every execution:

```text
=== ۱. متن خام ورودی (Raw Text) ===
<p>سلاااام به همگیییی 😍!! حال شما چطوره ? </p>
این يك متن آزمايشي است؛ کتاب ها و مقاله هاي علمی در حوزه #یادگیری_ماشین و #هوش_مصنوعی بسیار بزرگ تر شده‌اند. من امروز ساعت 14:30 به دانِشگاه مي‌روم و كِتاب ها را مطالعه می کنم... 
برای ارتباط با @ali_ai به آدرس https://example.com/nlp?id=45 یا ایمیل info@university.ac.ir پیام دهید یا با شماره 09123456789 تماس بگیرید. كشششـــــيده نویسی و نویسه‌های نامرئی نیز حذف می‌شوند.

=== ۲. متن تمیز و نرمال‌شده (Mode: general) ===
سلام به همگی 😍! حال شما چطوره؟ این یک متن آزمایشی است؛ کتاب‌ها و مقاله‌های علمی در حوزه یادگیری ماشین و هوش مصنوعی بسیار بزرگ‌تر شده‌اند. من امروز ساعت <NUM>:<NUM> به دانشگاه می‌روم و کتاب‌ها را مطالعه می‌کنم… برای ارتباط با <USER> به آدرس <URL> یا ایمیل <EMAIL> پیام دهید یا با شماره <PHONE> تماس بگیرید. کشیده نویسی و نویسه‌های نامرئی نیز حذف می‌شوند.

=== ۳. گزارش آماری و تست کیفیت خط لوله ===
• تعداد کاراکترها   : 472  -->  384
• تعداد کل توکن‌ها  : 73   -->  61
• اندازه واژگان (V) : 66   -->  54 (کاهش نویز واژگان: 18.2%)
• نسبت حروف فارسی   : 91.2%
• تست Idempotency   : ✅ موفق (تغییرناپذیر در اجرای مجدد)
```

### Quality Assurance Criteria
1. **Idempotency Invariance:** Enforces $f(f(x)) = f(x)$. Running an already processed output through the normalization pipeline a second time results in an identical string with zero character mutation.
2. **Vocabulary Pruning ($V$):** Unifies fragmented surface forms (e.g., merging `كِتاب`, `کتاب`, and `کتاب ها` into a single vocabulary type `کتاب‌ها`), reducing sparse dimensions before vectorization.
3. **Lexical Isolation:** Validates that trailing affixes like `ها` or `تر` are linked by standard ZWNJ (`\u200c`) rather than ASCII whitespace or full concatenation, preventing subword tokenizers from fragmenting common roots.

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