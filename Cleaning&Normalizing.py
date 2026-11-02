"""
خط لوله پیش‌پردازش متن فارسی (Persian NLP Pipeline)
شامل سه مرحله:
۱. پاک‌سازی (Cleaning): حذف تگ‌های HTML، لینک‌ها، کاراکترهای نامرئی و ناشناس‌سازی اطلاعات شخصی (PII)
۲. نرمال‌سازی (Normalization): یکدست‌سازی یونیکد (NFKC)، حروف و ارقام، اصلاح نیم‌فاصله و کاهش تکرار حروف
۳. توکنایزیشن (Tokenization): شکستن متن به جملات (Sentence Tokenization) و واژه‌ها (Word Tokenization)
"""

import argparse
from dataclasses import dataclass
import html
import re
import unicodedata
from typing import Any, Dict, List

# کاراکتر استاندارد نیم‌فاصله (Zero-Width Non-Joiner)
ZWNJ = "\u200c"


@dataclass
class PipelineConfig:
    """تنظیمات خط لوله پیش‌پردازش وابسته به مسئله (Task-Dependent)"""

    task_mode: str = "general"  # 'general', 'sentiment', 'topic', 'ner'
    use_placeholders: bool = True  # جایگزینی لینک و عدد با <URL> و <NUM>
    anonymize_pii: bool = True  # ناشناس‌سازی شماره موبایل و ایمیل


class PersianNLPPipeline:

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self._init_translation_table()
        self._init_regex_patterns()

    def _init_translation_table(self) -> None:
        """ساخت جدول نگاشت نویسه‌ها برای یکدست‌سازی حروف عربی، ارقام و علائم"""
        char_map = {
            # تبدیل حروف عربی به معادل استاندارد فارسی
            "\u064A": "\u06CC",  # ي -> ی
            "\u0649": "\u06CC",  # ى (الف مقصوره) -> ی
            "\u0643": "\u06A9",  # ك -> ک
            "\u0629": "\u0647",  # ة -> ه
            "\u0623": "\u0627",  # أ -> ا
            "\u0625": "\u0627",  # إ -> ا
            "\u0671": "\u0627",  # ٱ -> ا
            # یکدست‌سازی علائم نگارشی انگلیسی به فارسی
            "?": "؟",
            ",": "،",
            ";": "؛",
            "“": "«",
            "”": "»",
        }
        # تبدیل ارقام عربی و انگلیسی به ارقام فارسی (۰ تا ۹)
        for i in range(10):
            persian_digit = chr(0x06F0 + i)
            char_map[chr(0x0660 + i)] = persian_digit  # ارقام عربی
            char_map[str(i)] = persian_digit  # ارقام انگلیسی

        self.translation_table = str.maketrans(char_map)

    def _init_regex_patterns(self) -> None:
        """کامپایل الگوهای Regex برای پاک‌سازی، نرمال‌سازی و توکنایزیشن"""
        # ۱. الگوهای مرحله پاک‌سازی
        self.html_tag_re = re.compile(r"<[^>]+>")
        # حذف کاراکترهای نامرئی و کنترلی (با حفظ عامدانه نیم‌فاصله \u200c و پیونددهنده ایموجی \u200d)
        self.invisibles_re = re.compile(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200e\u200f\u202a-\u202e\u2060\u00ad\ufeff]"
        )
        self.url_re = re.compile(r"https?://\S+|www\.\S+")
        self.email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
        self.phone_re = re.compile(r"\b(?:09|\+989|۰۹)[0-9۰-۹]{9}\b")
        self.user_re = re.compile(r"@\w+")
        self.hashtag_re = re.compile(r"#(\w+)")
        self.emoji_re = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+"
        )

        # ۲. الگوهای مرحله نرمال‌سازی
        # اعراب (فتحه، کسره، تنوین، تشدید) و خط کشیده/تطویل (\u0640)
        self.diacritics_re = re.compile(r"[\u0640\u064B-\u065F\u0670]")
        self.number_re = re.compile(r"\b[0-9۰-۹]+(?:\.[0-9۰-۹]+)?\b")

        # ۳. الگوهای مرحله توکنایزیشن
        # الف) جمله بندی: تشخیص مرز جمله (. ! ؟ …) بدون شکستن اعداد اعشاری مانند ۳.۱۴
        self.sentence_split_re = re.compile(
            r"(?<![0-9۰-۹])([.!?؟…]+[\s»)\]]*)(?![0-9۰-۹])"
        )
        # ب) توکن‌سازی واژه:
        # به ترتیب: ۱. تگ‌های جایگزین مثل <URL> | ۲. اعداد اعشاری/صحیح | ۳. واژه‌های فارسی و انگلیسی (همراه با نیم‌فاصله) | ۴. علائم نگارشی و ایموجی‌ها
        self.word_tokenize_re = re.compile(
            rf"<[A-Z]+>|[0-9۰-۹]+(?:\.[0-9۰-۹]+)?|[\w\u0600-\u06FF{ZWNJ}]+|[^\s\w\u0600-\u06FF{ZWNJ}]"
        )

    # =========================================================================
    # مرحله ۱: پاک‌سازی متن (Text Cleaning)
    # =========================================================================
    def clean(self, text: str) -> str:
        """حذف نویزهای ساختاری، تگ‌های HTML، لینک‌ها و اطلاعات شخصی"""
        # تبدیل Entityهای HTML مانند &nbsp; به کاراکتر واقعی
        text = html.unescape(text)
        # حذف تگ‌های HTML
        text = self.html_tag_re.sub(" ", text)
        # حذف کاراکترهای نامرئی مخرب
        text = self.invisibles_re.sub("", text)

        # جایگزینی لینک، ایمیل، شماره تماس و منشن (قبل از اعداد تا ساختارشان نشکند)
        keep = self.config.use_placeholders
        text = self.url_re.sub(" <URL> " if keep else " ", text)

        if self.config.anonymize_pii:
            text = self.email_re.sub(" <EMAIL> " if keep else " ", text)
            text = self.phone_re.sub(" <PHONE> " if keep else " ", text)

        text = self.user_re.sub(" <USER> " if keep else " ", text)

        # حذف علامت # از هشتگ و تبدیل زیرخط (_) به فاصله
        text = self.hashtag_re.sub(lambda m: m.group(1).replace("_", " "), text)

        # در طبقه‌بندی موضوعی ایموجی‌ها حذف می‌شوند
        if self.config.task_mode == "topic":
            text = self.emoji_re.sub(" ", text)

        return text

    # =========================================================================
    # مرحله ۲: نرمال‌سازی متن (Text Normalization)
    # =========================================================================
    def normalize(self, text: str) -> str:
        """نرمال‌سازی چهارسطحی: نویسه، قالب، واژه و فاصله/نیم‌فاصله"""
        # ۱. سطح نویسه: استاندارد NFKC یونیکد + نگاشت ک/ی و ارقام + حذف اعراب و کشیده
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(self.translation_table)
        text = self.diacritics_re.sub("", text)

        # ۲. سطح قالب: جایگزینی اعداد با <NUM> (مگر در تسک NER که تاریخ و عدد مهم است)
        if self.config.task_mode != "ner":
            if self.config.use_placeholders:
                text = self.number_re.sub("<NUM>", text)
            elif self.config.task_mode == "topic":
                text = self.number_re.sub(" ", text)

        # ۳. سطح واژه: کاهش تکرار حروف (سلاااام -> سلام) و علائم نگارشی (!!!! -> !)
        if self.config.task_mode != "sentiment":
            text = re.sub(r"(.)\1{2,}", r"\1", text)
        else:
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        text = re.sub(r"\.{3,}", "…", text)
        text = re.sub(r"([!؟؛،])\1+", r"\1", text)

        # ۴. سطح فاصله و نیم‌فاصله (ZWNJ)
        text = re.sub(rf"{ZWNJ}{{2,}}", ZWNJ, text)
        text = re.sub(rf"\s*{ZWNJ}\s*", ZWNJ, text)

        # اتصال پیشوندهای فعلی «می» و «نمی» با نیم‌فاصله
        text = re.sub(
            r"\b(ن?می)\s+(?!که\b|را\b|هم\b|و\b)", r"\1" + ZWNJ, text
        )

        # اتصال پسوندهای جمع، صفت برتر و ضمایر ملکی با نیم‌فاصله
        suffixes = r"(ها|های|هایی|تر|ترین|ام|ات|اش|مان|تان|شان)"
        text = re.sub(rf"(?<=[آ-ی])\s+{suffixes}\b", ZWNJ + r"\1", text)

        # تنظیم فاصله استاندارد اطراف علائم نگارشی
        text = re.sub(r"\s+([،؛؟!.…»)\]])", r"\1", text)
        text = re.sub(r"([«(\[])\s+", r"\1", text)
        text = re.sub(r"([،؛؟!])(?=[^\s»)\]])", r"\1 ", text)

        if self.config.task_mode == "topic":
            text = re.sub(r"[!؟؛،.…«»\"'()\[\]]", " ", text)

        return re.sub(r"\s+", " ", text).strip()

    # =========================================================================
    # مرحله ۳: توکنایزیشن (Tokenization - سطح جمله و سطح واژه)
    # =========================================================================
    def tokenize_sentences(self, text: str) -> List[str]:
        """جمله‌بندی (Sentence Tokenization) با حفظ علائم پایانی و اعداد اعشاری"""
        parts = self.sentence_split_re.split(text)
        sentences = []
        current = ""

        for part in parts:
            if not part:
                continue
            if self.sentence_split_re.match(part):
                current += part
                sentences.append(current.strip())
                current = ""
            else:
                current += part

        if current.strip():
            sentences.append(current.strip())

        return [s for s in sentences if s]

    def tokenize_words(self, text: str) -> List[str]:
        """
        توکن‌سازی واژگانی (Word Tokenization):
        - حفظ یکپارچگی کلمات مرکب دارای نیم‌فاصله (مانند «کتاب‌ها» یا «می‌روم»)
        - جداسازی علائم نگارشی چسبیده به کلمه (مانند «خوبه؟» -> ['خوبه', '؟'])
        - حفظ تگ‌های جایگزین (مانند '<URL>' و '<NUM>') به عنوان یک توکن واحد
        """
        return self.word_tokenize_re.findall(text)

    # =========================================================================
    # اجرای کامل پایپ‌لاین و استخراج آمار
    # =========================================================================
    def process(self, raw_text: str) -> Dict[str, Any]:
        """اجرای ترتیبی پاک‌سازی -> نرمال‌سازی -> توکنایزیشن"""
        cleaned = self.clean(raw_text)
        normalized = self.normalize(cleaned)
        sentences = self.tokenize_sentences(normalized)
        word_tokens = self.tokenize_words(normalized)
        tokens_per_sentence = [self.tokenize_words(s) for s in sentences]

        # محاسبه آمار واژگان (Token vs Type/Vocabulary Size)
        raw_tokens = raw_text.split()
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "normalized_text": normalized,
            "sentences": sentences,
            "word_tokens": word_tokens,
            "tokens_per_sentence": tokens_per_sentence,
            "stats": {
                "raw_token_count": len(raw_tokens),
                "raw_vocab_size": len(set(raw_tokens)),
                "final_token_count": len(word_tokens),
                "final_vocab_size": len(set(word_tokens)),
                "sentence_count": len(sentences),
                "is_idempotent": (
                    normalized == self.normalize(self.clean(normalized))
                ),
            },
        }


def print_pipeline_output(result: Dict[str, Any], mode: str) -> None:
    """نمایش خروجی مرحله‌به‌مرحله در ترمینال برای ارائه کلاسی"""
    BOLD, CYAN, GREEN, YELLOW, MAGENTA, RESET = (
        "\033[1m",
        "\033[96m",
        "\033[92m",
        "\033[93m",
        "\033[95m",
        "\033[0m",
    )

    print(f"\n{BOLD}{CYAN}=== ۱. متن خام ورودی (Raw Text) ==={RESET}")
    print(result["raw_text"])

    print(
        f"\n{BOLD}{GREEN}=== ۲. متن پس از پاک‌سازی و نرمال‌سازی (Mode: {mode}) ==={RESET}"
    )
    print(result["normalized_text"])

    print(
        f"\n{BOLD}{MAGENTA}=== ۳. توکنایزیشن در سطح جمله (Sentence Tokenization: {result['stats']['sentence_count']} جمله) ==={RESET}"
    )
    for i, sent in enumerate(result["sentences"], 1):
        print(f"  [جمله {i}] {sent}")

    print(
        f"\n{BOLD}{YELLOW}=== ۴. توکنایزیشن در سطح واژه (Word Tokenization: {result['stats']['final_token_count']} توکن) ==={RESET}"
    )
    print(result["word_tokens"])

    st = result["stats"]
    print(f"\n{BOLD}{CYAN}=== ۵. آمار متن (Token / Type) و تست کیفیت ==={RESET}")
    print(
        f"• تعداد کل توکن‌ها (Tokens)     : {st['raw_token_count']} (با split ساده)  -->  {st['final_token_count']} (توکنایزر دقیق)"
    )
    print(
        f"• اندازه واژگان یکتا (Types/V) : {st['raw_vocab_size']}  -->  {st['final_vocab_size']}"
    )
    print(
        f"• تست Idempotency              : {'✅ موفق' if st['is_idempotent'] else '❌ ناموفق'}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Persian NLP Cleaning, Normalization & Tokenization Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["general", "sentiment", "topic", "ner"],
        default="general",
        help="حالت پیش‌پردازش بر اساس نوع مسئله (پیش‌فرض: general)",
    )
    parser.add_argument(
        "--text", type=str, help="متن دلخواه برای پردازش از طریق خط فرمان"
    )
    args = parser.parse_args()

    config = PipelineConfig(task_mode=args.mode)
    pipeline = PersianNLPPipeline(config)

    sample_text = args.text or (
        "<p>سلاااام به همگیییی 😍!! حال شما چطوره ? </p>\n"
        "این يك متن آزمايشي در حوزه #پردازش_زبان_طبیعی است. "
        "من امروز به دانِشگاه مي‌روم و كِتاب ها را مطالعه می کنم! "
        "کلمه كِتاب با کتاب یا کتاب ها پس از نرمال سازی یکدست می‌شوند... "
        "نرخ رشد مدل ۳.۱۴ درصد گزارش شده است. "
        "برای ارتباط با @ali_ai به https://example.com/nlp?id=12 یا ایمیل info@uni.ac.ir پیام دهید "
        "یا با شماره 09123456789 تماس بگیرید. "
        "كشششـــــيده نویسی و نویسه\u200eهای نامرئی\u200b نیز برطرف شوند."
    )

    output = pipeline.process(sample_text)
    print_pipeline_output(output, args.mode)