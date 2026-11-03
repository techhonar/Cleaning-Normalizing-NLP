"""
خط لوله پیش‌پردازش متن فارسی (Persian NLP Pipeline)
شامل سه مرحله:
۱. پاک‌سازی (Cleaning): حذف تگ‌های HTML، لینک‌ها، کاراکترهای نامرئی و ناشناس‌سازی اطلاعات شخصی (PII)
۲. نرمال‌سازی (Normalization): یکدست‌سازی یونیکد (NFKC)، حروف و ارقام، اصلاح نیم‌فاصله و کاهش تکرار حروف
۳. توکنایزیشن (Tokenization): شکستن متن به جملات (Sentence Tokenization) و واژه‌ها (Word Tokenization)

حالت‌های پردازش (--mode):
  general   : حالت عمومی؛ تکرار حروف به یکی، عدد -> <NUM>، ایموجی حفظ می‌شود
  sentiment : تحلیل احساس؛ تکرار حروف به «دو تا» کاهش می‌یابد (تاکید حفظ شود)، ایموجی حفظ می‌شود
  topic     : طبقه‌بندی موضوعی؛ ایموجی و علائم نگارشی حذف می‌شوند
  ner       : شناسایی موجودیت‌ها؛ اعداد و تاریخ‌ها دست‌نخورده می‌مانند
"""

import argparse
import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# ثابت‌ها
# ---------------------------------------------------------------------------

# کاراکتر استاندارد نیم‌فاصله (Zero-Width Non-Joiner)
ZWNJ = "\u200c"

# حالت‌های مجاز پردازش
VALID_MODES = ("general", "sentiment", "topic", "ner")

# نام تگ‌های جایگزین؛ این‌ها نباید به‌عنوان تگ HTML حذف شوند
PLACEHOLDER_NAMES = ("URL", "EMAIL", "PHONE", "USER", "NUM")

# بازه ارقام انگلیسی، فارسی و عربی (داخل کلاس نویسه‌ای regex)
DIGITS = "0-9۰-۹٠-٩"

# «هر حرف» (فارسی/انگلیسی)؛ بدون رقم و زیرخط (برخلاف \w که رقم و _ را هم شامل می‌شود)
LETTER = r"[^\W\d_]"

# فقط حروف پایه فارسی/عربی (بدون علائم نگارشی عربی مثل ٪ که در بازه «آ-ی» هم قرار می‌گیرند)
FA_LETTER = r"[\u0621-\u063A\u0641-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06CC]"


@dataclass
class PipelineConfig:
    """تنظیمات خط لوله پیش‌پردازش وابسته به مسئله (Task-Dependent)"""

    task_mode: str = "general"  # 'general', 'sentiment', 'topic', 'ner'
    use_placeholders: bool = True  # جایگزینی لینک و عدد با <URL> و <NUM>
    anonymize_pii: bool = True  # ناشناس‌سازی شماره موبایل و ایمیل

    def __post_init__(self) -> None:
        # اعتبارسنجی: حالت اشتباه نباید بی‌صدا مثل general رفتار کند
        if self.task_mode not in VALID_MODES:
            raise ValueError(
                f"task_mode باید یکی از {VALID_MODES} باشد، نه {self.task_mode!r}"
            )


class PersianNLPPipeline:

    def __init__(self, config: Optional[PipelineConfig] = None):
        # آرگومان پیش‌فرض mutable (PipelineConfig()) بین همه نمونه‌ها مشترک می‌شد؛
        # به‌جای آن None می‌گیریم و برای هر نمونه یک کانفیگ تازه می‌سازیم
        self.config = config or PipelineConfig()
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

        # str.translate بسیار سریع‌تر از چندین replace پشت‌سرهم است
        self.translation_table = str.maketrans(char_map)

    def _init_regex_patterns(self) -> None:
        """کامپایل الگوهای Regex برای پاک‌سازی، نرمال‌سازی و توکنایزیشن"""
        placeholders = "|".join(PLACEHOLDER_NAMES)

        # ۱. الگوهای مرحله پاک‌سازی
        # Entity استاندارد HTML با «;» (&amp; ، &#1234; ، &#x1F600;)
        self.entity_re = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
        # تگ HTML: با «<» شروع شود و بلافاصله حرف (یا / یا !) بیاید تا عبارتی مثل «۳ < ۵ > ۲» حذف نشود.
        # [^<>]* یعنی داخل تگ «<» دیگری نیست؛ پس «a<b ... <NUM>» به‌اشتباه یک تگ حساب نمی‌شود.
        # lookahead منفی باعث می‌شود تگ‌های جایگزین (<NUM> و ...) تگ HTML حساب نشوند؛
        # در غیر این صورت اجرای دوباره clean() آن‌ها را پاک می‌کرد و تست idempotency شکست می‌خورد.
        self.html_tag_re = re.compile(rf"<(?!(?:{placeholders})>)/?[A-Za-z!][^<>]*>")
        # حذف کاراکترهای نامرئی و کنترلی (با حفظ عامدانه نیم‌فاصله \u200c و پیونددهنده ایموجی \u200d)
        self.invisibles_re = re.compile(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200e\u200f\u202a-\u202e\u2060\u00ad\ufeff]"
        )
        # لینک: آخرین نویسه نباید علامت نگارشی باشد تا «.» یا «،» پایان جمله جزو لینک حذف نشود
        self.url_re = re.compile(
            r"(?:https?://|www\.)[^\s<>]*[^\s<>.,;:!?؟،؛)\]»\"']"
        )
        # ایمیل: دامنه به‌صورت (.بخش)+ تا نقطه پایان جمله جزو ایمیل نشود
        self.email_re = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
        # موبایل ایران: 09xxxxxxxxx ، +989xxxxxxxxx ، 00989xxxxxxxxx با ارقام انگلیسی/فارسی/عربی.
        # (?<![\w+]) به‌جای \b: چون \b قبل از «+» هرگز برقرار نیست (+ حرف کلمه نیست)
        self.phone_re = re.compile(
            rf"(?<![\w+])(?:\+98|0098|[0۰٠])[9۹٩][{DIGITS}]{{9}}(?!\w)"
        )
        # منشن: الگو «ایمیل | @نام» است تا در یک پیمایش، ایمیل‌ها (که @ دارند) از منشن جدا شوند.
        # (با lookbehind نتیجه به نویسه‌های قبل از @ وابسته می‌شد که نرمال‌سازی بعداً تغییرشان می‌دهد
        #  و idempotency می‌شکست؛ اینجا ایمیل ناشناس‌نشده، دست‌نخورده می‌ماند)
        self.mention_re = re.compile(rf"{self.email_re.pattern}|@\w+")
        self.hashtag_re = re.compile(r"#(\w+)")
        self.emoji_re = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            r"\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U0001F1E0-\U0001F1FF"
            r"\u2600-\u26FF\u2700-\u27BF\ufe0f\u200d]+"  # \ufe0f و \u200d: بازمانده‌های ایموجی ترکیبی
        )

        # ۲. الگوهای مرحله نرمال‌سازی
        # اعراب (فتحه، کسره، تنوین، تشدید) و خط کشیده/تطویل (\u0640)
        self.diacritics_re = re.compile(r"[\u0640\u064B-\u065F\u0670]")
        # عدد صحیح یا اعشاری (بعد از تبدیل ارقام به فارسی اجرا می‌شود)
        self.number_re = re.compile(r"\b[0-9۰-۹]+(?:\.[0-9۰-۹]+)?\b")
        # علائم نگارشی که در حالت topic حذف می‌شوند
        # نقطه فقط وقتی حذف می‌شود که «بین دو حرف» نباشد؛ تا ایمیل‌های ناشناس‌نشده
        # (anonymize_pii=False) مثل info@uni.ac.ir در حالت topic نشکنند.
        self.topic_punct_re = re.compile(r"[!؟؛،…«»\"'()\[\]]|(?<!\w)\.|\.(?!\w)")

        # ۳. الگوهای مرحله توکنایزیشن
        # الف) جمله‌بندی: کمترین متن ممکن تا یک نشانه پایان جمله (. ! ؟ …) که بعدش فاصله یا انتهای متن بیاید.
        # شرط «بعدش فاصله» یعنی «۳.۱۴» و «example.com» جمله را نمی‌شکنند.
        # گزینه $ هم برای آخرین جمله‌ای است که نشانه پایان ندارد.
        self.sentence_re = re.compile(
            r".+?(?:[.!?؟…]+[»)\]\"']*(?=\s|$)|$)", re.S
        )
        # ب) توکن‌سازی واژه (به ترتیب اولویت):
        #    ۱. تگ‌های جایگزین مثل <URL>
        #    ۲. عدد صحیح/اعشاری
        #    ۳. واژه: حروف و ارقام + نیم‌فاصله. عمداً بازه \u0600-\u06FF نیست چون «،» «؛» «؟» هم داخل آن است
        #       و به واژه می‌چسبید («خوبه؟» یک توکن می‌شد)
        #    ۴. هر نویسه دیگر (علائم نگارشی، ایموجی و ...) به‌صورت یک توکن مستقل
        self.word_tokenize_re = re.compile(
            rf"<(?:{placeholders})>|[0-9۰-۹]+(?:\.[0-9۰-۹]+)?|(?:[^\W_]|{ZWNJ})+|\S"
        )

    @staticmethod
    def _unescape_match(m: "re.Match[str]") -> str:
        """تبدیل یک Entity به نویسه واقعی (Entity ناشناخته همان‌طور می‌ماند)"""
        return html.unescape(m.group())

    # =========================================================================
    # مرحله ۱: پاک‌سازی متن (Text Cleaning)
    # =========================================================================
    def clean(self, text: str) -> str:
        """حذف نویزهای ساختاری، تگ‌های HTML، لینک‌ها و اطلاعات شخصی"""
        # تبدیل Entityهای HTML مانند &nbsp; به کاراکتر واقعی (قبل از حذف تگ‌ها تا &lt;b&gt; هم
        # مثل <b> رفتار شود). تا پایدار شدن تکرار می‌شود تا متن دوبار-کدشده (&amp;lt;) هم باز شود.
        # فقط Entityهای دارای «;» باز می‌شوند؛ html.unescape خودش «&amp» بدون ; را هم باز می‌کند و
        # بعد از نرمال‌سازی (مثلاً «&amp3» ← «&amp۳») دوباره Entity ساخته می‌شد.
        while (unescaped := self.entity_re.sub(self._unescape_match, text)) != text:
            text = unescaped
        # حذف تگ‌های HTML (تگ‌های جایگزین <URL>، <NUM> و ... دست‌نخورده می‌مانند)
        text = self.html_tag_re.sub(" ", text)
        # حذف کاراکترهای نامرئی مخرب
        text = self.invisibles_re.sub("", text)
        # حذف اعراب و کشیده «پیش از» الگوهای PII: وگرنه «۰۹۱۲۳۴۵۶۷۸۹ـ» (کشیده بعد از شماره)
        # مرز شماره را می‌شکست و فقط در اجرای دوم تشخیص داده می‌شد (نقض idempotency)
        text = self.diacritics_re.sub("", text)

        # جایگزینی لینک، ایمیل، شماره تماس و منشن (قبل از اعداد تا ساختارشان نشکند)
        # keep=True: به‌جای حذف، تگ جایگزین می‌گذاریم تا ساختار جمله حفظ شود
        keep = self.config.use_placeholders
        text = self.url_re.sub(" <URL> " if keep else " ", text)

        # اطلاعات شخصی فقط وقتی ناشناس می‌شوند که کاربر خواسته باشد
        if self.config.anonymize_pii:
            text = self.email_re.sub(" <EMAIL> " if keep else " ", text)
            text = self.phone_re.sub(" <PHONE> " if keep else " ", text)

        # منشن‌ها (@نام) را جایگزین کن؛ ایمیل‌هایی که ناشناس نشده‌اند (anonymize_pii=False) سالم می‌مانند
        user_repl = " <USER> " if keep else " "
        text = self.mention_re.sub(
            lambda m: user_repl if m.group().startswith("@") else m.group(), text
        )

        # حذف علامت # از هشتگ و تبدیل زیرخط (_) به فاصله
        text = self.hashtag_re.sub(lambda m: m.group(1).replace("_", " "), text)

        # در طبقه‌بندی موضوعی ایموجی‌ها حذف می‌شوند (در سایر حالت‌ها حامل احساس/معنا هستند)
        if self.config.task_mode == "topic":
            text = self.emoji_re.sub(" ", text)

        return text

    # =========================================================================
    # مرحله ۲: نرمال‌سازی متن (Text Normalization)
    # =========================================================================
    def normalize(self, text: str) -> str:
        """نرمال‌سازی چهارسطحی: نویسه، قالب، واژه و فاصله/نیم‌فاصله"""
        # در حالت topic علائم نگارشی هم حذف می‌شوند
        return self._normalize(text, strip_punct=self.config.task_mode == "topic")

    def _normalize(self, text: str, strip_punct: bool) -> str:
        """
        پیاده‌سازی اصلی نرمال‌سازی.
        strip_punct جدا شده تا process() بتواند جمله‌بندی را روی نسخه «با علائم» انجام دهد؛
        وگرنه در حالت topic هیچ نشانه پایان جمله‌ای نمی‌ماند و کل متن یک جمله می‌شد.
        """
        mode = self.config.task_mode

        # ۱. سطح نویسه: استاندارد NFKC یونیکد + نگاشت ک/ی و ارقام + حذف اعراب و کشیده
        # (توجه: NFKC نویسه «…» را به «...» تبدیل می‌کند؛ پایین‌تر دوباره «…» می‌شود)
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(self.translation_table)
        text = self.diacritics_re.sub("", text)

        # ۲. سطح قالب: جایگزینی اعداد با <NUM> (مگر در تسک NER که تاریخ و عدد مهم است)
        if mode != "ner":
            if self.config.use_placeholders:
                text = self.number_re.sub("<NUM>", text)
            elif mode == "topic":
                text = self.number_re.sub(" ", text)

        # ۳. سطح واژه: کاهش تکرار حروف
        # کاهش تکرار روی «همه نویسه‌ها» ممنوع است: «...» به «.» تبدیل می‌شد
        # (قاعده … هرگز اجرا نمی‌شد) و «۱۰۰۰» به «۱۰» خراب می‌شد؛ علائم نگارشی پایین‌تر جداگانه پردازش می‌شوند.
        # کاهش تکرار فقط برای «حروف» (نه ارقام و علائم): سلاااام -> سلام
        # در sentiment به دو تا کاهش می‌یابد تا تاکید حفظ شود: سلاااام -> سلاام
        keep_count = 2 if mode == "sentiment" else 1
        text = re.sub(rf"({LETTER})\1{{2,}}", r"\1" * keep_count, text)

        # حالت topic: علائم نگارشی «پیش از» اتصال نیم‌فاصله حذف می‌شوند.
        # اگر آخر کار حذف شوند، «سلام ... تر» پس از حذف «...» به «سلام تر» می‌رسد که قاعده پسوند
        # فقط در اجرای دوم آن را «سلام‌تر» می‌کند (نقض idempotency).
        if strip_punct:
            text = self.topic_punct_re.sub(" ", text)

        # ۴. سطح فاصله و نیم‌فاصله (ZWNJ)
        text = re.sub(rf"{ZWNJ}{{2,}}", ZWNJ, text)  # چند نیم‌فاصله -> یکی
        text = re.sub(rf"\s*{ZWNJ}\s*", ZWNJ, text)  # فاصله اطراف نیم‌فاصله حذف شود

        # اتصال پیشوندهای فعلی «می» و «نمی» با نیم‌فاصله (می رود -> می‌رود)
        # فقط وقتی کلمه بعدی با حرف فارسی شروع شود؛ پس «می .» یا «می <NUM>» نیم‌فاصله نمی‌گیرد
        text = re.sub(
            rf"\b(ن?می)\s+(?!که\b|را\b|هم\b|و\b)(?={FA_LETTER})", r"\1" + ZWNJ, text
        )

        # اتصال پسوندهای جمع، صفت برتر و ضمایر ملکی با نیم‌فاصله (کتاب ها -> کتاب‌ها)
        suffixes = r"(ها|های|هایی|تر|ترین|ام|ات|اش|مان|تان|شان)"
        text = re.sub(rf"(?<={FA_LETTER})\s+{suffixes}\b", ZWNJ + r"\1", text)

        # نیم‌فاصله فقط بین دو حرف معنا دارد؛ در ابتدا/انتهای واژه یا کنار علامت نگارشی حذف می‌شود
        text = re.sub(rf"(?<!{LETTER}){ZWNJ}|{ZWNJ}(?!{LETTER})", "", text)

        # تنظیم فاصله استاندارد اطراف علائم نگارشی
        text = re.sub(r"\s+([،؛؟!.…»)\]])", r"\1", text)  # فاصله قبل از علامت حذف شود
        # کاهش علائم تکراری؛ باید «بعد» از خط قبل باشد چون حذف فاصله، علائم را کنار هم می‌آورد
        # («؟ ؟» -> «؟؟» یا «... ...» -> «......»). اگر زودتر اجرا شود، اجرای دوم نتیجه را عوض می‌کرد.
        text = re.sub(r"\.{3,}", "…", text)  # ... -> …
        text = re.sub(r"…{2,}", "…", text)
        text = re.sub(r"([!؟؛،])\1+", r"\1", text)  # !!!! -> !
        text = re.sub(r"([«(\[])\s+", r"\1", text)  # فاصله بعد از پرانتز/گیومه باز حذف شود
        # بعد از «، ؛ ؟ !» فاصله بگذار (مگر قبل از علامت دیگر؛ وگرنه «؟!» به «؟ !» تبدیل می‌شد)
        text = re.sub(r"([،؛؟!])(?=[^\s»)\]،؛؟!.…])", r"\1 ", text)

        # فاصله‌های تکراری را یکی کن و دو سر متن را بتراش
        return re.sub(r"\s+", " ", text).strip()

    # =========================================================================
    # مرحله ۳: توکنایزیشن (Tokenization - سطح جمله و سطح واژه)
    # =========================================================================
    def tokenize_sentences(self, text: str) -> List[str]:
        """جمله‌بندی (Sentence Tokenization) با حفظ علائم پایانی و اعداد اعشاری"""
        # هر match یک جمله کامل (همراه علامت پایانی) است؛ بخش‌های خالی کنار گذاشته می‌شوند
        sentences = (m.group().strip() for m in self.sentence_re.finditer(text))
        return [s for s in sentences if s]

    def tokenize_words(self, text: str) -> List[str]:
        """
        توکن‌سازی واژگانی (Word Tokenization):
        - حفظ یکپارچگی کلمات مرکب دارای نیم‌فاصله (مانند «کتاب‌ها» یا «می‌روم»)
        - جداسازی علائم نگارشی چسبیده به کلمه (مانند «خوبه؟» -> ['خوبه', '؟'])
        - حفظ تگ‌های جایگزین (مانند '<URL>' و '<NUM>') به عنوان یک توکن واحد
        """
        return self.word_tokenize_re.findall(text)

    def _strip_punctuation(self, text: str) -> str:
        """حذف علائم نگارشی (ویژه حالت topic) برای توکن‌های هر جمله"""
        return re.sub(r"\s+", " ", self.topic_punct_re.sub(" ", text)).strip()

    # =========================================================================
    # اجرای کامل پایپ‌لاین و استخراج آمار
    # =========================================================================
    def process(self, raw_text: str) -> Dict[str, Any]:
        """اجرای ترتیبی پاک‌سازی -> نرمال‌سازی -> توکنایزیشن"""
        cleaned = self.clean(raw_text)
        normalized = self.normalize(cleaned)

        # جمله‌بندی روی نسخه‌ای که علائم نگارشی دارد انجام می‌شود (در topic، normalized علامت ندارد)
        with_punct = self._normalize(cleaned, strip_punct=False)
        sentences = self.tokenize_sentences(with_punct)

        # توکن‌های کل متن و توکن‌های هر جمله
        word_tokens = self.tokenize_words(normalized)
        topic = self.config.task_mode == "topic"
        tokens_per_sentence = [
            self.tokenize_words(self._strip_punctuation(s) if topic else s)
            for s in sentences
        ]

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
                # idempotency: اجرای دوباره کل پایپ‌لاین روی خروجی نباید چیزی را عوض کند
                "is_idempotent": (
                    normalized == self.normalize(self.clean(normalized))
                ),
            },
        }


def print_pipeline_output(result: Dict[str, Any], mode: str) -> None:
    """نمایش خروجی مرحله‌به‌مرحله در ترمینال برای ارائه کلاسی"""
    # کدهای رنگ ANSI برای رنگی کردن عنوان بخش‌ها در ترمینال
    BOLD, CYAN, GREEN, YELLOW, MAGENTA, RESET = (
        "\033[1m",
        "\033[96m",
        "\033[92m",
        "\033[93m",
        "\033[95m",
        "\033[0m",
    )

    # بخش ۱: ورودی بدون هیچ تغییری
    print(f"\n{BOLD}{CYAN}=== ۱. متن خام ورودی (Raw Text) ==={RESET}")
    print(result["raw_text"])

    # بخش ۲: خروجی نهایی دو مرحله اول
    print(
        f"\n{BOLD}{GREEN}=== ۲. متن پس از پاک‌سازی و نرمال‌سازی (Mode: {mode}) ==={RESET}"
    )
    print(result["normalized_text"])

    # بخش ۳: جمله‌ها
    print(
        f"\n{BOLD}{MAGENTA}=== ۳. توکنایزیشن در سطح جمله (Sentence Tokenization: {result['stats']['sentence_count']} جمله) ==={RESET}"
    )
    for i, sent in enumerate(result["sentences"], 1):
        print(f"  [جمله {i}] {sent}")

    # بخش ۴: واژه‌ها
    print(
        f"\n{BOLD}{YELLOW}=== ۴. توکنایزیشن در سطح واژه (Word Tokenization: {result['stats']['final_token_count']} توکن) ==={RESET}"
    )
    print(result["word_tokens"])

    # بخش ۵: آمار و تست کیفیت
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
        choices=list(VALID_MODES),
        default="general",
        help="حالت پیش‌پردازش بر اساس نوع مسئله (پیش‌فرض: general)",
    )
    parser.add_argument(
        "--text", type=str, help="متن دلخواه برای پردازش از طریق خط فرمان"
    )
    args = parser.parse_args()

    # ساخت کانفیگ و پایپ‌لاین بر اساس حالت انتخاب‌شده
    config = PipelineConfig(task_mode=args.mode)
    pipeline = PersianNLPPipeline(config)

    # متن نمونه شامل همه نویزهای رایج: HTML، ایموجی، تکرار حروف، اعراب، لینک، ایمیل، موبایل و ...
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
