"""Tests for utils.text sanitization helpers."""

from graph_rag.utils.text import sanitize_text, is_corrupted, split_sentences, sanitize_answer


class TestSanitizeText:
    def test_empty(self):
        assert sanitize_text("") == ""
        assert sanitize_text(None) == ""

    def test_control_chars_removed(self):
        assert sanitize_text("hello\x00world this is fine") == "hello world this is fine"

    def test_zero_width_removed(self):
        assert sanitize_text("hel​lo world example text") == "hello world example text"

    def test_nbsp_normalized(self):
        assert sanitize_text("hello world and more words") == "hello world and more words"

    def test_whitespace_collapsed(self):
        assert sanitize_text("a  lot   of    space here") == "a lot of space here"

    def test_corrupted_returns_empty(self):
        assert sanitize_text("x" * 100) == ""

    def test_corruption_check_skippable(self):
        out = sanitize_text("x" * 100, check_corruption=False)
        assert out != ""


class TestIsCorrupted:
    def test_short_text_never_corrupted(self):
        assert not is_corrupted("hi")

    def test_normal_text(self):
        assert not is_corrupted("This is a perfectly normal sentence about dogs.")

    def test_excessive_repeats(self):
        assert is_corrupted("a" * 50)

    def test_unprintable_ratio(self):
        assert is_corrupted("\x01\x02\x03\x04\x05" * 10 + "hi")


class TestSplitSentences:
    def test_basic(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_empty(self):
        assert split_sentences("") == [""]


class TestSanitizeAnswer:
    def test_passthrough(self):
        assert sanitize_answer("A clean answer.") == "A clean answer."

    def test_zero_width(self):
        assert sanitize_answer("an​swer") == "answer"
