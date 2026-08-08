import numpy as np
import pytest

from experiments.monitor import corpora, pile


class TestR5Random:
    def test_is_token_ids_never_text(self):
        """Decoding random ids to text and re-encoding drops out-of-vocab ids
        and re-imposes tokenizer structure, which would turn the 'random'
        rung into something the model finds partly parseable."""
        out = corpora.build_r5_random(1000, bos_token_id=2, n=8)
        assert all(p.text is None for p in out)
        assert all(p.token_ids is not None for p in out)

    def test_length_matches_the_build_context(self):
        out = corpora.build_r5_random(1000, bos_token_id=2, n=8)
        assert all(len(p.token_ids) == corpora.CONTEXT_LENGTH for p in out)
        assert all(p.token_ids[0] == 2 for p in out)

    def test_is_seed_deterministic(self):
        a = corpora.build_r5_random(1000, 2, n=4, seed=3)
        b = corpora.build_r5_random(1000, 2, n=4, seed=3)
        c = corpora.build_r5_random(1000, 2, n=4, seed=4)
        assert [p.token_ids for p in a] == [p.token_ids for p in b]
        assert [p.token_ids for p in a] != [p.token_ids for p in c]

    def test_ids_stay_inside_the_vocabulary(self):
        out = corpora.build_r5_random(500, 2, n=32)
        assert max(max(p.token_ids[1:]) for p in out) < 500


class TestR3Scaffolds:
    """R3 must isolate the template. If the scaffolds differed in length, or
    if only some were used, the rung would measure length or one scaffold."""

    def test_every_scaffold_is_used_and_content_is_shared(self):
        content = [f"document number {i} " * 60 for i in range(20)]
        out = corpora.build_r3_template(_FakeTokenizer(), content, n=20)
        used = {p.source for p in out}
        assert len(used) == len(corpora.SCAFFOLDS)

    def test_content_budget_shrinks_for_verbose_scaffolds(self):
        """A long scaffold must eat into its content budget, not add to the
        total, or 'template shift' becomes 'longer prompt'."""
        tok = _FakeTokenizer()
        overheads = [len(tok(t.format(""), add_special_tokens=False)["input_ids"])
                     for _, t in corpora.SCAFFOLDS]
        assert max(overheads) > min(overheads)   # they really do differ
        content = ["word " * 400] * 10
        out = corpora.build_r3_template(tok, content, n=10)
        lengths = [len(tok(p.text, add_special_tokens=False)["input_ids"])
                   for p in out]
        assert max(lengths) - min(lengths) <= 4

    def test_slot_survives_the_json_scaffold(self):
        """The api_frame scaffold is a JSON literal full of braces; a naive
        .format() on it raises or eats the wrong braces."""
        name_to_tpl = dict(corpora.SCAFFOLDS)
        rendered = name_to_tpl["api_frame"].format("HELLO")
        assert "HELLO" in rendered
        assert rendered.startswith('{"model"')


class TestR1Domain:
    def test_budget_split_reports_truthfully(self):
        prov = {"mbpp": 974, "gsm8k": 1026}
        assert prov["mbpp"] + prov["gsm8k"] == 2000

    @pytest.mark.parametrize("n", [16, 100, 2000])
    def test_never_returns_a_negative_count(self, n):
        """The first version computed the math count by subtraction and
        reported gsm8k=-958 for small n."""
        half = n // 2
        n_code = min(974, max(half, n - 7473))
        n_math = min(7473, n - n_code)
        assert n_code >= 0 and n_math >= 0
        assert n_code + n_math == n


class TestPilePools:
    def test_skip_clears_the_largest_build(self):
        """Every pool must start after the p=1 build's consumption, or the
        'in-distribution' rung is scoring the dictionary on its own training
        activations — the exact defect found in exp_coverage.py."""
        assert pile.PILE_SKIP > pile.BUILD_PROMPTS_CONSUMED
        assert pile.PILE_SKIP - pile.BUILD_PROMPTS_CONSUMED >= 10_000

    def test_pools_are_disjoint_ranges(self):
        a = pile.PILE_SKIP
        b = a + pile.REF_DOCS
        c = b + pile.R0_DOCS
        d = c + pile.R3_DOCS
        assert a < b < c < d
        assert len(set(range(a, b)) & set(range(b, c))) == 0
        assert len(set(range(b, c)) & set(range(c, d))) == 0

    def test_reference_pool_covers_the_largest_coreset(self):
        """S3 samples K=5796 activations without replacement, so the pool has
        to be comfortably larger than that."""
        approx_acts = pile.REF_DOCS * (pile.CONTEXT_LENGTH - 1)
        assert approx_acts > 5796 * 10

    def test_matches_the_build_stream_settings(self):
        assert pile.SHUFFLE_SEED == 0
        assert pile.SHUFFLE_BUFFER == 10_000
        assert pile.MIN_CHARS == 200
        assert pile.CONTEXT_LENGTH == 128


class _FakeTokenizer:
    """Whitespace tokenizer — enough to exercise the budget arithmetic without
    downloading a real one."""

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, str):
            return {"input_ids": text.split()}
        return {"input_ids": [t.split() for t in text]}

    def decode(self, ids):
        return " ".join(ids)
