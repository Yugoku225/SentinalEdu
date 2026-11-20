from sentineledu.ui.components import highlight


def test_highlight_marks_quotes_case_insensitively():
    out = highlight("Oxygen is released as a by-product.", ["oxygen is released"])
    assert out == "<mark>Oxygen is released</mark> as a by-product."


def test_highlight_escapes_student_text():
    out = highlight("x < y & <script>alert(1)</script>", ["x < y"])
    assert out.startswith("<mark>x &lt; y</mark> &amp; &lt;script&gt;") and "<script>" not in out


def test_highlight_merges_overlaps_tolerates_spacing_and_skips_unfound():
    text = "Plants make food\nfrom sunlight and water."
    out = highlight(text, ["make food from sunlight", "sunlight and water", "not in the answer", "ab"])
    assert out == "Plants <mark>make food<br>from sunlight and water</mark>."
