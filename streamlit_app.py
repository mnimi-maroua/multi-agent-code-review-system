"""
Streamlit demo for the multi-agent code review system.

Paste a diff, click Review, and watch Style, Logic, Test, and Critic
agents respond -- each one a real API call to the currently active
model config (set via AGENT_CONFIG in .env, B or C).

Run from the project root:
    streamlit run streamlit_app.py

Note: the active config is fixed for the lifetime of the Streamlit
process (Python only reads AGENT_CONFIG once, at import time). To switch
between Config B and C, change .env and restart Streamlit.
"""

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable when running `streamlit run streamlit_app.py` from
# the project root, without needing to install the project as a package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from agents.agents import (  # noqa: E402
    run_style_agent, run_logic_agent, run_test_agent, run_critic_agent,
    ACTIVE_CONFIG_NAME, MODEL_FOR_ROLE,
)

SAMPLE_DIFF = '''--- a/typer/_click/parser.py
+++ b/typer/_click/parser.py
@@ -184,7 +184,7 @@ class ArgumentParser:
     def process(self, value, state):
-        if self.nargs == -1 and self.obj.envvar is not None and value == ():
-            # Replace empty tuple with None so that a value from the
-            # environment may be tried.
+        if self.nargs == -1 and value == ():
+            # Replace empty tuple with None so regular default resolution
+            # (env var, default map, and parameter default) can be tried.
             value = None
--- a/tests/test_types.py
+++ b/tests/test_types.py
@@ -95,6 +95,40 @@ def test_enum_choice_missing_message() -> None:
     assert "morty" in result.output


+def hello_all_options(name: list[str] = typer.Option(["World"])) -> None:
+    for n in name:
+        print(f"Hello {n}!")
+
+
+def hello_all_args(names: list[str] = typer.Argument(["World"])) -> None:
+    for name in names:
+        print(f"Hello {name}!")
+
+
+def test_list() -> None:
+    result = runner.invoke(
+        app, ["hello-all-options", "--name", "Rick", "--name", "Morty"]
+    )
+    assert result.exit_code == 0
+    assert "Hello World!" not in result.output
+    assert "Hello Rick!" in result.output
+    assert "Hello Morty!" in result.output
+
+
+def test_list_empty() -> None:
+    result = runner.invoke(app, ["hello-all-args"])
+    assert result.exit_code == 0
+    assert "Hello World!" in result.output
'''


st.set_page_config(page_title="Multi-Agent Code Review Demo", layout="wide")

st.title("Multi-Agent Code Review System")
st.caption(
    f"Active config: **{ACTIVE_CONFIG_NAME}** &nbsp;|&nbsp; "
    + " · ".join(f"{role}: `{model}`" for role, model in MODEL_FOR_ROLE.items())
)

st.markdown(
    "Paste a diff below and click **Run Review** to see how Style, Logic, "
    "and Test Coverage agents analyze it independently, then how the "
    "Critic agent synthesizes their findings into one final review."
)

with st.expander("ℹ️ About this demo / limitations", expanded=False):
    st.markdown(
        "- This calls the Groq API live, using whichever model config is "
        "active (see caption above). Free-tier rate limits apply -- if a "
        "call fails, wait a bit and retry.\n"
        "- Diffs are truncated to 8000 characters to stay within a safe "
        "context window.\n"
        "- This demo runs agents sequentially (not in parallel), matching "
        "the production pipeline -- see the README's *Lessons learned* "
        "section for why."
    )

if "diff_text_input" not in st.session_state:
    st.session_state.diff_text_input = ""

col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("Load sample diff"):
        st.session_state.diff_text_input = SAMPLE_DIFF

diff_input = st.text_area(
    "Diff to review",
    height=300,
    placeholder="Paste a unified diff here...",
    key="diff_text_input",
)

MAX_DIFF_CHARS = 8000

if st.button("Run Review", type="primary", disabled=not diff_input.strip()):
    diff_text = diff_input[:MAX_DIFF_CHARS]

    style_col, logic_col, test_col = st.columns(3)

    with st.spinner("Running Style agent..."):
        try:
            style_review = run_style_agent(diff_text)
        except Exception as exc:  # noqa: BLE001
            style_review = f"[ERROR] {exc}"
    with style_col:
        st.subheader("🎨 Style")
        st.write(style_review)

    with st.spinner("Running Logic agent..."):
        try:
            logic_review = run_logic_agent(diff_text)
        except Exception as exc:  # noqa: BLE001
            logic_review = f"[ERROR] {exc}"
    with logic_col:
        st.subheader("🧠 Logic")
        st.write(logic_review)

    with st.spinner("Running Test Coverage agent..."):
        try:
            test_review = run_test_agent(diff_text)
        except Exception as exc:  # noqa: BLE001
            test_review = f"[ERROR] {exc}"
    with test_col:
        st.subheader("🧪 Test Coverage")
        st.write(test_review)

    st.divider()

    with st.spinner("Running Critic agent (synthesizing)..."):
        try:
            final_review = run_critic_agent(style_review, logic_review, test_review)
        except Exception as exc:  # noqa: BLE001
            final_review = f"[ERROR] {exc}"

    st.subheader("✅ Final Review (Critic)")
    st.markdown(final_review)