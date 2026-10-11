"""Offline tests: no network, no API keys."""
import json
import os
import types

import pytest

from gaia_agent import agent as agent_mod
from gaia_agent import config, llm, runner, store, submit as submit_mod
from gaia_agent.tools import files as files_tools
from gaia_agent.tools import web
from gaia_agent.tools.base import make_tool


# ---------- answer parsing / agent helpers ----------
def test_extract_final_answer():
    f = agent_mod.extract_final_answer
    assert f("blah\nFINAL ANSWER: 42.") == "42"
    assert f("**FINAL ANSWER: Paris**") == "Paris"
    assert f("The answer is a long sentence with many many words in it here") == "UNKNOWN"
    assert f("reasoning...\n3") == "3"


def test_trim_and_filter():
    msgs = [{"role": "tool", "content": "x" * 5000} for _ in range(4)]
    agent_mod._trim_history(msgs)
    assert len(msgs[0]["content"]) < 1400
    assert len(msgs[-1]["content"]) == 5000 and len(msgs[-2]["content"]) == 5000

    def tool(a, b=1):
        return a

    assert agent_mod._filter_args(tool, {"a": 1, "bogus": 2}) == {"a": 1}


# ---------- web helpers ----------
def test_safe_url():
    assert not web._safe_url("http://169.254.169.254/computeMetadata/v1/")
    assert not web._safe_url("http://localhost:8080")
    assert not web._safe_url("file:///etc/passwd")


def test_extract_relevant_and_markdown():
    md = "\n".join(["# Intro", "x " * 40] + [f"## Section {i}\n" + ("filler text " * 60) for i in range(8)]
                   + ["## Albums\n" + "Studio albums released 1965 1970 table " * 30])
    out = web._extract_relevant(md, "studio albums with release years")
    assert "Albums" in out and "Studio albums" in out

    html = "<html><body><nav>menu</nav><h1>T</h1><table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table></body></html>"
    m = web._html_to_markdown(html)
    assert "menu" not in m and "| a | b |" in m.replace("  ", " ")


# ---------- file tools ----------
def test_run_python_hides_secrets(monkeypatch):
    monkeypatch.setenv("MY_SECRET_KEY", "hunter2")
    out = files_tools.run_python(code="import os; print([k for k in os.environ if 'SECRET' in k]); print(2+2)")
    assert "4" in out and "MY_SECRET_KEY" not in out


def test_read_file_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    assert "2 rows x 2 columns" in files_tools.read_file(str(p))


# ---------- config / store ----------
def test_secret_aliases(monkeypatch):
    monkeypatch.setitem(config.SECRET_ALIASES, "GROQ_API_KEY_1", "groq1")
    monkeypatch.setenv("GROQ_API_KEY_1", "")   # registered for cleanup, empty so the alias is used
    monkeypatch.setenv("groq1", "abc")
    assert "GROQ_API_KEY_1" in config.load_secrets(force=True)
    assert os.environ["GROQ_API_KEY_1"] == "abc"


def test_no_aliases_by_default():
    assert config.SECRET_ALIASES == {}


def test_dotenv_is_loaded_from_cwd(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY_2=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "x")   # registers cleanup ...
    monkeypatch.delenv("OPENROUTER_API_KEY_2")        # ... and makes the variable absent for this test
    assert "OPENROUTER_API_KEY_2" in config.load_secrets(force=True)
    assert os.environ["OPENROUTER_API_KEY_2"] == "from-dotenv"


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY_3=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY_3", "from-env")
    config.load_secrets(force=True)
    assert os.environ["OPENROUTER_API_KEY_3"] == "from-env"


# ---------- CLI ----------
def test_cli_ask(monkeypatch, capsys):
    import gaia_agent.agent as agent_pkg
    import gaia_agent.tools as tools_pkg
    from gaia_agent import cli

    class FakeAgent:
        def __init__(self, tools=None): pass
        def run(self, prompt, max_steps=12, verbose=True, preview=200):
            return "Paris" if "capital" in prompt else "?"

    monkeypatch.setattr(agent_pkg, "Agent", FakeAgent)
    monkeypatch.setattr(tools_pkg, "build_tools", lambda: [])
    monkeypatch.setattr(cli, "load_secrets", lambda: [])
    cli.main(["ask", "What is the capital of France?", "--quiet"])
    assert "FINAL ANSWER: Paris" in capsys.readouterr().out


def test_cli_ask_attached_file_path_is_in_prompt(tmp_path, monkeypatch):
    import gaia_agent.agent as agent_pkg
    import gaia_agent.tools as tools_pkg
    from gaia_agent import cli

    seen = {}

    class FakeAgent:
        def __init__(self, tools=None): pass
        def run(self, prompt, max_steps=12, verbose=True, preview=200):
            seen["prompt"] = prompt
            return "ok"

    f = tmp_path / "data.csv"
    f.write_text("a\n1\n")
    monkeypatch.setattr(agent_pkg, "Agent", FakeAgent)
    monkeypatch.setattr(tools_pkg, "build_tools", lambda: [])
    monkeypatch.setattr(cli, "load_secrets", lambda: [])
    cli.main(["ask", "Sum of a?", "--file", str(f), "--quiet"])
    assert str(f.resolve()) in seen["prompt"]


def test_cli_ask_missing_file_and_no_keys(monkeypatch):
    from gaia_agent import cli

    monkeypatch.setattr(cli, "load_secrets", lambda: [])
    with pytest.raises(SystemExit, match="File not found"):
        cli.main(["ask", "q", "--file", "/definitely/not/here.csv"])

    import gaia_agent.agent as agent_pkg
    import gaia_agent.tools as tools_pkg

    class BrokenAgent:
        def __init__(self, tools=None): pass
        def run(self, *a, **k):
            raise llm.QuotaExhausted("daily quota")

    monkeypatch.setattr(agent_pkg, "Agent", BrokenAgent)
    monkeypatch.setattr(tools_pkg, "build_tools", lambda: [])
    with pytest.raises(SystemExit, match="Stopped: daily quota"):
        cli.main(["ask", "q"])


def test_store(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIA_DATA_DIR", str(tmp_path))
    store.save_answer("t1", "5", "ok")
    store.save_answer("t1", "UNKNOWN", "ok")
    store.save_answer("t2", "ERR", "error")
    assert store.load_answers() == {"t1": "5"}


# ---------- LLM layer ----------
def test_retry_after():
    assert abs(llm._retry_after(Exception("Please try again in 7m12.5s. Need more")) - 432.5) < 1e-6
    assert abs(llm._retry_after(Exception("try again in 850ms")) - 0.85) < 1e-6
    assert llm._retry_after(Exception("Try again in 60 seconds")) is None


class FakeRouter:
    def __init__(self, errors):
        self.errors, self.calls = errors, 0

    def completion(self, **kw):
        self.calls += 1
        if self.errors:
            raise Exception(self.errors.pop(0))
        return "ok"


def test_complete_waits_then_succeeds(monkeypatch):
    router = FakeRouter(["RateLimitError 429 Please try again in 3.2s"])
    monkeypatch.setattr(llm, "get_router", lambda: (router, "primary"))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    assert llm.complete(messages=[]) == "ok" and router.calls == 2


def test_complete_stops_on_daily_quota(monkeypatch):
    router = FakeRouter(["RateLimitError 429 Please try again in 41m10.2s"])
    monkeypatch.setattr(llm, "get_router", lambda: (router, "primary"))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(llm.QuotaExhausted):
        llm.complete(messages=[])
    assert router.calls == 1


def test_complete_propagates_other_errors(monkeypatch):
    router = FakeRouter(["400 bad request"])
    monkeypatch.setattr(llm, "get_router", lambda: (router, "primary"))
    with pytest.raises(Exception, match="bad request"):
        llm.complete(messages=[])
    assert router.calls == 1


def test_router_entry_group(monkeypatch):
    pytest.importorskip("litellm")
    for i in range(1, 6):
        monkeypatch.delenv(f"GROQ_API_KEY_{i}", raising=False)
        monkeypatch.delenv(f"OPENROUTER_API_KEY_{i}", raising=False)
    monkeypatch.setattr(llm, "load_secrets", lambda *a, **k: [])
    monkeypatch.setenv("OPENROUTER_API_KEY_1", "dummy")
    assert llm.build_router()[1] == "fallback"
    monkeypatch.setenv("GROQ_API_KEY_1", "dummy")
    assert llm.build_router()[1] == "primary"
    monkeypatch.delenv("GROQ_API_KEY_1")
    monkeypatch.delenv("OPENROUTER_API_KEY_1")
    with pytest.raises(RuntimeError):
        llm.build_router()


# ---------- agent loop with a fake LLM ----------
def _call(id_, name, args):
    return types.SimpleNamespace(id=id_, function=types.SimpleNamespace(name=name, arguments=args))


def _resp(content=None, calls=None):
    msg = types.SimpleNamespace(content=content, tool_calls=calls)
    return types.SimpleNamespace(model="fake", choices=[types.SimpleNamespace(message=msg)])


def test_agent_loop(monkeypatch):
    script = [
        _resp(calls=[_call("1", "echo", json.dumps({"text": "hi", "bogus": 1}))]),  # invented parameter
        _resp(calls=[_call("2", "echo", "{not json")]),                              # invalid JSON
        _resp(content="I think it is 42"),                                          # no marker -> nudge
        _resp(content="FINAL ANSWER: 42."),
    ]
    monkeypatch.setattr(agent_mod, "complete", lambda **kw: script.pop(0))
    tool = make_tool("echo", "d", {"text": {"type": "string"}}, ["text"], lambda text: text)
    assert agent_mod.Agent(tools=[tool]).run("q", verbose=False) == "42"
    assert not script


def test_agent_out_of_steps(monkeypatch):
    script = [_resp(calls=[_call("1", "echo", json.dumps({"text": "x"}))]), _resp(content="FINAL ANSWER: 7")]
    monkeypatch.setattr(agent_mod, "complete", lambda **kw: script.pop(0))
    tool = make_tool("echo", "d", {"text": {"type": "string"}}, ["text"], lambda text: text)
    assert agent_mod.Agent(tools=[tool]).run("q", max_steps=1, verbose=False) == "7"


# ---------- runner / submit ----------
QUESTIONS = [
    {"task_id": "aaaaaaaa-1", "question": "Q1?", "file_name": ""},
    {"task_id": "bbbbbbbb-2", "question": "Q2?", "file_name": ""},
    {"task_id": "cccccccc-3", "question": "Q3?", "file_name": ""},
]


def test_iter_questions_and_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(runner, "get_questions", lambda refresh=False: QUESTIONS)
    monkeypatch.setattr(runner, "build_tools", lambda: [])

    class FakeAgent:
        def __init__(self, tools=None): pass
        def run(self, prompt, verbose=True, preview=200):
            return "UNKNOWN" if prompt.startswith("Q2") else "ans-" + prompt[:2]

    monkeypatch.setattr(runner, "Agent", FakeAgent)
    results = runner.run_questions(pause=0)
    assert [r["answer"] for r in results] == ["ans-Q1", "UNKNOWN", "ans-Q3"]
    again = runner.run_questions(missing=True, pause=0)   # only Q2 is still missing
    assert [r["index"] for r in again] == [2]
    assert store.load_answers()["aaaaaaaa-1"] == "ans-Q1"


def test_iter_questions_stops_on_quota(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(runner, "get_questions", lambda refresh=False: QUESTIONS)
    monkeypatch.setattr(runner, "build_tools", lambda: [])

    class FakeAgent:
        def __init__(self, tools=None): self.n = 0
        def run(self, prompt, verbose=True, preview=200):
            self.n += 1
            if self.n == 2:
                raise llm.QuotaExhausted("daily quota")
            return "ok"

    monkeypatch.setattr(runner, "Agent", FakeAgent)
    assert len(runner.run_questions(pause=0)) == 1


def test_submit_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(submit_mod, "get_questions", lambda: QUESTIONS)
    store.save_answer("aaaaaaaa-1", "5", "ok")
    payload, missing = submit_mod.build_payload("me", "https://huggingface.co/spaces/me/x/tree/main")
    assert missing == [2, 3]
    assert payload["answers"][0] == {"task_id": "aaaaaaaa-1", "submitted_answer": "5"}
    assert payload["answers"][1]["submitted_answer"] == "UNKNOWN"
    assert payload["username"] == "me"
    monkeypatch.delenv("HF_USERNAME", raising=False)
    with pytest.raises(ValueError):
        submit_mod.submit("https://huggingface.co/spaces/me/x/tree/main", dry_run=True)
