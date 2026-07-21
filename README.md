# Activation Steering for Caveman-style Prompts

The experiments in this repository investigate the question:

> When a model already has an instruction telling it to be concise, can **constant activation steering** push its code explanations even shorter, while keeping those explanations accurate and clear?

The setup follows the line of work on activation steering for instruction following
studied by Stolfo et al. [1], among others, applying it to a very practical usecase: the `caveman` skill for Claude. 

`caveman` is directly useful for developers who want their coding agent to drop the filler and answer concisely — its motto is *"why use many token when few token do trick"* — trimming roughly 65% of output tokens while keeping code and technical content intact. It shrinks what the agent *says*, not what it knows, entirely through prompting.

The model we test is `Qwen2.5-Coder-7B-Instruct`. The thing we are trying to compress is its natural-language explanations of code. Prompting alone can already shorten these explanations; the real question is whether steering adds anything *on top of* prompting, and whether it can do so without degrading the explanation itself.

To answer that, we compare four conditions for every test example.

<br>

## The Four Conditions

**1. Base**

A neutral prompt — no conciseness instruction, no steering. This is our reference point
for what the model does when left to its own devices.

**2. Prompt**

The caveman "full" mode instruction, reproduced word for word from
[caveman](https://github.com/juliusbrussee/caveman)'s `skills/caveman/SKILL.md`. We keep
the opening line, the Persistence section, the Rules, the no-self-reference paragraph, and
the Pattern line (assembled in `model_common.CAVEMAN_SUFFIX`).

We deliberately drop three parts of the original instruction — Auto-Clarity, Boundaries,
and the language-preservation paragraph — because the scenarios they govern (destructive
operations, writing commits or PRs, and non-English input) simply don't occur in this
task. Including them would add instruction text that never applies.

**3. Steer**

The diff-in-means steering vector, added at inference time, on top of the **Base** prompt.
There is no conciseness instruction in the prompt here — the only pressure toward brevity
comes from the steering vector itself. This isolates what steering does on its own.

**4. Prompt+Steer**

The same steering vector, added at inference time, but this time layered *on top of* the
**Prompt** condition. This is our primary comparison, and it targets the central question
directly:

> Does steering still add anything once the instruction is already present?

<br>

## The Task and Data

All experiments run on the **CodeXGLUE code-to-text** task, where the model is asked to
explain a function in natural language.

One important preprocessing step: before any code is used as a reference, we strip its
docstrings out via AST parsing (`src/data_prep.py`). This prevents answer leakage — without
it, the reference explanation could bleed into the prompt body and quietly inflate the
model's apparent accuracy. Stripping the docstrings keeps the evaluation honest.

## Main Result

We split the corpus into 180 train, 50 dev, and 180 held-out test examples, and report method comparisons on the test set. For correctness, we employ `gpt-4o-mini` to evaluate each output explanation, assigning a score of 0 (incorrect), 1 (partially correct), or 2 (completely correct).

| Condition | Avg tokens | Fully correct |
|---|---|---|
| Base | 150.0 (censored — see below) | 90-95% |
| Prompt | 56.8 | 90-95% |
| Steer alone | 149.9 | 90-95% |
| **Prompt+Steer** | **49.3** | 90-95% |

**Steering on top of the prompt gets a further ~13% token reduction beyond prompting alone.** Constant steering alone results in 149.9 tokens, which makes it indistinguishable from Base; the coefficient was calibrated specifically for the combined regime, not as a standalone replacement for the instruction. This matches a pattern we observed at higher coefficients during calibration: difference-of-means steering alone is substantially weaker than prompting.

The average token count reported for Base is lower than its true, uncensored value, since 95.6% of Base responses hit `MAX_NEW_TOKENS` without the model choosing to stop. The comparison we find most informative is Prompt vs. Prompt+Steer, since both are close to fully natural stopping.

![Test set: token count vs. correctness across the four conditions](results/summary_plot_test.png)

## Is caveman's style actually being followed?

The rule from caveman's `SKILL.md` is explicit and mechanically checkable: "Drop: articles (a/an/the)... Fragments OK." Instead of eyeballing compliance, we ran a regex search counting how often `a`, `an`, or `the` appears despite being told not to. Article usage varies sharply across the four conditions:

| Condition | Responses containing a/an/the | Articles per 100 words |
|---|---|---|
| Base | 100% | 12.68 |
| Steer alone | 100% | 12.47 |
| Prompt | 46.1% | 4.27 |
| **Prompt+Steer** | **31.1%** | **2.48** |

The prompt instruction (`model_common.CAVEMAN_SUFFIX`) changes model behavior substantially: articles drop roughly 66% relative to Base. Compliance with the specific "drop articles" rule remains only partial, however, and 46% of prompted responses still use one. Steer alone (calibrated for the combined regime, not as a standalone replacement) shows no effect on style either, consistent with its lack of effect on length. Steering on top of the prompt pushes compliance further than the text instruction manages alone. Overall, the model uses fewer articles at all, and at a lower rate when they do.

## Examples

In the cases below, both `Prompt` and `Prompt + Steer` examples are judged fully correct (`correct=2`). The `Prompt` response  is already article-free, while steering compresses further without dropping explanatory value:

<details open>
<summary><code>initial_step</code> (CORDIC algorithm) — 86 → 29 tokens</summary>

```python
def initial_step(self, phase, x, y):
    self.x[0] = x
    self.y[0] = y
    self.phase[0] = phase
    if self.MODE == CordicMode.ROTATION:
        if phase > 0.5:
            self.x[0] = -x
            self.phase[0] = phase - 1.0
        elif phase < -0.5:
            self.x[0] = -x
            self.phase[0] = phase + 1.0
    elif self.MODE == CordicMode.VECTORING:
        if x < 0.0 and y > 0.0:
            self.x[0] = -x
            self.y[0] = -y
            self.phase[0] = 1.0
        elif x < 0.0 and y < 0.0:
            self.x[0] = -x
            self.y[0] = -y
            self.phase[0] = -1.0
```

- **Prompt (86 tok):** <q>Function `initial_step` sets initial values for coordinates `x`, `y`, and phase. Depending on `MODE`, it adjusts these values based on specific conditions. If in rotation mode, it flips `x` and adjusts phase if outside [-0.5, 0.5]. In vectoring mode, it flips both `x` and `y` and sets phase to 1 or -1 based on quadrant.</q>
- **Prompt+Steer (29 tok):** <q>Set initial values for `x`, `y`, and `phase`. Adjust based on `MODE` and input values. Continue with next step.</q>
</details>

<details>
<summary><code>gen</code> (docs build) — 70 → 31 tokens</summary>

```python
def gen(skipdirhtml=False):
    docs_changelog = 'docs/changelog.rst'
    check_git_unchanged(docs_changelog)
    pandoc('--from=markdown', '--to=rst', '--output=' + docs_changelog, 'CHANGELOG.md')
    if not skipdirhtml:
        sphinx_build['-b', 'dirhtml', '-W', '-E', 'docs', 'docs/_build/dirhtml'] & FG
    sphinx_build['-b', 'html', '-W', '-E', 'docs', 'docs/_build/html'] & FG
```

- **Prompt (70 tok):** <q>Function `gen` generates documentation. It first checks if `docs/changelog.rst` is unchanged using `check_git_unchanged`. If unchanged, it converts `CHANGELOG.md` to `docs/changelog.rst` using Pandoc. Then, based on `skipdirhtml`, it builds HTML and optionally dirHTML documentation using Sphinx.</q>
- **Prompt+Steer (31 tok):** <q>Function `gen` generates documentation. Skips directory HTML build if specified. Converts Markdown changelog to RST. Builds Sphinx HTML and dirHTML outputs.</q>
</details>

<details>
<summary><code>Uninstall</code> (ADB package removal) — 58 → 26 tokens</summary>

```python
def Uninstall(self, package_name, keep_data=False, timeout_ms=None):
    cmd = ['pm uninstall']
    if keep_data:
        cmd.append('-k')
    cmd.append('"%s"' % package_name)
    return self.Shell(' '.join(cmd), timeout_ms=timeout_ms)
```

- **Prompt (58 tok):** <q>Uninstalls specified app. Removes it from device. Optionally keeps user data. Uses `pm uninstall` command. Passes package name. Adds `-k` flag if keeping data. Executes via `Shell` method. Waits up to `timeout_ms`. Returns result of uninstall operation.</q>
- **Prompt+Steer (26 tok):** <q>Uninstalls app. Removes it from device. Keeps data if specified. Runs command. Waits for completion. Returns result.</q>
</details>

<details>
<summary><code>_call</code> (HTTP wrapper with 401 handling) — 75 → 47 tokens</summary>

```python
def _call(self, path, method, body=None, headers=None):
    try:
        resp = self.http.do_call(path, method, body, headers)
    except http.HTTPError as err:
        if err.status == 401:
            raise PermissionError('Insufficient permissions to query ' + '%s with user %s :%s' % (path, self.user, err))
        raise
    return resp
```

- **Prompt (75 tok):** <q>Function `_call` makes HTTP request. Takes `path`, `method`, optional `body`, and `headers`. Calls `self.http.do_call` with these parameters. If HTTP error occurs, checks status code. If 401, raises `PermissionError` with message about insufficient permissions. Otherwise, re-raises original error. Returns response from HTTP call otherwise.</q>
- **Prompt+Steer (47 tok):** <q>Function `_call` sends HTTP request. Takes `path`, `method`, optional `body`, and `headers`. Tries sending request. If gets `401` error, raises `PermissionError`. Otherwise, returns response.</q>
</details>

<details>
<summary><code>callgraph</code> (static call-graph builder) — 65 → 38 tokens</summary>

```python
def callgraph(G, stmt_list):
    func_list = []
    for stmt in stmt_list:
        try:
            G.add_node(stmt.head.ident.name)
            func_list.append(stmt)
        except:
            pass
    for func in func_list:
        assert isinstance(func, node.function)
        func_name = func.head.ident.name
        for s in node.postorder(func):
            if s.__class__ is node.funcall and s.func_expr.__class__ is node.ident:
                G.add_edge(func_name, s.func_expr.name)
```

- **Prompt (65 tok):** <q>Function `callgraph` takes graph `G` and list of statements `stmt_list`. It adds nodes for functions found in `stmt_list` to graph `G`. Then it iterates over these functions, adding edges to graph `G` for each function call within them. This helps visualize function calls in code structure.</q>
- **Prompt+Steer (38 tok):** <q>Function `callgraph` takes graph `G` and list of statements `stmt_list`. It adds nodes for functions found in `stmt_list` and connects nodes with edges representing function calls.</q>
</details>

### Methodology

We chose the steering coefficient by sweeping (layer, coefficient) pairs on the
development set — 20 configurations across 50 dev examples — and discarding any that
produced degenerate, repetitive output. The full grid (`src/sweep_dev.py`,
`src/judge_sweep.py`, `src/analyze_sweep.py`) shows why simply picking the most
aggressive non-degenerate configuration is unsafe. Under constant steering, correctness
holds at 90–95%+ for coefficients 6–12 at layers 14 and 18, then drops sharply past a
coefficient of roughly 20, falling to 40–65%. We select layer 14, coefficient 6: the
point that gives up a little token reduction in exchange for correctness.

This tradeoff is reflected in the prior literature. Stolfo et al. [1] report the same direction on IFEval length instructions — larger steering weights `c` yield increasingly concise outputs (their Figure 5a, they have `c ∈ {0, 5, 10, 20, 40}`). They note that steering degraded generation quality in a few cases.

Heyman & Vandeputte [2] give a mechanistic explanation. A real prompt's influence varies sharply by token position, but a constant coefficient applies the same intervention to every position, whether or not that position needs it. 

Constant steering is therefore prone to oversteering. Once the coefficient exceeds what any single position calls for,
the target attribute keeps moving in the intended direction while coherence collapses. Their fix, Prompt Steering Replacement, learns a token-specific coefficient instead of a constant one.

Overall, conciseness and correctness seem to trade off against one other: pushing the
coefficient for shorter responses usually costs correctness or intelligibility. But these
experiments show that activation steering for instruction following *can* keep responses
short without sacrificing their explanatory value.

![Dev sweep: token count vs. correctness across all 20 (layer, coefficient) configs](results/summary_sweep_plot_dev.png)

## Further work

- Prompt Steering Replacement
- The task is code **explanation** only, mirroring the use case within Caveman and popular for code assistant usage as a whole.

## Running it

Everything in `src/data_prep.py`, `src/judge*.py`, and `src/analyze*.py` can run locally — no GPU
needed. The GPU-bound steps (`steering_const.py`, `sweep_dev.py`, `generate.py`) run on a rented
GPU pod via `infra/run_gpu_pipeline.sh` (or invoked directly, as when re-running just one step).

```
python3 src/data_prep.py                        # local — data/{train,dev,test}.jsonl

# on the GPU pod:
bash infra/setup_runpod.sh
python3 src/steering_const.py                   # -> results/const_steer_{config.json,directions.pt}
python3 src/sweep_dev.py                        # -> results/sweep_dev.jsonl (all grid configs, dev)

# back locally: judge the sweep, inspect the frontier, edit const_steer_config.json's
# layer/coeff to the chosen operating point (see "Methodology" above)
python3 src/judge_sweep.py                      # -> results/judged_sweep_dev.jsonl (needs openai.key)
python3 src/analyze_sweep.py                    # -> results/summary_sweep_dev.json, plot

# back on the GPU pod, with the final chosen config:
python3 src/generate.py --split test            # -> results/generations_test.jsonl

# back locally:
python3 src/judge.py --split test               # -> results/judged_test.jsonl (needs openai.key)
python3 src/analyze.py --split test             # -> results/summary_test.json, summary_plot_test.png
```

## References

[1] Alessandro Stolfo, Vidhisha Balachandran, Safoora Yousefi, Eric Horvitz, Besmira Nushi.
"Improving Instruction-Following in Language Models through Activation Steering." arXiv:2410.12877.
https://arxiv.org/abs/2410.12877

[2] Geert Heyman, Frederik Vandeputte. "Steer Like the LLM: Activation Steering that Mimics
Prompting." arXiv:2605.03907. https://arxiv.org/abs/2605.03907
