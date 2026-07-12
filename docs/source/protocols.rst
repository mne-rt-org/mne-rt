.. _protocols:

NF Protocols
============

A **neurofeedback protocol** decides *when* and *how much* to reward.
It sits between the raw NF feature value (e.g. alpha power) and the
feedback signal delivered to the participant.  MNE-RT ships ten protocols,
covering the full spectrum from simple fixed-threshold designs to
adaptive psychophysics staircases, reinforcement-learning thresholds,
operant conditioning schedules, cross-session transfer, and double-blind
sham control.

All protocols share the same two-value contract:

.. contents::
   :local:
   :depth: 1

----

Live Threshold Display
-----------------------

Every protocol also exposes a read-only ``current_threshold`` property, in
the same raw units as the NF feature it evaluates.  For protocols with a
literal fixed or adaptive level (:class:`~mne_rt.protocols.ThresholdProtocol`,
:class:`~mne_rt.protocols.UpDownStaircaseProtocol`,
:class:`~mne_rt.protocols.RLProtocol`, :class:`~mne_rt.protocols.PercentileProtocol`)
this is just that level.  For relative-criterion protocols
(:class:`~mne_rt.protocols.ZScoreProtocol`, :class:`~mne_rt.protocols.TransferProtocol`)
it is the z-score boundary converted back to raw units
(``mean ± zscore_threshold × std``).  Wrapper protocols
(:class:`~mne_rt.protocols.ShamProtocol`, :class:`~mne_rt.protocols.OperantProtocol`)
pass the value through from whatever they wrap.
:class:`~mne_rt.protocols.LinearTrendProtocol` always returns ``None`` since
it rewards a *trend*, not a level.

When a protocol is passed to :meth:`~mne_rt.RTStream.record_main` (or
:meth:`~mne_rt.RTStream.replay`) with ``show_nf_signal=True``,
``current_threshold`` is read on every analysis window and drawn as a
dashed horizontal line on the corresponding :class:`~mne_rt.viz.NFPlot`
subplot — updating live for adaptive protocols.  The line can be toggled
from the *Display* panel of the NFPlot window.  See :doc:`visualization`.

----

Choosing a Protocol
-------------------

.. raw:: html

   <div style="overflow-x:auto; margin:16px 0;">
   <table style="border-collapse:separate; border-spacing:0; width:100%; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <thead>
     <tr>
       <th style="background:#1e40af;color:white;padding:8px 14px;border-radius:8px 0 0 0;">Protocol</th>
       <th style="background:#1e40af;color:white;padding:8px 14px;">Best for</th>
       <th style="background:#1e40af;color:white;padding:8px 14px;">Key property</th>
       <th style="background:#1e40af;color:white;padding:8px 14px;border-radius:0 8px 0 0;">Requires baseline?</th>
     </tr>
   </thead>
   <tbody>
     <tr style="background:#eff6ff;">
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;font-weight:600;">ThresholdProtocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Clinical NF, demos, debugging</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Fixed or slowly-adapting threshold; most interpretable</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">No</td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;font-weight:600;">ZScoreProtocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">Within-session adaptation, variable signals</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">Rewards deviation from running mean — self-calibrating</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">No (warmup windows)</td>
     </tr>
     <tr style="background:#eff6ff;">
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;font-weight:600;">PercentileProtocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Gamification, progressive training</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Threshold tracks own rolling distribution — difficulty auto-scales</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">No</td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;font-weight:600;">LinearTrendProtocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">Slow-learning paradigms, gradual skill acquisition</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">Rewards slope, not level — sustained change wins</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">No</td>
     </tr>
     <tr style="background:#eff6ff;">
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;font-weight:600;">ShamProtocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">RCTs, sham control conditions</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Wraps any protocol; shuffles feedback on fraction of windows</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Depends on inner</td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;font-weight:600;">UpDownStaircaseProtocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">Threshold estimation, psychophysics</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">Converges threshold to target success rate via n-up/n-down rule</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">No</td>
     </tr>
     <tr style="background:#eff6ff;">
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;font-weight:600;">MultiBandProtocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Dual-band NF (alpha↑ + theta↓, SMR↑ + theta↓)</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">AND/OR combination of two independent inner protocols</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Depends on inner protocols</td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;font-weight:600;">RLProtocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">Automated threshold search, fully adaptive training</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">ε-greedy exploration + hit-rate-driven threshold update</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dcfce7;">No (warmup windows)</td>
     </tr>
     <tr style="background:#eff6ff;">
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;font-weight:600;">OperantProtocol</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Partial reinforcement, ratio/interval schedules</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Wraps any protocol; gates rewards by FR/VR/FI/VI schedule</td>
       <td style="padding:7px 14px;border-bottom:1px solid #dbeafe;">Depends on inner</td>
     </tr>
     <tr style="background:#f0fdf4;">
       <td style="padding:7px 14px;font-weight:600;">TransferProtocol</td>
       <td style="padding:7px 14px;">Cross-session transfer, prior-seeded z-score</td>
       <td style="padding:7px 14px;">Seeds running statistics from a prior session file — zero warmup</td>
       <td style="padding:7px 14px;">No (prior file)</td>
     </tr>
   </tbody>
   </table>
   </div>

----

Threshold Protocol
------------------

.. raw:: html

   <div style="background:#eff6ff; border-left:4px solid #3b82f6; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Reward rule:</strong> &nbsp;
   crossed &nbsp;=&nbsp; <em>x</em> &gt; <em>θ</em> &nbsp;&nbsp;(direction = "up")
   &nbsp;|&nbsp; <em>x</em> &lt; <em>θ</em> &nbsp;&nbsp;(direction = "down")
   </div>

The simplest and most widely used protocol.  A fixed threshold
:math:`\theta` is placed on the NF feature scale and the participant
receives feedback whenever the signal crosses it.

.. math::

   \text{crossed} = \begin{cases}
     x_t > \theta & \text{if direction = "up"} \\
     x_t < \theta & \text{if direction = "down"}
   \end{cases}
   \qquad
   \text{magnitude} = \begin{cases}
     |x_t - \theta| & \text{if crossed} \\
     0 & \text{otherwise}
   \end{cases}

The ``adapt_rate`` parameter adds slow threshold drift to maintain a
target success rate over time — increase :math:`\theta` after rewarded
windows, decrease after missed windows.

**When to use:**  Quick prototyping, clinical applications where the
threshold is set by an expert, or as the inner protocol for
:class:`~mne_rt.protocols.ShamProtocol` or
:class:`~mne_rt.protocols.MultiBandProtocol`.

.. code-block:: python

    from mne_rt.protocols import ThresholdProtocol

    proto = ThresholdProtocol(threshold=1.2e-10, direction="up")
    crossed, mag = proto.evaluate(alpha_power)

----

Z-Score Protocol
----------------

.. raw:: html

   <div style="background:#f0fdf4; border-left:4px solid #22c55e; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Reward rule:</strong> &nbsp;
   z &nbsp;=&nbsp; (x &minus; μ) / σ &nbsp;&nbsp;→&nbsp;&nbsp; reward if z &gt; z<sub>thr</sub>
   </div>

During a short warmup phase (typically 20 windows) the protocol collects
enough values to seed its running mean :math:`\hat\mu` and standard
deviation :math:`\hat\sigma` using Welford's online algorithm.  From then
on, each new value is z-scored against the participant's own distribution:

.. math::

   z_t = \frac{x_t - \hat\mu_t}{\hat\sigma_t}, \qquad
   \text{crossed} = z_t > z_\text{thr}

Because the statistics adapt to each individual, the same threshold
:math:`z_\text{thr}` (e.g. 0.5) produces approximately the same success
rate regardless of signal amplitude differences between participants or
sessions.

**Typical trajectory**

.. raw:: html

   <pre style="background:#1e293b; color:#e2e8f0; padding:12px 16px; border-radius:8px; font-size:12px; line-height:1.6; overflow-x:auto;">
   window:  1  2  3  4  5 … 20 | 21  22  23  24  25
   value:   1  2  1  3  2 … 2  |  3   1   5   2   4
   z-score: —  —  —  —  — … —  |0.2  -0.8 2.1 0.0 1.3
   reward:  ·  ·  ·  ·  · … ·  |  ·   ·   ✓   ·   ✓     (z_thr = 0.5)
   warmup phase ───────────────┘ live phase ──────────
   </pre>

**When to use:**  When signal amplitude varies across sessions or
participants and you want consistent difficulty without manually
re-setting thresholds.

.. code-block:: python

    from mne_rt.protocols import ZScoreProtocol

    proto = ZScoreProtocol(direction="up", zscore_threshold=0.5, warmup_windows=20)
    for value in nf_stream:
        crossed, mag = proto.evaluate(value)

----

Percentile Protocol
-------------------

.. raw:: html

   <div style="background:#fefce8; border-left:4px solid #eab308; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Reward rule:</strong> &nbsp;
   reward if x &gt; P<sub>n</sub>(history) &nbsp;&nbsp;|&nbsp;&nbsp; typically n = 75
   </div>

A rolling buffer of the last ``history_len`` values is maintained.  The
N-th percentile of this buffer acts as the current threshold:

.. math::

   \theta_t = \operatorname{Percentile}_n\!\bigl(\{x_{t-W}, \ldots, x_{t-1}\}\bigr)

Because the threshold tracks the participant's own recent distribution,
difficulty automatically scales up during peak performance and relaxes
during fatigue — producing a self-calibrating reward rate of
approximately :math:`(100 - n)\%`.

**When to use:**  Training paradigms where the goal is *relative
improvement* over recent performance rather than crossing an absolute
target.  The 75th-percentile default rewards the best ~25 % of windows.

.. code-block:: python

    from mne_rt.protocols import PercentileProtocol

    proto = PercentileProtocol(percentile=75.0, direction="up", history_len=100)
    crossed, mag = proto.evaluate(alpha_power)

----

Linear-Trend Protocol
---------------------

.. raw:: html

   <div style="background:#fdf4ff; border-left:4px solid #a855f7; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Reward rule:</strong> &nbsp;
   OLS slope &gt; slope<sub>thr</sub> &nbsp;<em>and</em>&nbsp; R² &ge; min_r2
   </div>

Rather than comparing the *current value* to a threshold, this protocol
fits an ordinary-least-squares regression over the last ``window`` values
and rewards when the slope is in the target direction with sufficient
goodness-of-fit:

.. math::

   [a,\, b] &= \operatorname{OLS} \!\bigl(\{1, \ldots, W\},\, \{x_{t-W+1}, \ldots, x_t\}\bigr) \\
   \text{crossed} &= a > a_\text{thr} \;\land\; R^2 \geq R^2_\text{min}

This avoids rewarding transient spikes and instead encourages *sustained
directional change* — a more meaningful signal for genuine learning.

**Typical trajectory**

.. raw:: html

   <pre style="background:#1e293b; color:#e2e8f0; padding:12px 16px; border-radius:8px; font-size:12px; line-height:1.6; overflow-x:auto;">
   values over 5-window regression:

   spike (not rewarded)     sustained rise (rewarded)
   ▲  ·                      ▲          ·
   │ ╭╮                      │        ╭─╯
   │╭╯╰╮                     │      ╭─╯
   │╯  ╰──                   │    ╭─╯
   └──────────               └──────────
   slope ≈ 0, R² low         slope > 0, R² high
   </pre>

**When to use:**  Slow-learning paradigms (e.g. tinnitus suppression,
chronic pain, depression) where the target is gradual signal
reorganisation over minutes rather than fast-reacting moment-to-moment
control.

.. code-block:: python

    from mne_rt.protocols import LinearTrendProtocol

    proto = LinearTrendProtocol(direction="up", window=20, slope_threshold=0.0, min_r2=0.3)
    crossed, mag = proto.evaluate(alpha_power)

----

Sham Protocol
-------------

.. raw:: html

   <div style="background:#fff1f2; border-left:4px solid #f43f5e; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Design:</strong> &nbsp; wraps any inner protocol — on <em>sham_rate</em> fraction of windows,
   returns a randomly-drawn historical value instead of the real one.
   </div>

In a double-blind NF design the participant should not be able to tell
when feedback is real and when it is sham.  :class:`~mne_rt.protocols.ShamProtocol`
intercepts the inner protocol's output on a configurable fraction of
windows and substitutes a randomly-drawn historical reward value:

.. raw:: html

   <pre style="background:#1e293b; color:#e2e8f0; padding:12px 16px; border-radius:8px; font-size:12px; line-height:1.6; overflow-x:auto;">
   window:    1     2     3     4     5     6     7     8  …
   real:    (T,1) (F,0) (T,2) (F,0) (T,3) (F,0) (T,1) (F,0)
   output:  (T,1) (F,0) (T,1) (F,0) (T,3) (T,2) (T,1) (F,0)
   sham?:     no    no   yes    no    no   yes    yes    no
   </pre>

The inner protocol's state always advances correctly — only the
*delivered* feedback is sometimes replaced.  The ``sham_log`` attribute
records which windows were sham for post-session unblinding.

**Important:** the inner protocol must be configured separately; the
:class:`~mne_rt.protocols.ShamProtocol` wrapper is agnostic to the inner
protocol type.

**When to use:**  Any experiment that requires a within-session sham
control condition.  Set ``sham_rate=0.5`` for a 50/50 real/sham split,
or use ``rng_seed`` for exact reproducibility.

.. code-block:: python

    from mne_rt.protocols import ZScoreProtocol
    from mne_rt.protocols.sham import ShamProtocol

    inner = ZScoreProtocol(direction="up")
    proto = ShamProtocol(inner, sham_rate=0.5, rng_seed=42)
    for value in nf_stream:
        crossed, magnitude = proto.evaluate(value)

    # After session:
    sham_indices = [i for i, s in enumerate(proto.sham_log) if s]

----

Up-Down Staircase Protocol
--------------------------

.. raw:: html

   <div style="background:#ecfdf5; border-left:4px solid #10b981; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Design:</strong> &nbsp;
   threshold rises after n<sub>up</sub> successes, falls after n<sub>down</sub> failures
   — converges to a target success rate.
   </div>

The classic psychophysics staircase of :footcite:t:`levitt1971transformed`.
The threshold :math:`\theta` is adjusted after each window based on the
participant's recent success/failure run:

.. math::

   \theta_{t+1} = \begin{cases}
     \theta_t + \Delta & \text{after } n_\text{up} \text{ consecutive successes}\\
     \theta_t - \Delta & \text{after } n_\text{down} \text{ consecutive failures}
   \end{cases}

The step size :math:`\Delta` is halved after every
``n_reversals_before_halving`` direction reversals (a *reversal* is when
the staircase changes direction), zooming in progressively on the
participant's current threshold.

**Convergence levels** for common rules:

.. raw:: html

   <div style="overflow-x:auto; margin:8px 0 16px 0;">
   <table style="border-collapse:collapse; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <thead>
     <tr>
       <th style="background:#064e3b;color:white;padding:6px 14px;">Rule</th>
       <th style="background:#064e3b;color:white;padding:6px 14px;">Success rate</th>
       <th style="background:#064e3b;color:white;padding:6px 14px;">Use case</th>
     </tr>
   </thead>
   <tbody>
     <tr style="background:#ecfdf5;"><td style="padding:5px 14px;border:1px solid #d1fae5;">1-up / 1-down</td><td style="padding:5px 14px;border:1px solid #d1fae5;">50 %</td><td style="padding:5px 14px;border:1px solid #d1fae5;">Equal success/fail balance</td></tr>
     <tr style="background:#f0fdf4;"><td style="padding:5px 14px;border:1px solid #d1fae5;">1-up / 2-down</td><td style="padding:5px 14px;border:1px solid #d1fae5;">70.7 %</td><td style="padding:5px 14px;border:1px solid #d1fae5;">Standard; motivating default</td></tr>
     <tr style="background:#ecfdf5;"><td style="padding:5px 14px;border:1px solid #d1fae5;">1-up / 3-down</td><td style="padding:5px 14px;border:1px solid #d1fae5;">79.4 %</td><td style="padding:5px 14px;border:1px solid #d1fae5;">Easier start, more rewards early</td></tr>
   </tbody>
   </table>
   </div>

**Typical staircase trajectory**

.. raw:: html

   <pre style="background:#1e293b; color:#e2e8f0; padding:12px 16px; border-radius:8px; font-size:12px; line-height:1.6; overflow-x:auto;">
   threshold
   0.70  ·         ·─·
   0.65  ·─·     ·─╯ ╰─·
   0.60    ╰─·─·─╯     ╰─·
   0.55                  ╰─·─·
   0.50  initial             ╰──    (converging)
         1  2  3  4  5  6  7  8  9  window
         S  S  F  F  S  S  F  S  F   (S=success, F=fail, 1-up/2-down)
   reversal points ↑ stored in reversal_thresholds
   </pre>

**When to use:**  When you need an objective, data-driven estimate of
the participant's NF threshold — analogous to threshold tracking in
audiometry or psychophysics.

.. code-block:: python

    from mne_rt.protocols.staircase import UpDownStaircaseProtocol

    proto = UpDownStaircaseProtocol(
        initial_threshold=0.5, direction="up",
        n_up=1, n_down=2, step_size=0.05,
    )
    for value in nf_stream:
        crossed, mag = proto.evaluate(value)

    # After session — estimate perceptual threshold:
    threshold_estimate = float(np.mean(proto.reversal_thresholds[-6:]))

References: :footcite:t:`levitt1971transformed`, :footcite:t:`garcia1998forced`

----

Multi-Band Protocol
-------------------

.. raw:: html

   <div style="background:#f0f9ff; border-left:4px solid #0ea5e9; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Design:</strong> &nbsp;
   reward = (alpha ↑ AND theta ↓) &nbsp;|&nbsp; combined magnitude = √(mag<sub>up</sub> × mag<sub>down</sub>)
   </div>

Many clinical NF protocols target two frequency bands simultaneously —
reward alpha *up-regulation* while penalising theta *up-regulation*
(focus training), or reward SMR while suppressing theta (ADHD protocol
:footcite:p:`sterman2006foundation`).

:class:`~mne_rt.protocols.MultiBandProtocol` wraps two independent inner
protocols and combines their outputs:

.. math::

   \text{crossed} &= \text{crossed}_\text{up} \;\land\; \text{crossed}_\text{down}
   \quad \text{(AND, require\_both=True)} \\
   \text{magnitude} &= \sqrt{\text{mag}_\text{up} \times \text{mag}_\text{down}}
   \quad \text{(geometric mean)}

The **geometric mean** ensures both bands contribute equally: a very
large alpha reward cannot compensate for zero theta suppression —
both must be non-zero to produce a non-zero combined magnitude.

**AND vs OR logic**

.. raw:: html

   <pre style="background:#1e293b; color:#e2e8f0; padding:12px 16px; border-radius:8px; font-size:12px; line-height:1.6; overflow-x:auto;">
   window:     1     2     3     4
   alpha↑:   (T,2) (T,1) (F,0) (T,2)
   theta↓:   (F,0) (T,1) (T,1) (T,0.5)

   AND logic:  ✗     ✓     ✗     ✓     (both must cross)
   magnitude:  0   √1×1=1  0   √2×0.5≈1.0

   OR logic:   ✓     ✓     ✓     ✓     (either crosses)
   </pre>

**When to use:**  Any protocol that targets two brain rhythms
simultaneously.  The two inner protocols are independent and can be of
different types — e.g. a :class:`~mne_rt.protocols.ThresholdProtocol` for
alpha and a :class:`~mne_rt.protocols.ZScoreProtocol` for theta.

.. code-block:: python

    from mne_rt.protocols import ZScoreProtocol
    from mne_rt.protocols.multiband import MultiBandProtocol

    alpha_proto = ZScoreProtocol(direction="up",   warmup_windows=20)
    theta_proto = ZScoreProtocol(direction="down",  warmup_windows=20)

    proto = MultiBandProtocol(
        protocol_up=alpha_proto,
        protocol_down=theta_proto,
        require_both=True,
        up_label="alpha",
        down_label="theta",
    )
    for alpha_val, theta_val in zip(alpha_stream, theta_stream):
        crossed, magnitude = proto.evaluate(alpha_val, theta_val)

References: :footcite:t:`sterman2006foundation`

----

RL Protocol
-----------

.. raw:: html

   <div style="background:#fdf4ff; border-left:4px solid #a855f7; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Reward rule:</strong> &nbsp;
   threshold adapts via hit-rate error &nbsp;|&nbsp; ε-greedy exploration prevents early convergence
   </div>

A lightweight reinforcement-learning protocol that finds the right reward
threshold automatically, without requiring any manual calibration.  During a
short warmup phase no rewards are issued while the protocol collects
statistics.  Afterwards, the threshold :math:`\theta` is adjusted after each
window to keep the rolling hit rate close to the target:

.. math::

   \delta_t &= \hat{h}_t - h^* \qquad
   (\hat{h}_t = \text{rolling hit rate},\; h^* = \text{target hit rate}) \\[4pt]
   \theta_{t+1} &= \theta_t + \eta \cdot \delta_t
   \quad (\text{direction = "up"})

With probability :math:`\varepsilon` (the ``epsilon`` parameter) the current
value is treated as a *forced hit* (exploration step), preventing the threshold
from drifting so high that the participant never succeeds.

**When to use:**  When there is no prior calibration data and you want
the protocol to self-tune from scratch.  Works particularly well in
conjunction with :class:`~mne_rt.protocols.ShamProtocol` for within-session
sham control.

.. code-block:: python

    from mne_rt.protocols import RLProtocol

    proto = RLProtocol(
        direction="up",
        target_hit_rate=0.70,
        lr=0.01,
        epsilon=0.1,
        warmup_windows=20,
        history_len=50,
    )
    for value in nf_stream:
        crossed, mag = proto.evaluate(value)

----

Operant Protocol
----------------

.. raw:: html

   <div style="background:#fff7ed; border-left:4px solid #f97316; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Design:</strong> &nbsp; wraps any inner protocol — gates the reward output through a
   classical operant conditioning schedule (FR / VR / FI / VI).
   </div>

Partial reinforcement schedules are more resistant to extinction than
continuous reinforcement :footcite:t:`ferster1957schedules`.
:class:`~mne_rt.protocols.OperantProtocol` wraps any existing NF protocol and
filters its reward output through one of four classical schedules:

.. raw:: html

   <div style="overflow-x:auto; margin:8px 0 16px 0;">
   <table style="border-collapse:collapse; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <thead>
     <tr>
       <th style="background:#431407;color:white;padding:6px 14px;">Schedule</th>
       <th style="background:#431407;color:white;padding:6px 14px;">Abbreviation</th>
       <th style="background:#431407;color:white;padding:6px 14px;">Rule</th>
     </tr>
   </thead>
   <tbody>
     <tr style="background:#fff7ed;"><td style="padding:5px 14px;border:1px solid #fed7aa;">Fixed Ratio</td><td style="padding:5px 14px;border:1px solid #fed7aa;">FR</td><td style="padding:5px 14px;border:1px solid #fed7aa;">Reward on every <em>N</em>-th hit</td></tr>
     <tr style="background:#fffbeb;"><td style="padding:5px 14px;border:1px solid #fed7aa;">Variable Ratio</td><td style="padding:5px 14px;border:1px solid #fed7aa;">VR</td><td style="padding:5px 14px;border:1px solid #fed7aa;">Each hit rewarded with probability 1/<em>N</em></td></tr>
     <tr style="background:#fff7ed;"><td style="padding:5px 14px;border:1px solid #fed7aa;">Fixed Interval</td><td style="padding:5px 14px;border:1px solid #fed7aa;">FI</td><td style="padding:5px 14px;border:1px solid #fed7aa;">First hit after exactly <em>T</em> seconds is rewarded</td></tr>
     <tr style="background:#fffbeb;"><td style="padding:5px 14px;border:1px solid #fed7aa;">Variable Interval</td><td style="padding:5px 14px;border:1px solid #fed7aa;">VI</td><td style="padding:5px 14px;border:1px solid #fed7aa;">First hit after a random interval (mean <em>T</em>) is rewarded</td></tr>
   </tbody>
   </table>
   </div>

The inner protocol's state always advances regardless of whether the schedule
releases a reward — so running statistics (z-score, staircase threshold, etc.)
continue to update correctly.

**When to use:**  Whenever reduced reward density is experimentally desirable
(e.g. to study partial reinforcement extinction effects in NF, or to maintain
engagement over long sessions by preventing saturation).

.. code-block:: python

    from mne_rt.protocols import ZScoreProtocol, OperantProtocol

    inner = ZScoreProtocol(direction="up", warmup_windows=20)
    proto = OperantProtocol(inner, schedule="VR", ratio=3, rng_seed=42)
    for value in nf_stream:
        crossed, mag = proto.evaluate(value)

References: :footcite:t:`ferster1957schedules`

----

Transfer Protocol
-----------------

.. raw:: html

   <div style="background:#f0f9ff; border-left:4px solid #0ea5e9; padding:10px 16px; margin-bottom:16px; border-radius:0 8px 8px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px;">
   <strong>Design:</strong> &nbsp; seeds running z-score statistics from a prior session file —
   rewards from the very first window of the new session.
   </div>

The standard :class:`~mne_rt.protocols.ZScoreProtocol` needs a warmup phase to
estimate :math:`\hat\mu` and :math:`\hat\sigma` before rewards can be issued.
:class:`~mne_rt.protocols.TransferProtocol` eliminates warmup by loading the
population statistics from a previous session's ``beh.json`` file and seeding
the Welford accumulators directly:

.. math::

   \hat\mu_0 = \bar{x}_\mathrm{prior},
   \qquad
   \hat\sigma_0 = \sigma_\mathrm{prior},
   \qquad
   n_0 = N_\mathrm{prior}

From the first window onward, each new value is z-scored against this
informed prior and updates the statistics via Welford's algorithm, gradually
replacing the prior with session-specific data.

**Session file format** (BIDS-compatible ``beh.json``)::

    {
      "meta": {"modalities": ["sensor_power"]},
      "data": {"sensor_power": [0.12, 0.14, 0.11, …]}
    }

Such files are written automatically by :meth:`~mne_rt.RTStream.save`.

**When to use:**  Multi-session training programmes where consistent reward
rates across sessions improve participant motivation.  Also useful in studies
where the first session's statistics serve as the participant's personalised
baseline.

.. code-block:: python

    from mne_rt.protocols import TransferProtocol

    proto = TransferProtocol(
        fname="sub-01_ses-01_task-nf_beh.json",
        modality="sensor_power",
        direction="up",
        zscore_threshold=0.5,
    )
    for value in nf_stream:
        crossed, mag = proto.evaluate(value)

----

References
-------------

.. footbibliography::
