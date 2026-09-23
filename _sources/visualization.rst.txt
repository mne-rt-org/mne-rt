.. _visualization:

Visualisation
=============

Multiple purpose-built live display windows — each dark-themed, Qt-native,
and fully thread-safe. Push data from a background acquisition thread; 
all rendering happens on the main Qt thread via an internal 30 Hz timer.

.. tabs::

   .. tab:: RawPlot

      .. raw:: html

         <div class="viz-tab-card">
           <div class="viz-tab-header">
             <div class="viz-tab-title-row">
               <h3 class="viz-tab-title">RawPlot</h3>
             </div>
             <p class="viz-tab-subtitle">Multi-channel scrolling M/EEG raw signal viewer</p>
           </div>
           <div class="viz-tab-body">
             <p class="viz-tab-desc">
               Dark-themed multi-channel raw signal display.  All channels are stacked
               vertically with colour-coded traces and channel-name Y-axis labels,
               modelled on the standard M/EEG browser layout.
               All signal processing is applied <em>from now</em>: enabling a filter
               or re-reference leaves existing buffer data intact and processes only
               newly incoming chunks.  A built-in Riemannian Potato detector can
               automatically flag artefacted segments in real time.
             </p>
             <div class="viz-feature-grid">
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Left-click</strong> Y-axis label → toggle bad channel (greyed trace)</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Double-click</strong> signal area → mark bad-segment start / end</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Online filter</strong> — HP / LP / band-pass / notch (causal SOS)</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Re-reference</strong> — Average, Mastoid (TP9/TP10), any single channel</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Artifact correction</strong> — LMS, ASR, GEDAI, ORICA, Maxwell SSS</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Riemannian Potato</strong> — automatic bad-segment detection (pyriemann)</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>SSP projector</strong> checkbox when <code>info</code> is provided</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Export</strong> bad segments via <code>raw_plot.to_annotations()</code></span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Vertical scrollbar + mouse-wheel channel paging</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">÷2 / ×2 amplitude scale · DC-removal toggle</span>
               </div>
             </div>
             <div class="viz-media-wrap">
               <video autoplay muted loop playsinline>
                 <source src="_static/RawPlot.mp4" type="video/mp4">
               </video>
             </div>
           </div>
         </div>

      .. code-block:: python

         raw_plot = RawPlot(
             ch_names=info["ch_names"],
             sfreq=1000,
             info=info,       # enables SSP projector checkbox
         )
         raw_plot.show()
         # background acquisition thread:
         raw_plot.push(chunk)   # shape (n_channels, n_samples)

      See :class:`mne_rt.viz.RawPlot` for the full API reference.

   .. tab:: EpochPlot

      .. raw:: html

         <div class="viz-tab-card">
           <div class="viz-tab-header">
             <div class="viz-tab-title-row">
               <h3 class="viz-tab-title">EpochPlot</h3>
             </div>
             <p class="viz-tab-subtitle">Scrolling viewer with live epoch and trigger overlays</p>
           </div>
           <div class="viz-tab-body">
             <p class="viz-tab-desc">
               Same dark-themed scrolling display as RawPlot, extended with visual
               epoch markers.  Each stimulus event (<code>push_trigger</code>) draws three
               overlays in a per-condition colour: a <strong>solid vertical line</strong>
               at t&nbsp;=&nbsp;0, a <strong>semi-transparent shaded band</strong> spanning
               [tmin, tmax], and <strong>dashed boundary lines</strong> at the epoch edges.
               The epoch window is adjustable live without stopping the stream.
               Event colours are distinct (green, cyan, yellow, …).
             </p>
             <div class="viz-feature-grid">
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Trigger line</strong> — solid vertical at t = 0 per condition</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Epoch band</strong> — semi-transparent [tmin, tmax] shading</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Live tmin / tmax</strong> spinboxes — adjust without restarting</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Per-condition colour legend</strong> from event_id mapping</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Time window options: 2, 5, 10, 20 seconds</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Left-click</strong> an epoch's shaded band → mark it bad (red); click again to unmark</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Dashed boundary lines · Pause / Resume · Clear · Screenshot</span>
               </div>
             </div>
             <div class="viz-media-wrap">
               <video autoplay muted loop playsinline>
                 <source src="_static/EpochPlot.mp4" type="video/mp4">
               </video>
             </div>
           </div>
         </div>

      .. code-block:: python

         ep = EpochPlot(
             ch_names=info["ch_names"],
             sfreq=1000,
             event_id={"target": 1, "standard": 2},
             tmin=-0.1,
             tmax=0.5,
         )
         ep.show()
         ep.push(chunk)       # continuous data — shape (n_ch, n_samples)
         ep.push_trigger(1)   # mark event code 1 at the current position
         ...
         ep.bad_epoch_ids     # -> [2, 5]  (ids the user marked bad by clicking)

      See :class:`mne_rt.viz.EpochPlot` for the full API reference.

   .. tab:: NFPlot

      .. raw:: html

         <div class="viz-tab-card">
           <div class="viz-tab-header">
             <div class="viz-tab-title-row">
               <h3 class="viz-tab-title">NFPlot</h3>
             </div>
             <p class="viz-tab-subtitle">Scrolling real-time neurofeedback signal monitor</p>
           </div>
           <div class="viz-tab-body">
             <p class="viz-tab-desc">
               A scrolling dark-themed multi-channel NF feature monitor.
               One colour-coded trace per active neurofeedback modality is updated
               live as each analysis window completes.  The amplitude scale is
               adjustable on the fly and each modality gets a distinct hue for
               instant visual separation.  When a reward protocol is active,
               its current threshold (fixed or adaptive) is overlaid as a
               dashed horizontal line, and windows where the subject is
               currently being rewarded are washed in a translucent green
               span behind the trace.
             </p>
             <div class="viz-feature-grid">
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>One trace per modality</strong> — colour-coded scrolling lines</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>÷2 / ×2 scale buttons</strong> with live µV readout</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Configurable time window</strong> — 5 s to 60 s</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Live threshold line</strong> — fixed or adaptive, toggleable</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Reward-on span</strong> — scrolling green wash, toggleable</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Pause / Resume</strong> and screenshot export</span>
               </div>
             </div>
             <div class="viz-media-wrap">
               <video autoplay muted loop playsinline>
                 <source src="_static/NFPlot.mp4" type="video/mp4">
               </video>
             </div>
           </div>
         </div>

      .. code-block:: python

         nf_plot = NFPlot(
             modalities=["alpha", "beta", "gamma"],
             scales_dict={"alpha": 1e-12, "beta": 1e-12, "gamma": 1e-12},
             sfreq=30,
         )
         nf_plot.show()
         # inside the acquisition loop:
         nf_plot.push([alpha_val, beta_val, gamma_val])
         # with a per-modality threshold line (None = no line for that modality):
         nf_plot.push([alpha_val, beta_val, gamma_val], thresholds=[alpha_thr, None, None])
         # with reward spans too (True/False = reward on/off, None = no protocol):
         nf_plot.push(
             [alpha_val, beta_val, gamma_val],
             thresholds=[alpha_thr, None, None],
             rewards=[alpha_rewarded, None, None],
         )

      :meth:`~mne_rt.RTStream.record_main` drives
      ``push(..., thresholds=..., rewards=...)`` automatically from each
      modality's protocol — see :doc:`protocols`.
      See :class:`mne_rt.viz.NFPlot` for the full API reference.

   .. tab:: ButterflyPlot

      .. raw:: html

         <div class="viz-tab-card">
           <div class="viz-tab-header">
             <div class="viz-tab-title-row">
               <h3 class="viz-tab-title">ButterflyPlot</h3>
             </div>
             <p class="viz-tab-subtitle">All channels overlaid per condition with region colour gradient</p>
           </div>
           <div class="viz-tab-body">
             <p class="viz-tab-desc">
               All channels overlaid in a single panel per condition, coloured by scalp
               region:
               <span style="color:#58a6ff;">blue (frontal)</span> →
               <span style="color:#4dd0e1;">cyan</span> →
               <span style="color:#66bb6a;">green</span> →
               <span style="color:#ffb74d;">amber</span> →
               <span style="color:#ef5350;">red (occipital)</span>.
               The gradient makes spatial ERP patterns instantly readable and outlier
               channels visually obvious without any manual channel selection.
             </p>
             <div class="viz-feature-grid">
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>One panel per condition</strong> — all channels overlaid</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Region gradient</strong> — frontal blue → occipital red</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Trial counts</strong> — shown per condition panel header</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Y-scale, time window, line width, grid, export PNG</span>
               </div>
             </div>
             <div class="viz-media-wrap">
               <video autoplay muted loop playsinline>
                 <source src="_static/ButterflyPlot.mp4" type="video/mp4">
               </video>
             </div>
           </div>
         </div>

      .. code-block:: python

         butt = ButterflyPlot(
             ch_names=ch_names,
             sfreq=1000,
             tmin=-0.1,
             tmax=0.5,
             event_id={"left": 1, "right": 2},
             montage="easycap-M1",
         )
         butt.update(epochs, conditions)

      See :class:`mne_rt.viz.ButterflyPlot` for the full API reference.

   .. tab:: CompareEvoked

      .. raw:: html

         <div class="viz-tab-card">
           <div class="viz-tab-header">
             <div class="viz-tab-title-row">
               <h3 class="viz-tab-title">CompareEvoked</h3>
             </div>
             <p class="viz-tab-subtitle">Per-channel large plots with all conditions overlaid and peak markers</p>
           </div>
           <div class="viz-tab-body">
             <p class="viz-tab-desc">
               Large individual plots per electrode with all conditions overlaid,
               ±1 SEM shading, and a scatter dot marking the peak latency in the
               post-stimulus window.  Channels are selected by clicking on a
               clickable mini scalp-topomap in the sidebar — making it trivially
               easy to drill into any electrode without hunting through a list.
             </p>
             <div class="viz-feature-grid">
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Click-to-select</strong> channels via sidebar topomap</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>±1 SEM shading</strong> per condition</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Peak-latency markers</strong> — auto-detected in post-stimulus window</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Visible ms and amplitude axes with gridlines</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Y-scale, SEM and peak toggles, export PNG</span>
               </div>
             </div>
             <div class="viz-media-wrap">
               <video autoplay muted loop playsinline>
                 <source src="_static/CompareEvoked.mp4" type="video/mp4">
               </video>
             </div>
           </div>
         </div>

      .. code-block:: python

         cmp = CompareEvoked(
             ch_names=ch_names,
             sfreq=1000,
             tmin=-0.1,
             tmax=0.5,
             event_id={"left": 1, "right": 2},
             montage="easycap-M1",
         )
         cmp.update(epochs, conditions)

      See :class:`mne_rt.viz.CompareEvoked` for the full API reference.

   .. tab:: TopoPlot

      .. raw:: html

         <div class="viz-tab-card">
           <div class="viz-tab-header">
             <div class="viz-tab-title-row">
               <h3 class="viz-tab-title">TopoPlot</h3>
             </div>
             <p class="viz-tab-subtitle">Scalp-layout evoked display — one mini plot per electrode</p>
           </div>
           <div class="viz-tab-body">
             <p class="viz-tab-desc">
               One mini plot per electrode placed at its true 2-D scalp position
               (from <code>mne.channels.find_layout</code>).  Condition averages
               with ±1 SEM shading re-render after every accepted trial, giving an
               instant whole-brain view of the evolving ERP.
               Re-referencing, smoothing, and y-scale are all adjustable from the sidebar.
             </p>
             <div class="viz-feature-grid">
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Scalp-layout grid</strong> — true electrode positions (EEG &amp; MEG)</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>±1 SEM shading</strong> per condition, updates every trial</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Re-reference</strong> — None / Average / Mastoids</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Unit-aware</strong> — µV, fT, fT/cm auto-detected</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Stimulus-time marker · channel labels · export PNG</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Y-scale, line width, smoothing, background colour controls</span>
               </div>
             </div>
             <div class="viz-media-wrap">
               <video autoplay muted loop playsinline>
                 <source src="_static/TopoPlot.mp4" type="video/mp4">
               </video>
             </div>
           </div>
         </div>

      .. code-block:: python

         topo = TopoPlot(
             ch_names=ch_names,
             sfreq=1000,
             tmin=-0.1,
             tmax=0.5,
             event_id={"left": 1, "right": 2},
             montage="easycap-M1",
         )
         topo.update(epochs, conditions)   # call after each accepted trial

      See :class:`mne_rt.viz.TopoPlot` for the full API reference.

   .. tab:: TopomapPlot

      .. raw:: html

         <div class="viz-tab-card">
           <div class="viz-tab-header">
             <div class="viz-tab-title-row">
               <h3 class="viz-tab-title">TopomapPlot</h3>
             </div>
             <p class="viz-tab-subtitle">Live per-band power scalp topographic map</p>
           </div>
           <div class="viz-tab-body">
             <p class="viz-tab-desc">
               Live scalp topographic map that displays the spatial distribution of
               per-band power in real time.  Built on MNE-Python's matplotlib pipeline
               with a Qt canvas, it refreshes continuously as new data windows arrive.
               Electrode positions are overlaid and a custom frequency range can be
               typed directly into the sidebar without restarting.
             </p>
             <div class="viz-feature-grid">
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Per-band power</strong> — interpolated scalp colour map</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Built-in bands</strong> — delta, theta, alpha, beta, gamma</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Custom frequency range</strong> — set any lo–hi Hz in the sidebar</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Electrode overlay</strong> — sensor positions on the scalp map</span>
               </div>
             </div>
           <div class="viz-media-wrap">
               <video autoplay muted loop playsinline>
                 <source src="_static/TopomapPlot.mp4" type="video/mp4">
               </video>
             </div>
           </div>
         </div>

      .. code-block:: python

         tmap = TopomapPlot(
             info=raw_info,
             sfreq=1000,
             frange=(8, 13),       # start with the alpha band
         )
         tmap.show()
         tmap.push(data_window)    # shape (n_channels, n_samples)

      See :class:`mne_rt.viz.TopomapPlot` for the full API reference.

   .. tab:: TFRPlot

      .. raw:: html

         <div class="viz-tab-card">
           <div class="viz-tab-header">
             <div class="viz-tab-title-row">
               <h3 class="viz-tab-title">TFRPlot</h3>
             </div>
             <p class="viz-tab-subtitle">Morlet wavelet time-frequency heatmaps across conditions</p>
           </div>
           <div class="viz-tab-body">
             <p class="viz-tab-desc">
               Morlet wavelet TFR heatmaps arranged in a (channels × conditions) grid.
               Two modes: <em>induced</em> (average of per-epoch TFR) and
               <em>evoked</em> (TFR of the trial average).
               Colour limits are shared across conditions for direct comparison.
               Channels are selected via a clickable sidebar topomap — the same
               click-to-inspect paradigm used in CompareEvoked.
             </p>
             <div class="viz-feature-grid">
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Induced vs Evoked</strong> mode toggle</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Click-to-select</strong> channels via sidebar topomap</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>dB baseline</strong> correction or raw power display</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Shared colour scale</strong> across all conditions</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Colormaps</strong> — Hot, RdBu, Viridis, Plasma, Turbo, Greys</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Manual vmin / vmax or auto percentile limits</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Frequency range sliders · export PNG</span>
               </div>
             </div>
             <div class="viz-media-wrap">
               <video autoplay muted loop playsinline>
                 <source src="_static/TFRPlot.mp4" type="video/mp4">
               </video>
             </div>
           </div>
         </div>

      .. code-block:: python

         import numpy as np
         tfr = TFRPlot(
             ch_names=ch_names,
             sfreq=1000,
             tmin=-0.1,
             tmax=0.5,
             freqs=np.arange(4, 40),
             event_id={"left": 1, "right": 2},
             montage="easycap-M1",
         )
         tfr.update(epochs, conditions)

      See :class:`mne_rt.viz.TFRPlot` for the full API reference.

   .. tab:: BrainPlot

      .. raw:: html

         <div class="viz-tab-card">
           <div class="viz-tab-header">
             <div class="viz-tab-title-row">
               <h3 class="viz-tab-title">BrainPlot</h3>
             </div>
             <p class="viz-tab-subtitle">Interactive 3-D cortical surface with colour-mapped source activity</p>
           </div>
           <div class="viz-tab-body">
             <p class="viz-tab-desc">
               Interactive 3-D cortical surface rendered with PyVista, with colour-mapped
               source-space activity that updates live as new NF windows arrive.
               Hemisphere toggles, surface switching, anatomical parcellation borders,
               and a full set of view presets make it the most feature-rich display in
               the suite.  Video recording (MP4) is built in.
             </p>
             <div class="viz-feature-grid">
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Live colour map</strong> — EMA-smoothed source-space activity</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Hemisphere toggles</strong> — left / right / both</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Surface modes</strong> — inflated, pial, white, sphere</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Parcellation borders</strong> — aparc (DK) or aparc.a2009s (Destrieux)</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Display modes</strong> — Alpha, Beta, Theta, Gamma, SMR, custom band</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>5 view presets</strong> — lateral L/R, dorsal, frontal, ventral + key bindings</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text">Colormap, opacity, threshold, clim, background controls</span>
               </div>
               <div class="viz-feature-item">
                 <span class="viz-fi-icon">✓</span>
                 <span class="viz-fi-text"><strong>Video recording</strong> — MP4 export via imageio-ffmpeg</span>
               </div>
             </div>
             <div class="viz-media-wrap">
               <video autoplay muted loop playsinline>
                 <source src="_static/BrainPlot.mp4" type="video/mp4">
               </video>
             </div>
           </div>
         </div>

      .. code-block:: python

         brain = BrainPlot(
             subjects_fs_dir="/path/to/freesurfer",
             surf="inflated",
             clim=[0, 0.6],
             display_smoothing=5,
         )
         brain.show()
         brain.push(stc_data)   # source-space activity array

      See :class:`mne_rt.viz.BrainPlot` for the full API reference.
