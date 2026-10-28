"""Booth/stage dashboard: three panes (robot view | your view | twin), a
Curio on/off switch, and the latency waterfall.

For now `curio-sim shot` renders the same three-pane layout to images (see
sim/shots.py). TODO(owner D): live version as a small local web page served
by a stdlib HTTP server (MJPEG frames, the runtime tracer's waterfall), with keyboard
controls: Space = Curio on/off, R = recalibrate, N = next scenario.
"""
