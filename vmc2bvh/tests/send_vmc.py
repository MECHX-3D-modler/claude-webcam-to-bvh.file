"""Stream the synthetic animation as VMC OSC packets (like a VMC performer)."""
import sys, time
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder
from synth import pose

port = int(sys.argv[1]) if len(sys.argv) > 1 else 39539
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
c = SimpleUDPClient("127.0.0.1", port)
t0 = time.perf_counter()
while (t := time.perf_counter() - t0) < dur:
    bones, root = pose(t)
    b = OscBundleBuilder(IMMEDIATELY)
    def msg(addr, *a):
        m = OscMessageBuilder(addr)
        for v in a: m.add_arg(v)
        b.add_content(m.build())
    msg("/VMC/Ext/OK", 1)
    msg("/VMC/Ext/T", float(t))
    msg("/VMC/Ext/Root/Pos", "root", *map(float, root[0]), *map(float, root[1]))
    for n, (p, q) in bones.items():
        msg("/VMC/Ext/Bone/Pos", n, *map(float, p), *map(float, q))
    c.send(b.build())
    time.sleep(1/90)
