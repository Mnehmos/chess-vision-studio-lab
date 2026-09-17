"""A minimal fake UCI engine used by the Stockfish provider tests.

Speaks just enough UCI to exercise `cvslab.funnel.stockfish` without shipping a 114 MB binary:
options, a readiness handshake, a net announcement, and one deterministic search whose mate
reporting depends on the position so the mate convention can be tested.
"""
import sys

sys.stdout.reconfigure(line_buffering=True)   # a pipe would otherwise buffer and deadlock the client

MATE_FEN_MARKER = "8/8/8/8/8/5K2/6Q1/6k1 w"  # any FEN containing this prefix mimes a mate score


def main(argv: list[str]) -> int:
    transcript = None
    if "--transcript" in argv:
        transcript = open(argv[argv.index("--transcript") + 1], "a", encoding="utf-8")
    fen = ""
    nodes_requested = 0
    while True:
        raw = sys.stdin.readline()      # readline, not iteration: iteration read-aheads and deadlocks
        if not raw:
            return 0
        line = raw.strip()
        if transcript is not None:
            transcript.write(line + "\n")
            transcript.flush()
        if line == "uci":
            print("id name FakeFish 1.0")
            print("id author cvslab tests")
            print("option name Threads type spin default 1 min 1 max 1024")
            print("option name Hash type spin default 16 min 1 max 33554432")
            print("option name MultiPV type spin default 1 min 1 max 500")
            print("uciok")
        elif line.startswith("setoption"):
            pass
        elif line == "isready":
            print("readyok")
        elif line.startswith("position fen "):
            fen = line[len("position fen "):]
        elif line.startswith("go nodes "):
            nodes_requested = int(line.split()[2])
            print(f"info string NNUE evaluation using nn-test.nnue (1MiB, (1,2,3))")
            if MATE_FEN_MARKER in fen:
                print(f"info depth 11 seldepth 13 multipv 1 score mate 3 nodes {nodes_requested} "
                      f"time 9 pv g2g7 g1h1 g7h7")
                print("bestmove g2g7")
            else:
                print(f"info depth 7 seldepth 9 multipv 1 score cp 42 nodes {nodes_requested} "
                      f"time 5 pv e2e4 e7e5 g1f3")
                print("bestmove e2e4")
        elif line == "quit":
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
