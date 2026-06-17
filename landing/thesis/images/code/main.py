from pathlib import Path
from ui import ControlPanel


def main() -> None:
    panel = ControlPanel(Path(__file__).resolve().parent)
    panel.run()


if __name__ == "__main__":
    main()