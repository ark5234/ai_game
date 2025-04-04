# main.py
import tkinter as tk
from game_ui import GameUI

def main():
    root = tk.Tk()
    game_ui = GameUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
