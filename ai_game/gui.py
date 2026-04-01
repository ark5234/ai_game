"""Pygame GUI entry point.

Usage:
    python -m ai_game.gui [--rl]

Screens:
  MENU   — select/create profile, start game, view stats, toggle AI type
  GAME   — HP/MP bars, round counter, damage events, ensemble confidence
  END    — winner, summary stats, plot damage button
  STATS  — per-profile statistics
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Helpers used across screens
# ---------------------------------------------------------------------------

def _txt(surface, text, x, y, fnt, color, center=False):
    import pygame
    surf = fnt.render(str(text), True, color)
    rect = surf.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(surf, rect)


def _bar(surface, x, y, w, h, val, max_val, fg, bg=(50, 50, 50)):
    import pygame
    pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=4)
    filled = int(w * max(0, min(val, max_val)) / max_val)
    if filled > 0:
        pygame.draw.rect(surface, fg, (x, y, filled, h), border_radius=4)


def _btn(surface, x, y, w, h, label, fnt, color, hovered=False):
    import pygame
    col = tuple(min(255, c + 30) for c in color) if hovered else color
    pygame.draw.rect(surface, col, (x, y, w, h), border_radius=8)
    _txt(surface, label, x + w // 2, y + h // 2, fnt, (240, 240, 240), center=True)
    return pygame.Rect(x, y, w, h)


# ---------------------------------------------------------------------------
# Game state container (avoids nonlocal juggling)
# ---------------------------------------------------------------------------

class _State:
    def __init__(self):
        self.screen = "menu"
        self.selected_profile = "Player"
        self.profile_input_active = False
        self.profile_input_text = ""
        self.use_rl = False

        # Per-game objects
        self.engine = None
        self.player = None
        self.ai_fighter = None
        self.tracker = None
        self.rl_agent = None
        self.logs = []
        self.end_plots = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Fighting Game — GUI mode")
    parser.add_argument("--rl", action="store_true", help="Use RL agent instead of ML ensemble")
    args = parser.parse_args()

    try:
        import pygame
    except ImportError:
        print("pygame is not installed. Run: pip install pygame")
        sys.exit(1)

    from .fighter import Fighter
    from .ai_opponent import AdaptiveAIOpponent
    from .rl_agent import QLearningAgent
    from .profiles import PlayerProfile
    from .damage_tracker import MatchTracker
    from .battle_engine import BattleEngine
    from .visualize import plot_damage_per_round, plot_cumulative_damage

    pygame.init()
    W, H = 1000, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("AI Fighting Game")
    clock = pygame.time.Clock()

    # Colours
    BLACK  = (10,  10,  10)
    WHITE  = (240, 240, 240)
    RED    = (220,  50,  50)
    GREEN  = ( 50, 200,  50)
    BLUE   = ( 50, 120, 220)
    YELLOW = (240, 210,  50)
    GRAY   = (100, 100, 100)
    DGRAY  = ( 50,  50,  50)
    LBLUE  = (100, 160, 240)
    ORANGE = (240, 140,  50)

    # Fonts
    f_sm = pygame.font.Font(None, 26)
    f_md = pygame.font.Font(None, 34)
    f_lg = pygame.font.Font(None, 52)
    f_xl = pygame.font.Font(None, 70)

    st = _State()
    st.use_rl = args.rl

    # ------------------------------------------------------------------
    def new_game():
        st.logs = []
        st.end_plots = []
        st.player = Fighter(st.selected_profile)
        st.ai_fighter = AdaptiveAIOpponent("AI")
        st.ai_fighter.load_history()

        st.rl_agent = None
        if st.use_rl:
            st.rl_agent = QLearningAgent()
            st.rl_agent.load()

        st.tracker = MatchTracker()
        st.engine = BattleEngine(
            st.player, st.ai_fighter,
            rl_agent=st.rl_agent,
            use_rl=st.use_rl,
            tracker=st.tracker,
        )

    def finalize_match():
        if st.tracker:
            st.tracker.save()
        profile = PlayerProfile.load_or_create(st.selected_profile)
        if st.engine and st.player:
            profile.record_match(
                won=(st.engine.winner == st.selected_profile),
                damage_dealt=st.player.total_damage_dealt,
                damage_taken=st.player.total_damage_taken,
                moves=st.player.total_moves,
                move_usage=st.player.move_usage,
            )
            profile.save()
        if st.tracker:
            p1 = plot_damage_per_round(st.tracker.match_id)
            p2 = plot_cumulative_damage(st.tracker.match_id)
            st.end_plots = [p for p in (p1, p2) if p]

    # ------------------------------------------------------------------
    running = True
    while running:
        screen.fill(BLACK)
        mx, my = pygame.mouse.get_pos()
        events = pygame.event.get()

        for ev in events:
            if ev.type == pygame.QUIT:
                running = False

        # ==============================================================
        if st.screen == "menu":
        # ==============================================================
            _txt(screen, "AI Fighting Game", W // 2, 60, f_xl, YELLOW, center=True)
            _txt(screen, "Ensemble ML + RL Opponent", W // 2, 118, f_sm, GRAY, center=True)

            # Profile input
            _txt(screen, "Profile", 80, 170, f_md, WHITE)
            input_rect = pygame.Rect(80, 205, 320, 38)
            pygame.draw.rect(screen, BLUE if st.profile_input_active else DGRAY,
                             input_rect, border_radius=6)
            pygame.draw.rect(screen, GRAY, input_rect, 2, border_radius=6)
            disp = (st.profile_input_text + "|") if st.profile_input_active else \
                   (st.profile_input_text or "Type new profile name…")
            _txt(screen, disp, 90, 215, f_sm,
                 WHITE if st.profile_input_text else GRAY)

            r_create = _btn(screen, 415, 205, 130, 38, "Create", f_sm, BLUE,
                            pygame.Rect(415, 205, 130, 38).collidepoint(mx, my))

            # Profile list
            _txt(screen, "Select:", 80, 257, f_sm, GRAY)
            profiles = PlayerProfile.list_profiles()
            prof_rects = []
            for i, pname in enumerate(profiles[:6]):
                col = GREEN if pname == st.selected_profile else DGRAY
                r = pygame.Rect(80 + i * 145, 280, 135, 34)
                pygame.draw.rect(screen, col, r, border_radius=6)
                _txt(screen, pname[:14], 86 + i * 145, 289, f_sm)
                prof_rects.append((r, pname))

            # AI toggle
            ai_lbl = "AI: RL Agent" if st.use_rl else "AI: ML Ensemble"
            r_toggle = _btn(screen, 80, 338, 220, 38, ai_lbl, f_sm, ORANGE,
                            pygame.Rect(80, 338, 220, 38).collidepoint(mx, my))

            # Main action buttons
            r_start = _btn(screen, 80, 400, 240, 52, "Start Game", f_md, GREEN,
                           pygame.Rect(80, 400, 240, 52).collidepoint(mx, my))
            r_stats = _btn(screen, 340, 400, 240, 52, "View Stats", f_md, LBLUE,
                           pygame.Rect(340, 400, 240, 52).collidepoint(mx, my))
            r_quit  = _btn(screen, 600, 400, 160, 52, "Quit", f_md, RED,
                           pygame.Rect(600, 400, 160, 52).collidepoint(mx, my))

            for ev in events:
                if ev.type == pygame.KEYDOWN:
                    if st.profile_input_active:
                        if ev.key == pygame.K_RETURN:
                            st.profile_input_active = False
                        elif ev.key == pygame.K_BACKSPACE:
                            st.profile_input_text = st.profile_input_text[:-1]
                        elif len(st.profile_input_text) < 20:
                            st.profile_input_text += ev.unicode
                    if ev.key == pygame.K_ESCAPE:
                        st.profile_input_active = False

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if input_rect.collidepoint(mx, my):
                        st.profile_input_active = True
                    else:
                        st.profile_input_active = False

                    if r_create.collidepoint(mx, my) and st.profile_input_text.strip():
                        st.selected_profile = st.profile_input_text.strip()
                        # Pre-create the profile file
                        PlayerProfile(st.selected_profile).save()
                        st.profile_input_text = ""

                    for r, pname in prof_rects:
                        if r.collidepoint(mx, my):
                            st.selected_profile = pname

                    if r_toggle.collidepoint(mx, my):
                        st.use_rl = not st.use_rl

                    if r_start.collidepoint(mx, my):
                        new_game()
                        st.screen = "game"

                    if r_stats.collidepoint(mx, my):
                        st.screen = "stats"

                    if r_quit.collidepoint(mx, my):
                        running = False

        # ==============================================================
        elif st.screen == "game":
        # ==============================================================
            if st.engine is None:
                st.screen = "menu"
                continue

            p   = st.player
            aif = st.ai_fighter
            eng = st.engine

            # Header
            _txt(screen, f"Round {eng.round_num}", W // 2, 18, f_md, YELLOW, center=True)
            ai_lbl = "[RL AI]" if st.use_rl else "[ML Ensemble]"
            _txt(screen, ai_lbl, W - 140, 18, f_sm, GRAY)

            # Player bars
            _txt(screen, p.name, 40, 52, f_md, GREEN)
            _bar(screen, 40, 78, 340, 22, p.health, 100, GREEN)
            _txt(screen, f"HP {p.health}/100", 390, 78, f_sm, GREEN)
            _bar(screen, 40, 106, 340, 14, p.mp, 50, BLUE)
            _txt(screen, f"MP {p.mp}/50", 390, 106, f_sm, BLUE)

            # AI bars
            _txt(screen, "AI", W - 380, 52, f_md, RED)
            _bar(screen, W - 380, 78, 330, 22, aif.health, 100, RED)
            _txt(screen, f"HP {aif.health}/100", W - 380 - 115, 78, f_sm, RED)
            _bar(screen, W - 380, 106, 330, 14, aif.mp, 50, ORANGE)
            _txt(screen, f"MP {aif.mp}/50", W - 380 - 115, 106, f_sm, ORANGE)

            # Ensemble confidence
            if not st.use_rl:
                c = aif.last_confidences
                ctxt = (f"Ensemble: {aif.ensemble_confidence:.1f}%   "
                        f"RF:{c.get('rf',0):.0f}%  "
                        f"NN:{c.get('nn',0):.0f}%  "
                        f"NB:{c.get('nb',0):.0f}%")
                _txt(screen, ctxt, W // 2, 132, f_sm, GRAY, center=True)

            # Battle log
            pygame.draw.rect(screen, DGRAY, (40, 155, W - 80, 395), border_radius=6)
            _txt(screen, "Battle Log", 55, 162, f_sm, GRAY)
            for i, entry in enumerate(st.logs[-16:]):
                col = YELLOW if "defeated" in entry.lower() else WHITE
                _txt(screen, entry[:90], 55, 182 + i * 22, f_sm, col)

            # Move buttons
            r_atk  = _btn(screen,  50, 572, 200, 52, "1 — Attack",   f_sm, BLUE,
                          pygame.Rect( 50, 572, 200, 52).collidepoint(mx, my))
            r_spc  = _btn(screen, 270, 572, 200, 52, "2 — Special",  f_sm, ORANGE,
                          pygame.Rect(270, 572, 200, 52).collidepoint(mx, my))
            r_rgn  = _btn(screen, 490, 572, 200, 52, "3 — Regen",    f_sm, GREEN,
                          pygame.Rect(490, 572, 200, 52).collidepoint(mx, my))
            r_back = _btn(screen, 720, 572, 200, 52, "Menu (Esc)",   f_sm, GRAY,
                          pygame.Rect(720, 572, 200, 52).collidepoint(mx, my))

            def do_move(move):
                if move == 1 and p.mp < 10:
                    st.logs.append("Not enough MP for Attack — use Regen first.")
                    return
                if move == 2 and p.mp < 20:
                    st.logs.append("Not enough MP for Special — use Regen first.")
                    return
                eng.execute_player_move(move)
                st.logs.extend(eng.last_log_entries)
                if eng.game_over:
                    finalize_match()
                    st.screen = "end"

            for ev in events:
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_1:
                        do_move(1)
                    elif ev.key == pygame.K_2:
                        do_move(2)
                    elif ev.key == pygame.K_3:
                        do_move(3)
                    elif ev.key == pygame.K_ESCAPE:
                        st.screen = "menu"
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if r_atk.collidepoint(mx, my):
                        do_move(1)
                    elif r_spc.collidepoint(mx, my):
                        do_move(2)
                    elif r_rgn.collidepoint(mx, my):
                        do_move(3)
                    elif r_back.collidepoint(mx, my):
                        st.screen = "menu"

        # ==============================================================
        elif st.screen == "end":
        # ==============================================================
            eng = st.engine
            winner = eng.winner if eng else "?"
            col = GREEN if winner == st.selected_profile else RED
            _txt(screen, f"{winner} WINS!", W // 2, 95, f_xl, col, center=True)
            _txt(screen, "Match Summary", W // 2, 170, f_lg, WHITE, center=True)

            if st.player:
                _txt(screen, f"Damage dealt  : {st.player.total_damage_dealt}",
                     W // 2, 225, f_md, GREEN, center=True)
                _txt(screen, f"Damage taken  : {st.player.total_damage_taken}",
                     W // 2, 262, f_md, RED, center=True)
                _txt(screen, f"Rounds played : {eng.round_num}",
                     W // 2, 298, f_md, WHITE, center=True)
                if st.ai_fighter and not st.use_rl:
                    _txt(screen,
                         f"Final ensemble confidence: {st.ai_fighter.ensemble_confidence:.1f}%",
                         W // 2, 334, f_sm, GRAY, center=True)

            for i, p in enumerate(st.end_plots):
                _txt(screen, f"Saved: {os.path.basename(p)}", W // 2, 368 + i * 26,
                     f_sm, GRAY, center=True)

            r_plot   = _btn(screen, W // 2 - 310, 480, 280, 48, "Generate Plots", f_md, LBLUE,
                            pygame.Rect(W // 2 - 310, 480, 280, 48).collidepoint(mx, my))
            r_replay = _btn(screen, W // 2 +  30, 480, 280, 48, "Play Again",     f_md, GREEN,
                            pygame.Rect(W // 2 +  30, 480, 280, 48).collidepoint(mx, my))
            r_home   = _btn(screen, W // 2 - 130, 548, 260, 48, "Main Menu",      f_md, GRAY,
                            pygame.Rect(W // 2 - 130, 548, 260, 48).collidepoint(mx, my))

            for ev in events:
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if r_plot.collidepoint(mx, my) and st.tracker:
                        p1 = plot_damage_per_round(st.tracker.match_id)
                        p2 = plot_cumulative_damage(st.tracker.match_id)
                        st.end_plots = [p for p in (p1, p2) if p]

                    if r_replay.collidepoint(mx, my):
                        new_game()
                        st.screen = "game"

                    if r_home.collidepoint(mx, my):
                        st.screen = "menu"

        # ==============================================================
        elif st.screen == "stats":
        # ==============================================================
            prof = PlayerProfile.load_or_create(st.selected_profile)
            _txt(screen, f"Stats — {prof.name}", W // 2, 45, f_lg, YELLOW, center=True)

            rows = [
                ("Games played",       prof.games_played),
                ("Wins",               prof.wins),
                ("Losses",             prof.losses),
                ("Win rate",           f"{100 * prof.wins / max(1, prof.games_played):.1f}%"),
                ("Total damage dealt", prof.total_damage_dealt),
                ("Total damage taken", prof.total_damage_taken),
                ("Total moves",        prof.total_moves),
                ("Attack uses",        prof.move_usage_counts.get(1, 0)),
                ("Special uses",       prof.move_usage_counts.get(2, 0)),
                ("Regen uses",         prof.move_usage_counts.get(3, 0)),
                ("Last played",        prof.last_played or "—"),
            ]
            for i, (label, val) in enumerate(rows):
                _txt(screen, f"{label}:", 180, 105 + i * 46, f_md, GRAY)
                _txt(screen, str(val),    500, 105 + i * 46, f_md, WHITE)

            r_back = _btn(screen, W // 2 - 100, 628, 200, 46, "Back", f_md, LBLUE,
                          pygame.Rect(W // 2 - 100, 628, 200, 46).collidepoint(mx, my))
            for ev in events:
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if r_back.collidepoint(mx, my):
                        st.screen = "menu"
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    st.screen = "menu"

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
