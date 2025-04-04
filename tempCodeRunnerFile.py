def draw_mp_bar(x, y, mp):
    pygame.draw.rect(screen, GRAY, (x, y, 200, 10))
    pygame.draw.rect(screen, BLUE, (x, y, 2 * mp, 10))
    display_text(f"{mp}/50", x + 70, y, font, WHITE)