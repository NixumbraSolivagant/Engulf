import pygame

CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
    "AR PL UMing CN",
]


def load_font(size: int) -> pygame.font.Font:
    for name in CJK_FONT_CANDIDATES:
        try:
            font = pygame.font.SysFont(name, size)
            # Render a quick sample to ensure glyphs exist; if it returns width 0, skip
            if font is not None:
                test = font.render("中文ABC", True, (255, 255, 255))
                if test.get_width() > 0:
                    return font
        except Exception:
            pass
    # Fallback to default font
    return pygame.font.SysFont(None, size)
