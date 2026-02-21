import pygame, sys
import numpy as np

pygame.init()

W, H = 900, 900
screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
pygame.display.set_caption("Animated Warped Matrix (no bubble)")

clock = pygame.time.Clock()

# ---------- helpers ----------
def box_blur_same(a, r=2):
    """
    Fast box blur via integral image, returns SAME shape as input.
    """
    if r <= 0:
        return a.astype(np.float32)

    a = a.astype(np.float32)
    h, w = a.shape
    k = 2 * r + 1

    # pad by r on each side
    p = np.pad(a, ((r, r), (r, r)), mode="edge")  # (h+2r, w+2r)

    # integral image with an extra zero border to make indexing clean
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1), dtype=np.float32)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)

    # window sums -> shape (h, w)
    out = (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) / (k * k)
    return out

# ---------- animated field ----------
def make_frame(w, h, t):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = xx - w/2
    cy = yy - h/2

    # big flowing warp field (time-driven)
    wx = 65*np.sin(0.0038*cy + 2.4*t) + 30*np.sin(0.0065*cx - 1.7*t) + 18*np.sin(0.0050*(cx+cy) + 1.3*t)
    wy = 65*np.cos(0.0036*cx - 2.1*t) + 30*np.cos(0.0061*cy + 1.9*t) + 18*np.cos(0.0046*(cx-cy) - 1.1*t)

    X = cx + wx
    Y = cy + wy

    # swirl / bend
    r = np.sqrt(X*X + Y*Y) + 1e-6
    th = np.arctan2(Y, X)
    X += 32*np.sin(0.018*r + 2.0*np.sin(3.0*th + 1.2*t))
    Y += 32*np.cos(0.018*r + 2.0*np.cos(2.6*th - 1.0*t))

    # flow direction from the warp (smoothed)
    fx = box_blur_same(wx, r=6)
    fy = box_blur_same(wy, r=6)
    mag = np.sqrt(fx*fx + fy*fy) + 1e-6
    fx /= mag
    fy /= mag

    # advect coordinates along flow (pattern slides along wave direction)
    adv = 55.0
    X2 = X + adv * fx
    Y2 = Y + adv * fy

    # high-frequency interference “lines”
    a1 = 0.42*X2 + 0.12*Y2
    a2 = -0.18*X2 + 0.46*Y2
    a3 = 0.30*r

    stripes = (
        0.55*np.sin(2.6*a1 + 2.8*t) +
        0.50*np.sin(2.4*a2 - 2.4*t) +
        0.35*np.sin(2.0*a3 + 1.6*t)
    )
    stripes = np.tanh(3.2*stripes)
    base = (stripes + 1.0) * 0.5  # 0..1

    # emboss lighting
    gy, gx = np.gradient(base)
    light = 0.55 + 3.2 * (-0.75*gx + -0.95*gy)
    light = np.clip(light, 0.0, 1.35)
    base = np.clip(base * light, 0, 1)

    # rgb split (animated slightly)
    shift = int(4 + 2*np.sin(1.2*t))
    rch = np.roll(base, shift, axis=1)
    gch = base
    bch = np.roll(base, -shift, axis=0)

    img = np.stack([rch, gch, bch], axis=-1)
    img = np.clip(img**1.35, 0, 1)

    return (img * 255).astype(np.uint8)

# ---------- main ----------
running = True
t = 0.0
while running:
    dt = clock.tick(60) / 1000.0
    t += dt * 1.6  # speed multiplier

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        if event.type == pygame.VIDEORESIZE:
            W, H = event.w, event.h
            screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)

    frame = make_frame(W, H, t)
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surf, (0, 0))
    pygame.display.flip()

pygame.quit()
sys.exit()