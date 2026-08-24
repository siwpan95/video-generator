#!/usr/bin/env python3
"""Generate a randomized glitch art video prompt with non-repeating variation.

Each run produces a different prompt by randomly combining:
- Color palette cycles
- Effect ordering and emphasis
- Glitch technique intensities
- Mood/style modifiers
- Narrative sequencing
"""

import random
import sys

# ── Color Palette Library ──────────────────────────────────────────
PALETTES = [
    {
        "name": "cyberpunk",
        "desc": "cyberpunk gradient palette cycling: electric cyan through neon magenta to deep indigo and radiant gold",
    },
    {
        "name": "vaporwave",
        "desc": "synthwave vaporwave palette: hot pink to cool cyan, purple neon, retro orange sunset gradients",
    },
    {
        "name": "bioluminescent",
        "desc": "deep ocean bioluminescent palette: teal green phosphorescence, electric blue, glowing violet, and coral pink",
    },
    {
        "name": "inferno",
        "desc": "volcanic inferno palette: molten orange, crimson red, black smoke, and white-hot yellow heat",
    },
    {
        "name": "aurora",
        "desc": "aurora borealis gradient: emerald green, arctic blue, soft violet, and pale pink shimmer",
    },
    {
        "name": "circuitry",
        "desc": "circuit board palette: matrix green, oxide orange, deep black, and electric white traces",
    },
    {
        "name": "holographic",
        "desc": "holographic iridescent palette: prismatic rainbow shifts, silver metallic sheen, pearlescent white",
    },
    {
        "name": "void",
        "desc": "deep space void palette: pitch black, dark matter purple, cold blue starlight, and crimson nebula glow",
    },
]

# ── Effect Libraries ───────────────────────────────────────────────
COSMIC_EFFECTS = [
    "star trails sweep as long-exposure light painting arcs with RGB channel separation — red channel offset left, blue channel offset right, creating ghosting trails",
    "a stylized Milky Way manifests as a luminous diagonal particle stream flowing across the frame, its pixels continuously sorting and unsorting",
    "hyperspace light speed warp stretches the color field horizontally — starlight smears into infinite luminous streaks with motion blur chromatic smear",
    "deep cosmic void with swirling nebula dust clouds bleeding neon colors at the edges through chromatic aberration fringing",
    "concentric star trail rings orbit a central glow point, each ring at a different rotation speed creating hypnotic spiral patterns",
]

GLITCH_EFFECTS = [
    "pixel sorting cascades in vertical columns: dark pixels rearrange by luminosity, creating melting streaks and datamosh trails that ooze downward like digital lava",
    "edge detection reveals Canny-style neon wireframe contours outlining every shape in glowing magenta and cyan — geometric mesh overlays tracing through the abstract chaos",
    "CRT scanline texture overlays everything with horizontal line gaps, occasionally interrupted by VHS screen tears ripping random pixel rows horizontally",
    "data corruption blocks flash randomly — pure HSL noise tiles and JPEG-artifact glitch blocks in saturated cyan and magenta wash across the frame",
    "heavy chromatic aberration splits the RGB channels at frame edges — red fringing bleeding left, cyan bleeding right, intensity increasing toward corners",
    "pixel columns stretch vertically in random segments, then snap back — elastic glitch distortion like a corrupted buffer read",
    "RGB channel separation intensifies at edges: subject outlines trace in red-blue ghosting double images that drift apart and recombine",
]

LENS_EFFECTS = [
    "cinematic anamorphic lens flares bloom from bright points: multi-layer blue and gold optical reflections with rainbow diffraction rings and dust speckles",
    "deep volumetric glow radiates from every luminous element — soft ethereal bloom bleeding beyond boundaries, god rays piercing through the digital haze",
    "a massive lens flare burst with chromatic halo rings, hexagonal aperture artifacts, and light streaks radiating in eight directions",
    "bokeh particles float in foreground: out-of-focus circular light orbs in various sizes, creating depth and dreamy atmosphere",
]

STYLE_MODIFIERS = [
    "non-photorealistic abstract visual loop, purely atmospheric and psychedelic, corrupted signal aesthetic",
    "VJ background / visual music loop style, hypnotic and trance-inducing, designed for projection mapping",
    "surreal digital dreamscape crossing between machine error and cosmic beauty, like a sentient AI's fever dream",
    "generative art aesthetic reminiscent of TouchDesigner real-time visuals — fluid, procedural, ever-evolving",
    "glitch poetry — mistakes elevated to art, data degradation as creative medium, beauty in the broken signal",
    "retro-futuristic arcade aesthetic meets deep space — 80s synth nostalgia filtered through corrupted digital memory",
    "minimalist abstract composition — less noise, more deliberate distortion, zen-like glitch meditation",
]

TRANSITIONS = [
    "smooth gradient dissolve between color phases, each bleeding into the next with soft chromatic edge blending",
    "hard glitch transition: frame tears diagonally as if the video file is corrupted, revealing the next palette underneath",
    "pixel sorting wave sweeps horizontally, rearranging pixels left to right as a transition wipe",
    "data moshing: the current palette dissolves into compression artifact blocks before the next emerges from the digital noise",
    "RGB channel displacement: red channel slides right while blue slides left, revealing the next color phase through the split",
]


def build_prompt(seed: int = None) -> str:
    """Build a randomized glitch art video prompt."""
    if seed is not None:
        random.seed(seed)

    palette = random.choice(PALETTES)
    style = random.choice(STYLE_MODIFIERS)

    # Select effects with varying counts for more randomness
    cosmic_count = random.randint(2, 3)
    glitch_count = random.randint(3, 5)
    lens_count = random.randint(1, 2)

    cosmic = random.sample(COSMIC_EFFECTS, cosmic_count)
    glitch = random.sample(GLITCH_EFFECTS, glitch_count)
    lens = random.sample(LENS_EFFECTS, lens_count)

    # Randomize transition types
    transition_count = random.randint(2, 3)
    transitions = random.sample(TRANSITIONS, transition_count)

    # Random mood words
    mood_options = [
        "hypnotic", "surreal", "ominous", "euphoric", "melancholic",
        "aggressive", "serene", "cosmic", "cerebral", "visceral",
    ]
    mood = random.sample(mood_options, 3)

    # Build the master prompt
    parts = []

    # Opening style declaration
    parts.append(
        f"Abstract digital glitch art visual loop, {palette['desc']}. "
        f"Overall style: {style}. "
        f"Mood: {', '.join(mood)}."
    )

    # Cosmic/space elements
    random.shuffle(cosmic)
    parts.append(" ".join(cosmic + [". "]))

    # Glitch technique elements  
    random.shuffle(glitch)
    parts.append(" ".join(glitch + [". "]))

    # Lens flare and glow
    random.shuffle(lens)
    parts.append(" ".join(lens + [". "]))

    # Transitions between phases
    parts.append(
        f"Transitions between color phases use {random.choice(['a mix of'])}: "
        f"{'; '.join(transitions)}."
    )

    # Final composition note
    endings = [
        "The composition breathes like a living corrupted signal — never static, always in flux.",
        "Every element exists in superposition between order and chaos, data and noise.",
        "The result is a seamless flowing visual tapestry where digital decay meets cosmic radiance.",
        "Pure visual atmosphere — no narrative, no characters, just mesmerizing abstract motion.",
        "A single continuous take through a psychedelic digital dimension, hypnotic and immersive.",
    ]
    parts.append(random.choice(endings))

    return " ".join(parts)


if __name__ == "__main__":
    # Support optional seed for reproducibility
    seed = None
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            pass

    prompt = build_prompt(seed)
    print(prompt)
