"""``logic`` -- analysis modules for the song-vs-music ECoG project.

Every computation, I/O, plotting, and orchestration module lives inside
this package. Notebooks and scripts at the project root import from
``logic`` either by using fully qualified names (``from logic.decoding
import ...``) or, for backwards compatibility with the flat layout, by
prepending this directory to ``sys.path`` so flat imports such as
``from decoding import ...`` still resolve.
"""
