"""Kairos autotrader: a thin, bounded runner that trades EVEDEX perpetuals by itself.

Algorithms propose candidates with fixed stop/take levels, an optional LLM reviews them
together with typed news evidence, and the engine sizes, enters, protects and exits one
position at a time inside hard limits. There is no withdraw or transfer code.
"""

__version__ = "0.4.0"
