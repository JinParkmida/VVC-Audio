"""VVC — Virtual Voice Complaint: speech encoded as MIDI synthesis instructions."""
from .codec import Config, Params, analyze, synthesize, params_to_midi, midi_to_params, render_piano

__all__ = ["Config", "Params", "analyze", "synthesize", "params_to_midi", "midi_to_params", "render_piano"]
