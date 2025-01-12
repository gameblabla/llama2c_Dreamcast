llama2.c port for Sega Dreamcast
=================================

Here's a port of llama2.c port for Sega Dreamcast using TinyStories (262K parameters) models.
Currently it makes use of FMAC/FSSRA along with other SH-4 specific optimizations, 
altho the benefits are somewhat small but noticeable :p.

If you want to replace the model with yours, you can go ahead and replace the model.bin/tok.bin files with yours in cd folder.
Be warned that the Dreamcast only has 16MB and that larger models have a significant performance penalty (Tinystories 15M model would run at 1 tokens/s, Tinystories 262K is like 30x faster).