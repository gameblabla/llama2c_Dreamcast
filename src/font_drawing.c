#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <SDL/SDL.h>
#include "font_drawing.h"
#include "font_menudata.h"

extern SDL_Surface* gScreen;
#define setPixel(buffer, x,y,c) *((uint16_t* restrict)buffer + ((x) + (y) * (gScreen->pitch>>1))) = c;

/**
 * Returns whether a pixel at (x,y) in the 8x8 character is an "outline" pixel.
 * (Unchanged)
 */
static int32_t isOutlinePixel(uint8_t* charfont, int32_t x, int32_t y)
{
    int32_t xis0 = !x, xis7 = x == 7, yis0 = !y, yis7 = y == 7;
    
    if(xis0)
    {
        if(yis0)
        {
            return !(*charfont & 0x80) && (
                (*charfont & 0x40) ||
                (charfont[1] & 0x80) || (charfont[1] & 0x40));
        }
        else if(yis7)
        {
            return !(charfont[7] & 0x80) && (
                (charfont[7] & 0x40) ||
                (charfont[6] & 0x80) || (charfont[6] & 0x40));
        }
        else
        {
            return !(charfont[y] & 0x80) && (
                (charfont[y - 1] & 0x80) || (charfont[y - 1] & 0x40) ||
                (charfont[y] & 0x40) ||
                (charfont[y + 1] & 0x80) || (charfont[y + 1] & 0x40));
        }
    }
    else if(xis7)
    {
        if(yis0)
        {
            return !(*charfont & 0x01) && (
                (*charfont & 0x02) ||
                (charfont[1] & 0x01) || (charfont[1] & 0x02));
        }
        else if(yis7)
        {
            return !(charfont[7] & 0x01) && (
                (charfont[7] & 0x02) ||
                (charfont[6] & 0x01) || (charfont[6] & 0x02));
        }
        else
        {
            return !(charfont[y] & 0x01) && (
                (charfont[y - 1] & 0x01) || (charfont[y - 1] & 0x02) ||
                (charfont[y] & 0x02) ||
                (charfont[y + 1] & 0x01) || (charfont[y + 1] & 0x02));
        }
    }
    else
    {
        int32_t b = 1 << (7 - x);
        if(yis0)
        {
            return !(*charfont & b) && (
                (*charfont & (b << 1)) || (*charfont & (b >> 1)) ||
                (charfont[1] & (b << 1)) || (charfont[1] & b) || (charfont[1] & (b >> 1)));
        }
        else if(yis7)
        {
            return !(charfont[7] & b) && (
                (charfont[7] & (b << 1)) || (charfont[7] & (b >> 1)) ||
                (charfont[6] & (b << 1)) || (charfont[6] & b) || (charfont[6] & (b >> 1)));
        }
        else
        {
            return !(charfont[y] & b) && (
                (charfont[y] & (b << 1)) || (charfont[y] & (b >> 1)) ||
                (charfont[y - 1] & (b << 1)) || (charfont[y - 1] & b) || (charfont[y - 1] & (b >> 1)) ||
                (charfont[y + 1] & (b << 1)) || (charfont[y + 1] & b) || (charfont[y + 1] & (b >> 1)));
        }
    }
}

/**
 * Draws a single character. Modified to make the output 8×16 (twice as tall).
 */
static void drawChar(uint16_t* restrict buffer, int32_t *x, int32_t *y,
                     int32_t margin, char ch, uint32_t fc, uint32_t olc)
{
    int32_t i, j;
    uint8_t *charSprite;

    // If newline, reset x and move y down by 16 (instead of 8).
    if (ch == '\n')
    {
        *x = margin;
        *y += 16; // Twice the normal line height
    }
    else if(*y < gScreen->h - 1)
    {
        // Get pointer to the 8×8 font data for character ch
        charSprite = ch * 8 + n2DLib_font;

        // Draw charSprite as 8×16:
        // For each row in [0..7], draw it twice (i*2) and (i*2+1).
        for(i = 0; i < 8; i++)
        {
            for(j = 7; j >= 0; j--)
            {
                // Is this bit set in the font?
                if ((charSprite[i] >> j) & 1)
                {
                    setPixel(buffer, *x + (7 - j), *y + (i * 2),     fc);
                    setPixel(buffer, *x + (7 - j), *y + (i * 2) + 1, fc);
                }
                else if (isOutlinePixel(charSprite, 7 - j, i))
                {
                    setPixel(buffer, *x + (7 - j), *y + (i * 2),     olc);
                    setPixel(buffer, *x + (7 - j), *y + (i * 2) + 1, olc);
                }
            }
        }

        // Advance to the next character by 8 pixels horizontally
        *x += 8;
    }
}

/**
 * Draws a string of characters. (Unchanged except that we now expect
 * the new line to jump 16 pixels down, which is handled inside drawChar.)
 */
static void drawString(uint16_t* restrict buffer, int32_t *x, int32_t *y,
                       int32_t _x, const char *str, uint32_t fc, uint32_t olc)
{
    unsigned long i, max = strlen(str) + 1;
    for(i = 0; i < max; i++)
        drawChar(buffer, x, y, _x, str[i], fc, olc);
}

/**
 * Public function to print a string at (x,y).
 * No changes needed here except that it uses the new drawChar that doubles the height.
 */
void print_string(const char *s, const uint32_t fg_color, const uint32_t bg_color,
                  int32_t x, int32_t y, uint16_t* restrict buffer) 
{
    drawString(buffer, &x, &y, 0, s, fg_color, bg_color);
}
