#include "mem_dc.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "sh4_math.h"

void * memcpy6 (void *dest, const void *src, size_t len)
{
  /*if(!len)
  {
    return dest;
  }*/

  const uint8_t *s = (uint8_t *)src;
  uint8_t *d = (uint8_t *)dest;

  uint32_t diff = (uint32_t)d - (uint32_t)(s + 1); // extra offset because input gets incremented before output is calculated
  // Underflow would be like adding a negative offset

  // Can use 'd' as a scratch reg now
  asm volatile (
    "clrs\n" // Align for parallelism (CO) - SH4a use "stc SR, Rn" instead with a dummy Rn
  ".align 2\n"
  "0:\n\t"
    "dt %[size]\n\t" // (--len) ? 0 -> T : 1 -> T (EX 1)
    "mov.b @%[in]+, %[scratch]\n\t" // scratch = *(s++) (LS 1/2)
    "bf.s 0b\n\t" // while(s != nexts) aka while(!T) (BR 1/2)
    " mov.b %[scratch], @(%[offset], %[in])\n" // *(datatype_of_s*) ((char*)s + diff) = scratch, where src + diff = dest (LS 1)
    : [in] "+&r" ((uint32_t)s), [scratch] "=&r" ((uint32_t)d), [size] "+&r" (len) // outputs
    : [offset] "z" (diff) // inputs
    : "t", "memory" // clobbers
  );

  return dest;
}


void * memsetasm (void *dest, const uint8_t val, size_t len)
{
  /*if(!len)
  {
    return dest;
  }*/

  uint8_t * d = (uint8_t*)dest;
  uint8_t * nextd = d + len;

  asm volatile (
    "clrs\n\t" // Align for parallelism (CO) - SH4a use "stc SR, Rn" instead with a dummy Rn
    "dt %[size]\n" // Decrement and test size here once to prevent extra jump (EX 1)
  ".align 2\n"
  "1:\n\t"
    // *--nextd = val
    "mov.b %[in], @-%[out]\n\t" // (LS 1/1)
    "bf.s 1b\n\t" // (BR 1/2)
    " dt %[size]\n" // (--len) ? 0 -> T : 1 -> T (EX 1)
    : [out] "+r" ((uint32_t)nextd), [size] "+&r" (len) // outputs
    : [in] "r" (val) // inputs
    : "t", "memory" // clobbers
  );

  return dest;
}


/* Only if not linked against math lib */
/*float floorf(float x)
{
	return MATH_Fast_Floorf(x);
}
*/


// Ian micheal optimized SQ function
void bit64_sq_cpy(void *dest, const void *src, size_t n)
{
    uint32 *d, *s;
    uint32 r0, r1, r2, r3, r4, r5, r6, r7;
    
    // Set up destination pointer with SQ address space and alignment
    d = (uint32 *)(0xe0000000 | (((uint32)dest) & 0x03ffffe0));
    // Set up source pointer
    s = (uint32 *)(src);

    // Configure memory-mapped registers for SQ access
    *((volatile unsigned int*)0xFF000038) = ((((uint32)dest) >> 26) << 2) & 0x1c;
    *((volatile unsigned int*)0xFF00003C) = ((((uint32)dest) >> 26) << 2) & 0x1c;

    // Convert n to number of 64-byte blocks
    n >>= 6;

    // Main copy loop
    while (n--) 
    {
        // Expose loads to the compiler
        r0 = *s++; r1 = *s++;
        r2 = *s++; r3 = *s++;
        r4 = *s++; r5 = *s++;
        r6 = *s++; r7 = *s++;

        __asm__ volatile (
            // Store Queue 0 (sq0) operations
            "mov.l %2,@%0  ; mov.l %3,@(4,%0) \n\t"
            "mov.l %4,@(8,%0) ; mov.l %5,@(12,%0) \n\t"
            "mov.l %6,@(16,%0) ; mov.l %7,@(20,%0) \n\t"
            "mov.l %8,@(24,%0) ; mov.l %9,@(28,%0) \n\t"
            "pref @%0 \n\t"
            "ocbi @%1 \n\t"
            "add #32,%0 \n\t"

            // Load next batch
            : "+r" (d)
            : "r" (dest), "r" (r0), "r" (r1), "r" (r2), "r" (r3), 
              "r" (r4), "r" (r5), "r" (r6), "r" (r7)
            : "memory"
        );

        // Expose loads to the compiler for the next batch
        r0 = *s++; r1 = *s++;
        r2 = *s++; r3 = *s++;
        r4 = *s++; r5 = *s++;
        r6 = *s++; r7 = *s++;

        __asm__ volatile (
            // Store Queue 1 (sq1) operations
            "mov.l %2,@%0  ; mov.l %3,@(4,%0) \n\t"
            "mov.l %4,@(8,%0) ; mov.l %5,@(12,%0) \n\t"
            "mov.l %6,@(16,%0) ; mov.l %7,@(20,%0) \n\t"
            "mov.l %8,@(24,%0) ; mov.l %9,@(28,%0) \n\t"
            "pref @%0 \n\t"
            "ocbi @%1 \n\t"
            "add #32,%0 \n\t"

            : "+r" (d)
            : "r" (dest), "r" (r0), "r" (r1), "r" (r2), "r" (r3), 
              "r" (r4), "r" (r5), "r" (r6), "r" (r7)
            : "memory"
        );
    }

    // Clear SQ registers after operation
    *((uint32 *)(0xe0000000)) = 0;
    *((uint32 *)(0xe0000020)) = 0;
}