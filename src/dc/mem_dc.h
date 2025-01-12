#ifndef MEM_DC
#define MEM_DC

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
extern void * memcpy6 (void *dest, const void *src, size_t len);
extern void * memsetasm (void *dest, const uint8_t val, size_t len);
extern void bit64_sq_cpy(void *dest, const void *src, size_t n);
#ifdef __cplusplus
}
#endif

#endif
