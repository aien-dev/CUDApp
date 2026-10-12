/* omega_gpu_fmt.c -- see omega_gpu_fmt.h. Freestanding: stdarg only. */
#include "omega_gpu_fmt.h"
#include <stdarg.h>
#include <stdint.h>

typedef struct { char *out; size_t cap, n; } Sink;

static void put(Sink *s, char c) {
    if (s->n + 1 < s->cap) s->out[s->n++] = c;
}
static void put_str(Sink *s, const char *p) {
    if (!p) p = "(null)";
    while (*p) put(s, *p++);
}
static void put_u(Sink *s, unsigned long long v, unsigned base) {
    char tmp[24];
    int i = 0;
    do { unsigned d = (unsigned)(v % base); tmp[i++] = (char)(d < 10 ? '0' + d : 'a' + d - 10); v /= base; } while (v);
    while (i) put(s, tmp[--i]);
}
static void put_d(Sink *s, long long v) {
    if (v < 0) { put(s, '-'); put_u(s, (unsigned long long)(-(v + 1)) + 1ull, 10); }
    else put_u(s, (unsigned long long)v, 10);
}

int omega_gpu_fmt(char *out, size_t cap, const char *fmt, ...) {
    Sink s = { out, cap, 0 };
    va_list ap;
    if (!out || cap == 0) return 0;
    if (!fmt) fmt = "";
    va_start(ap, fmt);
    for (const char *p = fmt; *p; p++) {
        if (*p != '%') { put(&s, *p); continue; }
        const char *start = p++;
        int l = 0, z = 0;
        if (*p == 'z') { z = 1; p++; }
        else while (*p == 'l' && l < 2) { l++; p++; }
        switch (*p) {
        case 's': put_str(&s, va_arg(ap, const char *)); break;
        case 'c': put(&s, (char)va_arg(ap, int)); break;
        case 'd':
            if (z) put_d(&s, (long long)va_arg(ap, ptrdiff_t));
            else if (l == 2) put_d(&s, va_arg(ap, long long));
            else if (l == 1) put_d(&s, va_arg(ap, long));
            else put_d(&s, va_arg(ap, int));
            break;
        case 'u': case 'x': {
            unsigned base = *p == 'x' ? 16u : 10u;
            if (z) put_u(&s, va_arg(ap, size_t), base);
            else if (l == 2) put_u(&s, va_arg(ap, unsigned long long), base);
            else if (l == 1) put_u(&s, va_arg(ap, unsigned long), base);
            else put_u(&s, va_arg(ap, unsigned int), base);
            break;
        }
        case '%': put(&s, '%'); break;
        default: /* unsupported conversion: copy it literally, consume nothing */
            for (const char *q = start; q <= p && *q; q++) put(&s, *q);
            if (!*p) p--;
            break;
        }
    }
    va_end(ap);
    s.out[s.n] = '\0';
    return (int)s.n;
}
