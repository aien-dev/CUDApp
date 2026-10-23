   0: 1C A0 10 06                                get_sr           0, r1, sr32 (thread_position_in_grid.x)
   4: 9F 11 54 04 03 00 08 88 10 14              iadd             r2.cache, 0, r1, lsl 2, wait 0
   e: 9F 01 54 08 02 02 10 8C 11 04              iadd             r4, 1, r2.cache
  18: 9F 01 54 06 02 04 10 88 11 04              iadd             r3, 2, r2
  22: 67 00 54 00 00 02 20 00 51 01 00 40 46 00  device_load      5, i32, x, r0, u0_u1, r2, lsl 2, 0, 1, 1, 0
  30: 67 00 54 16 01 00 00 00 51 01 00 40 02 00  device_load      5, i32, x, r11, u2_u3, 0, 0, 1, 1, 0
  3e: 9F 01 54 04 02 06 10 88 15 04              iadd             r2, 3, r2.discard
  48: 67 00 54 10 00 84 20 00 11 00 00 40 46 00  device_load      0, i32, x, r8, u0_u1, r4.discard, lsl 2, 0, 1, 1, 0
  56: 67 00 54 14 01 00 00 00 11 80 00 40 02 00  device_load      0, i32, x, r10, u2_u3, 0, 4, 1, 1, 0
  64: 67 00 54 0A 00 83 20 00 51 00 00 40 46 00  device_load      1, i32, x, r5, u0_u1, r3.discard, lsl 2, 0, 1, 1, 0
  72: 67 00 54 0E 01 00 00 00 51 00 01 40 02 00  device_load      1, i32, x, r7, u2_u3, 0, 8, 1, 1, 0
  80: 67 00 54 04 00 82 20 00 91 00 00 40 46 00  device_load      2, i32, x, r2, u0_u1, r2.discard, lsl 2, 0, 1, 1, 0
  8e: 67 00 44 08 01 00 00 00 91 80 01 40 02 00  device_load      2, i32, x, r4, u2_u3, 0, 12, 1, 0, 0
  9c: 9F 00 56 00 03 2C 00 00 D0 26 0A 00        imadd            r0.cache, r11.discard, r0.discard, 0, wait 5
  a8: 9F 10 54 00 03 28 40 00 D0 2F 2A 00        imadd            r0.cache, r10.discard, r8.discard, r0.discard, wait 0
  b4: 9F 20 54 00 03 1C 28 00 D0 2F 2A 00        imadd            r0.cache, r7.discard, r5.discard, r0.discard, wait 1
  c0: 9F 40 54 00 02 10 10 00 D0 2F 2A 00        imadd            r0, r4.discard, r2.discard, r0.discard, wait 2
  cc: E7 00 54 00 02 01 21 00 11 00 00 90 11 00  device_store     0, i32, x, r0, u4_u5, r1.discard, lsl 2, 0, 1, 1
  da: 0E 00 00 00                                stop             
