   0: 4C A0 10 06                                get_sr           0, r4, sr32 (thread_position_in_grid.x)
   4: 9F 11 54 00 02 00 20 88 10 14              iadd             r0, 0, r4, lsl 2, wait 0
   e: 67 00 44 00 00 80 20 00 57 01 00 40 46 00  device_load      5, i32, xyzw, r0_r1_r2_r3, u0_u1, r0.discard, lsl 2, 0, 1, 0, 0
  1c: 9F 00 56 00 03 18 00 00 D0 26 08 00        imadd            r0.cache, u6, r0.discard, 0, wait 5
  28: 9F 00 54 00 03 1C 08 00 D0 2F 28 00        imadd            r0.cache, u7, r1.discard, r0.discard
  34: 9F 00 54 02 03 20 10 00 D0 2F 28 00        imadd            r1.cache, u8, r2.discard, r0.discard
  40: 9F 00 54 00 02 24 18 10 D0 2F 28 00        imadd            r0, u9, r3.discard, r1.discard
  4c: E7 00 54 00 02 04 21 00 11 00 00 90 11 00  device_store     0, i32, x, r0, u4_u5, r4.discard, lsl 2, 0, 1, 1
  5a: 0E 00 00 00                                stop             
