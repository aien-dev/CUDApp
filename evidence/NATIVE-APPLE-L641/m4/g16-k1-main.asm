   0: 1C A0 10 06                                get_sr           0, r1, sr32 (thread_position_in_grid.x)
   4: 67 10 44 00 00 01 20 00 51 01 00 40 46 00  device_load      5, i32, x, r0, u0_u1, r1, lsl 2, 0, wait 0, 1, 0, 0
  12: 9F 00 56 00 02 18 00 00 D0 26 08 00        imadd            r0, u6, r0.discard, 0, wait 5
  1e: E7 00 54 00 02 01 21 00 11 00 00 90 11 00  device_store     0, i32, x, r0, u4_u5, r1.discard, lsl 2, 0, 1, 1
  2c: 0E 00 00 00                                stop             
