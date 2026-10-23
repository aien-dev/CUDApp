   0: 0C A0 11 06 00 00 00 10                    get_sr           0, r64, sr32 (thread_position_in_grid.x)
   8: 9F 11 54 00 03 00 00 8A 10 14              iadd             r0.cache, 0, r64, lsl 2, wait 0
  12: 9F 01 54 00 02 00 00 88 14 04              iadd             r0, 0, r0.discard, lsl 4
  1c: 67 00 54 78 00 00 20 00 17 00 00 40 46 00  device_load      0, i32, xyzw, r60_r61_r62_r63, u0_u1, r0, lsl 2, 0, 1, 1, 0
  2a: 9F 01 54 02 02 08 00 8C 11 04              iadd             r1, 4, r0.cache
  34: 67 00 54 70 00 81 20 00 17 00 00 40 46 00  device_load      0, i32, xyzw, r56_r57_r58_r59, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
  42: 9F 01 54 02 02 10 00 8C 11 04              iadd             r1, 8, r0.cache
  4c: 67 00 54 68 00 81 20 00 17 00 00 40 46 00  device_load      0, i32, xyzw, r52_r53_r54_r55, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
  5a: 9F 01 54 02 02 18 00 8C 11 04              iadd             r1, 12, r0.cache
  64: 67 00 54 60 00 81 20 00 17 00 00 40 46 00  device_load      0, i32, xyzw, r48_r49_r50_r51, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
  72: 9F 01 54 02 02 20 00 8C 11 04              iadd             r1, 16, r0.cache
  7c: 67 00 54 58 00 81 20 00 17 00 00 40 46 00  device_load      0, i32, xyzw, r44_r45_r46_r47, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
  8a: 9F 01 54 02 02 28 00 8C 11 04              iadd             r1, 20, r0.cache
  94: 67 00 54 50 00 81 20 00 17 00 00 40 46 00  device_load      0, i32, xyzw, r40_r41_r42_r43, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
  a2: 9F 01 54 02 02 30 00 8C 11 04              iadd             r1, 24, r0.cache
  ac: 67 00 54 48 00 81 20 00 57 01 00 40 46 00  device_load      5, i32, xyzw, r36_r37_r38_r39, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
  ba: 9F 01 54 02 02 38 00 8C 11 04              iadd             r1, 28, r0.cache
  c4: 67 00 54 40 00 81 20 00 57 01 00 40 46 00  device_load      5, i32, xyzw, r32_r33_r34_r35, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
  d2: 9F 01 54 02 02 40 00 8C 11 04              iadd             r1, 32, r0.cache
  dc: 67 00 54 38 00 81 20 00 17 01 00 40 46 00  device_load      4, i32, xyzw, r28_r29_r30_r31, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
  ea: 9F 01 54 02 02 48 00 8C 11 04              iadd             r1, 36, r0.cache
  f4: 67 00 54 30 00 81 20 00 17 01 00 40 46 00  device_load      4, i32, xyzw, r24_r25_r26_r27, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
 102: 9F 01 54 02 02 50 00 8C 11 04              iadd             r1, 40, r0.cache
 10c: 67 00 54 28 00 81 20 00 D7 00 00 40 46 00  device_load      3, i32, xyzw, r20_r21_r22_r23, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
 11a: 9F 01 54 02 02 58 00 8C 11 04              iadd             r1, 44, r0.cache
 124: 67 00 54 20 00 81 20 00 D7 00 00 40 46 00  device_load      3, i32, xyzw, r16_r17_r18_r19, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
 132: 9F 01 54 02 02 60 00 8C 11 04              iadd             r1, 48, r0.cache
 13c: 67 00 54 18 00 81 20 00 97 00 00 40 46 00  device_load      2, i32, xyzw, r12_r13_r14_r15, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
 14a: 9F 01 54 02 02 68 00 8C 11 04              iadd             r1, 52, r0.cache
 154: 67 00 54 10 00 81 20 00 97 00 00 40 46 00  device_load      2, i32, xyzw, r8_r9_r10_r11, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
 162: 9F 01 54 02 02 70 00 8C 11 04              iadd             r1, 56, r0.cache
 16c: 67 00 54 08 00 81 20 00 97 00 00 40 46 00  device_load      2, i32, xyzw, r4_r5_r6_r7, u0_u1, r1.discard, lsl 2, 0, 1, 1, 0
 17a: 9F 01 54 00 02 78 00 88 15 04              iadd             r0, 60, r0.discard
 184: 67 00 44 00 00 80 20 00 57 00 00 40 46 00  device_load      1, i32, xyzw, r0_r1_r2_r3, u0_u1, r0.discard, lsl 2, 0, 1, 0, 0
 192: 9F 10 54 78 03 18 E0 01 D0 26 08 00        imadd            r60.cache, u6, r60.discard, 0, wait 0
 19e: 9F 00 54 78 03 1C E8 C1 D3 2F 28 00        imadd            r60.cache, u7, r61.discard, r60.discard
 1aa: 9F 00 54 78 03 20 F0 C1 D3 2F 28 00        imadd            r60.cache, u8, r62.discard, r60.discard
 1b6: 9F 00 54 78 03 24 F8 C1 D3 2F 28 00        imadd            r60.cache, u9, r63.discard, r60.discard
 1c2: 9F 00 54 70 03 28 C0 C1 D3 2F 28 00        imadd            r56.cache, u10, r56.discard, r60.discard
 1ce: 9F 00 54 70 03 2C C8 81 D3 2F 28 00        imadd            r56.cache, u11, r57.discard, r56.discard
 1da: 9F 00 54 70 03 30 D0 81 D3 2F 28 00        imadd            r56.cache, u12, r58.discard, r56.discard
 1e6: 9F 00 54 70 03 34 D8 81 D3 2F 28 00        imadd            r56.cache, u13, r59.discard, r56.discard
 1f2: 9F 00 54 68 03 38 A0 81 D3 2F 28 00        imadd            r52.cache, u14, r52.discard, r56.discard
 1fe: 9F 00 54 68 03 3C A8 41 D3 2F 28 00        imadd            r52.cache, u15, r53.discard, r52.discard
 20a: 9F 00 54 68 03 40 B0 41 D3 2F 28 00        imadd            r52.cache, u16, r54.discard, r52.discard
 216: 9F 00 54 68 03 44 B8 41 D3 2F 28 00        imadd            r52.cache, u17, r55.discard, r52.discard
 222: 9F 00 54 60 03 48 80 41 D3 2F 28 00        imadd            r48.cache, u18, r48.discard, r52.discard
 22e: 9F 00 54 60 03 4C 88 01 D3 2F 28 00        imadd            r48.cache, u19, r49.discard, r48.discard
 23a: 9F 00 54 60 03 50 90 01 D3 2F 28 00        imadd            r48.cache, u20, r50.discard, r48.discard
 246: 9F 00 54 60 03 54 98 01 D3 2F 28 00        imadd            r48.cache, u21, r51.discard, r48.discard
 252: 9F 00 54 58 03 58 60 01 D3 2F 28 00        imadd            r44.cache, u22, r44.discard, r48.discard
 25e: 9F 00 54 58 03 5C 68 C1 D2 2F 28 00        imadd            r44.cache, u23, r45.discard, r44.discard
 26a: 9F 00 54 5A 03 60 70 C1 D2 2F 28 00        imadd            r45.cache, u24, r46.discard, r44.discard
 276: 9F 00 54 58 03 64 78 D1 D2 2F 28 00        imadd            r44.cache, u25, r47.discard, r45.discard
 282: 9F 00 54 50 03 68 40 C1 D2 2F 28 00        imadd            r40.cache, u26, r40.discard, r44.discard
 28e: 9F 00 54 50 03 6C 48 81 D2 2F 28 00        imadd            r40.cache, u27, r41.discard, r40.discard
 29a: 9F 00 54 52 03 70 50 81 D2 2F 28 00        imadd            r41.cache, u28, r42.discard, r40.discard
 2a6: 9F 00 54 50 03 74 58 91 D2 2F 28 00        imadd            r40.cache, u29, r43.discard, r41.discard
 2b2: 9F 00 56 48 03 78 20 81 D2 2F 28 00        imadd            r36.cache, u30, r36.discard, r40.discard, wait 5
 2be: 9F 00 54 48 03 7C 28 41 D2 2F 28 00        imadd            r36.cache, u31, r37.discard, r36.discard
 2ca: 9F 00 54 4A 03 80 30 41 D2 2F 28 00        imadd            r37.cache, u32, r38.discard, r36.discard
 2d6: 9F 00 54 48 03 84 38 51 D2 2F 28 00        imadd            r36.cache, u33, r39.discard, r37.discard
 2e2: 9F 00 54 40 03 88 00 41 D2 2F 28 00        imadd            r32.cache, u34, r32.discard, r36.discard
 2ee: 9F 00 54 40 03 8C 08 01 D2 2F 28 00        imadd            r32.cache, u35, r33.discard, r32.discard
 2fa: 9F 00 54 42 03 90 10 01 D2 2F 28 00        imadd            r33.cache, u36, r34.discard, r32.discard
 306: 9F 00 54 40 03 94 18 11 D2 2F 28 00        imadd            r32.cache, u37, r35.discard, r33.discard
 312: 9F 00 55 38 03 98 E0 00 D2 2F 28 00        imadd            r28.cache, u38, r28.discard, r32.discard, wait 4
 31e: 9F 00 54 38 03 9C E8 C0 D1 2F 28 00        imadd            r28.cache, u39, r29.discard, r28.discard
 32a: 9F 00 54 3A 03 A0 F0 C0 D1 2F 28 00        imadd            r29.cache, u40, r30.discard, r28.discard
 336: 9F 00 54 38 03 A4 F8 D0 D1 2F 28 00        imadd            r28.cache, u41, r31.discard, r29.discard
 342: 9F 00 54 30 03 A8 C0 C0 D1 2F 28 00        imadd            r24.cache, u42, r24.discard, r28.discard
 34e: 9F 00 54 30 03 AC C8 80 D1 2F 28 00        imadd            r24.cache, u43, r25.discard, r24.discard
 35a: 9F 00 54 32 03 B0 D0 80 D1 2F 28 00        imadd            r25.cache, u44, r26.discard, r24.discard
 366: 9F 00 54 30 03 B4 D8 90 D1 2F 28 00        imadd            r24.cache, u45, r27.discard, r25.discard
 372: 9F 80 54 28 03 B8 A0 80 D1 2F 28 00        imadd            r20.cache, u46, r20.discard, r24.discard, wait 3
 37e: 9F 00 54 28 03 BC A8 40 D1 2F 28 00        imadd            r20.cache, u47, r21.discard, r20.discard
 38a: 9F 00 54 2A 03 C0 B0 40 D1 2F 28 00        imadd            r21.cache, u48, r22.discard, r20.discard
 396: 9F 00 54 28 03 C4 B8 50 D1 2F 28 00        imadd            r20.cache, u49, r23.discard, r21.discard
 3a2: 9F 00 54 20 03 C8 80 40 D1 2F 28 00        imadd            r16.cache, u50, r16.discard, r20.discard
 3ae: 9F 00 54 20 03 CC 88 00 D1 2F 28 00        imadd            r16.cache, u51, r17.discard, r16.discard
 3ba: 9F 00 54 22 03 D0 90 00 D1 2F 28 00        imadd            r17.cache, u52, r18.discard, r16.discard
 3c6: 9F 00 54 20 03 D4 98 10 D1 2F 28 00        imadd            r16.cache, u53, r19.discard, r17.discard
 3d2: 9F 40 54 18 03 D8 60 00 D1 2F 28 00        imadd            r12.cache, u54, r12.discard, r16.discard, wait 2
 3de: 9F 00 54 18 03 DC 68 C0 D0 2F 28 00        imadd            r12.cache, u55, r13.discard, r12.discard
 3ea: 9F 00 54 1A 03 E0 70 C0 D0 2F 28 00        imadd            r13.cache, u56, r14.discard, r12.discard
 3f6: 9F 00 54 18 03 E4 78 D0 D0 2F 28 00        imadd            r12.cache, u57, r15.discard, r13.discard
 402: 9F 00 54 10 03 E8 40 C0 D0 2F 28 00        imadd            r8.cache, u58, r8.discard, r12.discard
 40e: 9F 00 54 10 03 EC 48 80 D0 2F 28 00        imadd            r8.cache, u59, r9.discard, r8.discard
 41a: 9F 00 54 12 03 F0 50 80 D0 2F 28 00        imadd            r9.cache, u60, r10.discard, r8.discard
 426: 9F 00 54 10 03 F4 58 90 D0 2F 28 00        imadd            r8.cache, u61, r11.discard, r9.discard
 432: 9F 00 54 08 03 F8 20 80 D0 2F 28 00        imadd            r4.cache, u62, r4.discard, r8.discard
 43e: 9F 00 54 08 03 FC 28 40 D0 2F 28 00        imadd            r4.cache, u63, r5.discard, r4.discard
 44a: 9F 00 54 0A 03 00 31 40 D0 2F 28 00        imadd            r5.cache, u64, r6.discard, r4.discard
 456: 9F 00 54 08 03 04 39 50 D0 2F 28 00        imadd            r4.cache, u65, r7.discard, r5.discard
 462: 9F 20 54 00 03 08 01 40 D0 2F 28 00        imadd            r0.cache, u66, r0.discard, r4.discard, wait 1
 46e: 9F 00 54 00 03 0C 09 00 D0 2F 28 00        imadd            r0.cache, u67, r1.discard, r0.discard
 47a: 9F 00 54 02 03 10 11 00 D0 2F 28 00        imadd            r1.cache, u68, r2.discard, r0.discard
 486: 9F 00 54 00 02 14 19 10 D0 2F 28 00        imadd            r0, u69, r3.discard, r1.discard
 492: E7 00 54 00 02 40 21 00 11 00 00 90 11 00  device_store     0, i32, x, r0, u4_u5, r64.discard, lsl 2, 0, 1, 1
 4a0: 0E 00 00 00                                stop             
